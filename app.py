import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, redirect, request
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from nltk.tokenize import word_tokenize
import fitz  # PyMuPDF

# Initialize Flask app
app = Flask(__name__)

# Load job descriptions and skills dataset
job_desc_df = pd.read_csv('indeed_dataset.csv', encoding="latin-1")
skills_df = pd.read_csv('skills1.csv')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in word_tokenize(text) if word.isalpha()])  # Remove non-alphabetic characters
    return text

# Extract skills from job descriptions
def extract_skills(text, skills_df):
    skills = set()
    text_lower = text.lower()
    for skill in skills_df['skills']:
        skill_lower = skill.lower()
        if skill_lower in text_lower:
            skills.add(skill_lower)
    return list(skills)

# Function to get vector representation of a text
def get_text_vector(text, model):
    tokens = word_tokenize(text)
    vector = np.zeros(model.vector_size)
    count = 0
    for token in tokens:
        if token in model.wv.index_to_key:
            vector += model.wv[token]
            count += 1
    if count != 0:
        vector /= count
    return vector

def extract_text_from_pdf(file_path):
    text = ''
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Create a list of skills extracted from job descriptions
all_skills = []
for text in job_desc_df['description']:
    skills = extract_skills(text, skills_df)
    all_skills.extend(skills)

# Train Word2Vec model on extracted skills
word2vec_model_skills = Word2Vec([all_skills], min_count=1)

# Cluster job descriptions based on skill vectors
job_desc_df['vector'] = job_desc_df['description'].apply(lambda x: get_text_vector(x, word2vec_model_skills))
X = np.array(job_desc_df['vector'].tolist())
clustering = AgglomerativeClustering(n_clusters=6)
job_desc_df['cluster'] = clustering.fit_predict(X)

# Define Decision Tree classifier with pipeline
tree_clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())
X_train, X_test, y_train, y_test = train_test_split(X, job_desc_df['cluster'], test_size=0.25, random_state=42)

# Train the Decision Tree classifier
tree_clf.fit(X_train, y_train)

# Define Flask routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home():
    return redirect('/')

@app.route('/submit', methods=['POST'])
def submit_data():
    if request.method == 'POST':
        f = request.files['userfile']
        f.save(os.path.join(app.instance_path, 'resume_files', f.filename))

        # Extract skills from resume
        resume_text = extract_text_from_pdf(os.path.join(app.instance_path, 'resume_files', f.filename))
        resume_text = preprocess_text(resume_text)
        resume_skills = extract_skills(resume_text, skills_df)

        print(resume_skills)
        # Join the list of skills into a single string
        resume_skills_text = ' '.join(resume_skills)

        # Predict cluster for resume
        predicted_cluster = tree_clf.predict([get_text_vector(resume_skills_text, word2vec_model_skills)])[0]

        recommended_jobs = job_desc_df[job_desc_df['cluster'] == predicted_cluster]
        
        # Compute Euclidean distances between resume skills vector and job descriptions vectors
        euclidean_distances = []
        for _, row in recommended_jobs.iterrows():
            euclidean_distance = euclidean(get_text_vector(resume_skills_text, word2vec_model_skills), row['vector'])
            euclidean_distances.append(euclidean_distance)

        recommended_jobs['euclidean_distance'] = euclidean_distances

        # Sort recommended jobs based on similarity score
        recommended_jobs = recommended_jobs.sort_values(by='euclidean_distance', ascending=True)

        # Remove duplicate job descriptions
        recommended_jobs = recommended_jobs.drop_duplicates(subset=['description'])

        # Get top 10 recommended jobs
        top_10_jobs = recommended_jobs[['title', 'company', 'link']].head(10)

        return render_template('index.html', recommended_jobs=top_10_jobs,
                               column_names=['title', 'company', 'link'],
                               row_data=list(top_10_jobs.values.tolist()),
                               link_column="link", zip=zip)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
