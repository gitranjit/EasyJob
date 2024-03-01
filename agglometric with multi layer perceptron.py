import re
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ''
    with open(pdf_file, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Function to extract skills from text using a skills dataset
def extract_skills(text, skills_list):
    tokens = nltk.word_tokenize(text)
    # Filter out stop words
    stop_words = set(stopwords.words('english'))
    skills = [token for token in tokens if token.lower() in skills_list and token.lower() not in stop_words]
    return skills

# Load skills data from CSV
skills_data = pd.read_csv('skills.csv', engine='python')

skills_list = set(skills_data['Skills'].str.lower().str.split().explode().tolist())

# Load job descriptions from Excel
job_data = pd.read_csv('indeed_data.csv')  # Update with your file name and path
job_descriptions = job_data['description'].tolist()

# Preprocess job descriptions
preprocessed_job_descriptions = [preprocess_text(desc) for desc in job_descriptions]

# Extract skills from job descriptions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_job_descriptions)
job_sequences = tokenizer.texts_to_sequences(preprocessed_job_descriptions)
max_len = max([len(seq) for seq in job_sequences])
job_sequences_padded = pad_sequences(job_sequences, maxlen=max_len)
vocab_size = len(tokenizer.word_index) + 1

# Cluster job descriptions using Agglomerative Clustering
num_clusters = 5  # You can adjust the number of clusters as needed
agglomerative = AgglomerativeClustering(n_clusters=num_clusters)
job_clusters = agglomerative.fit_predict(job_sequences_padded)

# Split labeled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(job_sequences_padded, job_clusters, test_size=0.2, random_state=42)

# Train a model to predict clusters
classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)
predicted_clusters = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_clusters)
print("Accuracy:", accuracy)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, predicted_clusters, average='weighted')
recall = recall_score(y_test, predicted_clusters, average='weighted')
f1 = f1_score(y_test, predicted_clusters, average='weighted')
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Extract skills from resume
resume_file = 'resume_yash.pdf'  # Replace with your PDF file path
resume_text = extract_text_from_pdf(resume_file)
preprocessed_resume = preprocess_text(resume_text)
resume_sequence = tokenizer.texts_to_sequences([preprocessed_resume])
resume_sequence_padded = pad_sequences(resume_sequence, maxlen=max_len)

# Assign resume to the cluster with the most similar job descriptions
predicted_cluster = classifier.predict(resume_sequence_padded)[0]

# Get job titles and URLs for the predicted cluster
predicted_jobs = job_data[job_clusters == predicted_cluster]

print("Predicted Cluster:", predicted_cluster)
print("Predicted Jobs:")
job_similarities = []

for idx, job in predicted_jobs.iterrows():
    job_description_sequence = tokenizer.texts_to_sequences([preprocess_text(job['description'])])
    job_description_sequence_padded = pad_sequences(job_description_sequence, maxlen=max_len)
    similarity = cosine_similarity(resume_sequence_padded, job_description_sequence_padded)[0][0]
    job_similarities.append((job['title'], job['link'], similarity))

# Sort jobs by cosine similarity and print top 5
top_5_jobs = sorted(job_similarities, key=lambda x: x[2], reverse=True)[:5]
print("Top 5 Jobs:")
for title, url, similarity in top_5_jobs:
   
    print("Job Title:", title)
    print("Job URL:", url)
    print("Cosine Similarity:", similarity)
    print()