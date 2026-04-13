import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import os

print("Starting Data Preprocessing...")

movies = pd.read_csv('ml-32m/movies.csv')
tags = pd.read_csv('ml-32m/tags.csv')

# Count the number of tags per movie to determine popularity
tags_count = tags.groupby('movieId').size().reset_index(name='tag_count')

# Group tags by movie
tags.dropna(subset=['tag'], inplace=True)
tags['tag'] = tags['tag'].str.lower()
grouped_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Merge with movies
movies = movies.merge(grouped_tags, on='movieId', how='left')
movies = movies.merge(tags_count, on='movieId', how='left')
movies['tag_count'] = movies['tag_count'].fillna(0)

# Replace NaN tags with empty string
movies['tag'] = movies['tag'].fillna('')

# Format genres (they are pipe separated, we want spaces)
movies['genres_str'] = movies['genres'].str.replace('|', ' ')

# Create NLP features by combining genres and tags
movies['text_features'] = movies['genres_str'] + " " + movies['tag']

# Filter out movies that have empty text features
movies = movies[movies['text_features'].str.strip() != '']

# Process all movies instead of chunking to satisfy the requirement
movies_with_tags = movies[movies['tag'] != '']
    
print(f"Dataset selected: {len(movies_with_tags)} movies for processing. Top movie: {movies_with_tags.iloc[0]['title']}")

movies_with_tags = movies_with_tags.reset_index(drop=True)

# TF-IDF Vectorization
print("Running TF-IDF...")
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_with_tags['text_features'])

# PCA Dimensionality Reduction
print("Running PCA...")
pca = PCA(n_components=100, random_state=42)
pca_features = pca.fit_transform(tfidf_matrix.toarray())

# K-Means Clustering
print("Running K-Means...")
kmeans = KMeans(n_clusters=25, random_state=42, n_init=10)
movies_with_tags['cluster'] = kmeans.fit_predict(pca_features)

print("Calculating Metrics...")
sil_score = silhouette_score(pca_features, movies_with_tags['cluster'])
print(f"Silhouette Score: {sil_score:.4f}")

# Save models and data
print("Saving artifacts...")
movies_with_tags.to_parquet('processed_movies.parquet', index=False)

with open('tfidf_model.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

print("Data Preprocessing and Modeling Complete!")
