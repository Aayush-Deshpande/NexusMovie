import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from sklearn.decomposition import PCA

print("Starting Advanced EDA Process...")

os.makedirs('eda_assets', exist_ok=True)
stats = {}

print("Loading Data...")
# Load datasets
movies = pd.read_csv('ml-32m/movies.csv')
tags = pd.read_csv('ml-32m/tags.csv')
# For EDA purposes on 32M rows, load a random subset of ratings if memory is an issue, 
# but simply loading usecols is usually fine.
ratings = pd.read_csv('ml-32m/ratings.csv', usecols=['userId', 'movieId', 'rating'])

# --- 1. Missing Values Analysis ---
print("Analyzing Missing Values...")
stats['total_users'] = ratings['userId'].nunique()
stats['total_movies'] = movies['movieId'].nunique()
stats['total_ratings'] = len(ratings)

stats['missing_movies'] = int(movies.isnull().sum().sum())
stats['missing_tags'] = int(tags.isnull().sum().sum())
stats['missing_ratings'] = int(ratings.isnull().sum().sum())

# Sparsity calculation
total_possible = stats['total_users'] * stats['total_movies']
stats['sparsity'] = round((1 - (stats['total_ratings'] / total_possible)) * 100, 4)

# --- 2. Advanced Heatmap (User-Item Interactions) ---
print("Generating Interaction Heatmap...")
# Take top 50 active users and top 50 popular movies for the heatmap
top_users = ratings['userId'].value_counts().head(30).index
top_movies = ratings['movieId'].value_counts().head(30).index
subset = ratings[(ratings['userId'].isin(top_users)) & (ratings['movieId'].isin(top_movies))]

pivot = subset.pivot(index='userId', columns='movieId', values='rating')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot, cmap='mako', cbar_kws={'label': 'Rating'}, linewidths=.5, linecolor='#0f172a')
plt.title('User-Item Interaction Heatmap (Top Users vs Top Movies)', color='white')
plt.xlabel('Movie ID', color='white')
plt.ylabel('User ID', color='white')
plt.gcf().set_facecolor('#0f172a')
plt.gca().tick_params(colors='white', bottom=False, left=False)
plt.savefig('eda_assets/heatmap.png', bbox_inches='tight', transparent=True)
plt.close()

# --- 3. PCA Scatter Plot (Clusters) ---
print("Generating PCA Scatter Plot...")
try:
    processed_movies = pd.read_parquet('processed_movies.parquet')
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    # We fit a 2D PCA just for visualization
    pca_2d = PCA(n_components=2, random_state=42)
    components_2d = pca_2d.fit_transform(tfidf_matrix.toarray())
    
    plt.figure(figsize=(10, 8))
    # We use a subset for plotting to avoid massive overlap blob (e.g., sample 5000)
    scatter_df = pd.DataFrame({
        'PC1': components_2d[:, 0],
        'PC2': components_2d[:, 1],
        'Cluster': processed_movies['cluster']
    }).sample(min(5000, len(processed_movies)), random_state=42)
    
    sns.scatterplot(data=scatter_df, x='PC1', y='PC2', hue='Cluster', palette='tab20', s=20, alpha=0.7, legend=False)
    plt.title('PCA 2D Cluster Visualization (TF-IDF Vector Space)', color='white')
    plt.gcf().set_facecolor('#0f172a')
    plt.gca().set_facecolor('#1e293b')
    plt.gca().tick_params(colors='white')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('gray')
    plt.savefig('eda_assets/pca_scatter.png', bbox_inches='tight', transparent=True)
    plt.close()
except Exception as e:
    print(f"Skipping PCA Plot (models/parquet missing): {e}")

# --- 4. Basic Distributions ---
print("Generating Rating Distribution...")
plt.figure(figsize=(10, 6))
sns.histplot(ratings['rating'], bins=10, kde=False, color='#38bdf8')
plt.title('Distribution of Movie Ratings', fontsize=16, color='white')
plt.xlabel('Rating', fontsize=12, color='white')
plt.ylabel('Frequency', fontsize=12, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.gcf().set_facecolor('#0f172a')
plt.gca().set_facecolor('#0f172a')
plt.gca().tick_params(colors='white')
for spine in plt.gca().spines.values():
    spine.set_edgecolor('gray')
plt.savefig('eda_assets/rating_distribution.png', bbox_inches='tight')
plt.close()

print("Generating Genre Frequency...")
movies['genres_list'] = movies['genres'].str.split('|')
exploded_genres = movies.explode('genres_list')
genre_counts = exploded_genres['genres_list'].value_counts()

plt.figure(figsize=(12, 8))
sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette='crest', legend=False)
plt.title('Movie Frequency by Genre', fontsize=16, color='white')
plt.xlabel('Number of Movies', fontsize=12, color='white')
plt.ylabel('Genre', fontsize=12, color='white')
plt.gcf().set_facecolor('#0f172a')
plt.gca().set_facecolor('#0f172a')
plt.gca().tick_params(colors='white')
for spine in plt.gca().spines.values():
    spine.set_edgecolor('gray')
plt.savefig('eda_assets/genre_frequency.png', bbox_inches='tight')
plt.close()

# --- 5. Outliers (Z-Score & IQR) ---
print("Calculating Outliers...")
user_counts = ratings.groupby('userId').size()
Q1 = user_counts.quantile(0.25)
Q3 = user_counts.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

stats['outlier_users_iqr'] = int(len(user_counts[(user_counts < lower_bound) | (user_counts > upper_bound)]))
stats['median_ratings_per_user'] = float(user_counts.median())

mean_rating = ratings['rating'].mean()
std_rating = ratings['rating'].std()
stats['mean_rating'] = float(mean_rating)
stats['std_rating'] = float(std_rating)

ratings['z_score'] = (ratings['rating'] - mean_rating) / std_rating
stats['extreme_ratings_count'] = int(len(ratings[(ratings['z_score'] > 3) | (ratings['z_score'] < -3)]))

with open('eda_assets/stats.json', 'w') as f:
    json.dump(stats, f)

print("EDA Process Complete! Assets saved in eda_assets/")
