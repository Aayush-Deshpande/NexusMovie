import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class HybridRecommender:
    def __init__(self, data_path, tfidf_path, pca_path, kmeans_path, matrix_path):
        self.movies = pd.read_parquet(data_path)
        with open(tfidf_path, 'rb') as f:
            self.tfidf = pickle.load(f)
        with open(pca_path, 'rb') as f:
            self.pca = pickle.load(f)
        with open(kmeans_path, 'rb') as f:
            self.kmeans = pickle.load(f)
        with open(matrix_path, 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
            
    def get_nlp_based_recommendations(self, natural_language_query, top_n=10):
        # Transform query using NLP
        query_vec = self.tfidf.transform([natural_language_query])
        
        # Calculate Cosine Similarity with all movies
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top movies
        top_indices = sim_scores.argsort()[-top_n:][::-1]
        
        # We should only return recommendations that have at least some similarity
        non_zero_indices = [idx for idx in top_indices if sim_scores[idx] > 0]
        if not non_zero_indices:
            return pd.DataFrame()
            
        recommended_movies = self.movies.iloc[non_zero_indices].copy()
        recommended_movies['similarity'] = sim_scores[non_zero_indices]
        return recommended_movies[['title', 'genres', 'similarity', 'cluster']].head(top_n)

    def get_preference_based_recommendations(self, loved_titles, liked_titles, disliked_titles, top_n=10):
        # 1. Build User Profile Vector
        # +2 for Loved, +1 for Liked, -1 for Disliked
        user_profile_vec = np.zeros(self.tfidf_matrix.shape[1])
        weight_sum = 0
        
        def add_to_profile(titles, weight):
            nonlocal user_profile_vec, weight_sum
            for t in titles:
                movie_idx = self.movies[self.movies['title'] == t].index
                if len(movie_idx) > 0:
                    idx = movie_idx[0]
                    user_profile_vec += self.tfidf_matrix[idx].toarray().flatten() * weight
                    weight_sum += abs(weight)
                    
        add_to_profile(loved_titles, 2.0)
        add_to_profile(liked_titles, 1.0)
        add_to_profile(disliked_titles, -1.0)
        
        if weight_sum == 0:
            return pd.DataFrame()
            
        user_profile_vec = user_profile_vec / weight_sum
        user_profile_vec = user_profile_vec.reshape(1, -1)
        
        # Determine user's closest cluster via PCA
        user_pca = self.pca.transform(user_profile_vec)
        user_cluster = self.kmeans.predict(user_pca)[0]
        
        # Filter movies by this cluster or just do global similarity (global is better for fallback)
        # We will use cosine similarity over all movies, but prioritize cluster matches slightly
        
        sim_scores = cosine_similarity(user_profile_vec, self.tfidf_matrix).flatten()
        
        top_indices = sim_scores.argsort()[-top_n*2:][::-1]
        
        # Avoid already rated movies
        already_rated = set(loved_titles + liked_titles + disliked_titles)
        
        recommendations = []
        for idx in top_indices:
            movie = self.movies.iloc[idx]
            if movie['title'] not in already_rated:
                recommendations.append({
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'similarity': sim_scores[idx],
                    'cluster': movie['cluster'],
                    'cluster_match': movie['cluster'] == user_cluster
                })
                if len(recommendations) >= top_n:
                    break
                    
        return pd.DataFrame(recommendations)
