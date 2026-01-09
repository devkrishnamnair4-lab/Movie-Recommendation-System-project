"""
Collaborative Filtering Module for Movie Recommendation System.

This module implements:
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- SVD Matrix Factorization
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import surprise, handle if not installed
try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import cross_validate
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("Warning: scikit-surprise not installed. SVD recommender will not be available.")


class UserBasedCF:
    """
    User-Based Collaborative Filtering.
    
    Recommends movies that similar users have liked.
    """
    
    def __init__(self, k_neighbors: int = 50, min_common_items: int = 3):
        """
        Initialize User-Based CF.
        
        Args:
            k_neighbors: Number of similar users to consider
            min_common_items: Minimum items in common to consider similarity
        """
        self.k_neighbors = k_neighbors
        self.min_common_items = min_common_items
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_mean_ratings = None
        self.user_ids = None
        self.movie_ids = None
        self.is_fitted = False
        
    def fit(self, ratings_df: pd.DataFrame) -> 'UserBasedCF':
        """
        Fit the model with rating data.
        
        Args:
            ratings_df: DataFrame with user_id, movie_id, rating columns
            
        Returns:
            self
        """
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.movie_ids = self.user_item_matrix.columns.tolist()
        
        # Calculate mean ratings per user (excluding zeros)
        matrix_values = self.user_item_matrix.values.copy()
        matrix_mask = matrix_values > 0
        
        self.user_mean_ratings = {}
        for i, user_id in enumerate(self.user_ids):
            user_ratings = matrix_values[i][matrix_mask[i]]
            self.user_mean_ratings[user_id] = (
                user_ratings.mean() if len(user_ratings) > 0 else 0
            )
        
        # Calculate user similarity using cosine similarity
        self.user_similarity = cosine_similarity(matrix_values)
        
        self.is_fitted = True
        return self
    
    def get_similar_users(self, user_id: int, k: int = None) -> list:
        """
        Get k most similar users.
        
        Args:
            user_id: Target user ID
            k: Number of similar users (default: self.k_neighbors)
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if k is None:
            k = self.k_neighbors
        
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        similarities = self.user_similarity[user_idx]
        
        # Get top k similar users (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:k+1]
        
        similar_users = [
            (self.user_ids[idx], similarities[idx])
            for idx in similar_indices
            if similarities[idx] > 0
        ]
        
        return similar_users
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_ids:
            return self.user_mean_ratings.get(user_id, 3.0)
        
        if movie_id not in self.movie_ids:
            return self.user_mean_ratings.get(user_id, 3.0)
        
        user_idx = self.user_ids.index(user_id)
        movie_idx = self.movie_ids.index(movie_id)
        
        # Get similar users who rated this movie
        similarities = self.user_similarity[user_idx]
        matrix = self.user_item_matrix.values
        
        # Find users who rated this movie
        rated_mask = matrix[:, movie_idx] > 0
        rated_users_idx = np.where(rated_mask)[0]
        
        if len(rated_users_idx) == 0:
            return self.user_mean_ratings.get(user_id, 3.0)
        
        # Get similarities for users who rated this movie
        user_sims = similarities[rated_users_idx]
        user_ratings = matrix[rated_users_idx, movie_idx]
        
        # Select top k similar users
        top_k_idx = np.argsort(user_sims)[::-1][:self.k_neighbors]
        
        top_sims = user_sims[top_k_idx]
        top_ratings = user_ratings[top_k_idx]
        
        # Weighted average prediction
        if np.sum(np.abs(top_sims)) == 0:
            return self.user_mean_ratings.get(user_id, 3.0)
        
        prediction = np.sum(top_sims * top_ratings) / np.sum(np.abs(top_sims))
        
        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
    
    def recommend(self, user_id: int, n_recommendations: int = 10,
                  exclude_rated: bool = True) -> list:
        """
        Generate movie recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_ids:
            # Cold start: return popular movies
            return self._get_popular_movies(n_recommendations)
        
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix.values[user_idx]
        
        predictions = []
        
        for movie_idx, movie_id in enumerate(self.movie_ids):
            # Skip if already rated
            if exclude_rated and user_ratings[movie_idx] > 0:
                continue
            
            predicted_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def _get_popular_movies(self, n: int) -> list:
        """Get most popular movies for cold start."""
        movie_ratings = self.user_item_matrix.values
        avg_ratings = []
        
        for movie_idx, movie_id in enumerate(self.movie_ids):
            ratings = movie_ratings[:, movie_idx]
            non_zero = ratings[ratings > 0]
            if len(non_zero) > 0:
                avg_ratings.append((movie_id, non_zero.mean(), len(non_zero)))
        
        # Sort by number of ratings (popularity), then by average
        avg_ratings.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        return [(movie_id, avg) for movie_id, avg, _ in avg_ratings[:n]]


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering.
    
    Recommends movies similar to what the user has liked.
    """
    
    def __init__(self, k_neighbors: int = 50):
        """
        Initialize Item-Based CF.
        
        Args:
            k_neighbors: Number of similar items to consider
        """
        self.k_neighbors = k_neighbors
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_ids = None
        self.movie_ids = None
        self.is_fitted = False
        
    def fit(self, ratings_df: pd.DataFrame) -> 'ItemBasedCF':
        """
        Fit the model with rating data.
        
        Args:
            ratings_df: DataFrame with user_id, movie_id, rating columns
            
        Returns:
            self
        """
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.movie_ids = self.user_item_matrix.columns.tolist()
        
        # Calculate item similarity using cosine similarity
        # Transpose so items are rows
        item_matrix = self.user_item_matrix.values.T
        self.item_similarity = cosine_similarity(item_matrix)
        
        self.is_fitted = True
        return self
    
    def get_similar_items(self, movie_id: int, k: int = None) -> list:
        """
        Get k most similar movies.
        
        Args:
            movie_id: Target movie ID
            k: Number of similar items (default: self.k_neighbors)
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if k is None:
            k = self.k_neighbors
        
        if movie_id not in self.movie_ids:
            return []
        
        movie_idx = self.movie_ids.index(movie_id)
        similarities = self.item_similarity[movie_idx]
        
        # Get top k similar items (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:k+1]
        
        similar_items = [
            (self.movie_ids[idx], similarities[idx])
            for idx in similar_indices
            if similarities[idx] > 0
        ]
        
        return similar_items
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_ids or movie_id not in self.movie_ids:
            return 3.0  # Default rating
        
        user_idx = self.user_ids.index(user_id)
        movie_idx = self.movie_ids.index(movie_id)
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.values[user_idx]
        
        # Get similarities for this movie
        similarities = self.item_similarity[movie_idx]
        
        # Find items user has rated
        rated_mask = user_ratings > 0
        
        if not rated_mask.any():
            return 3.0
        
        # Get similarities and ratings for rated items
        item_sims = similarities[rated_mask]
        item_ratings = user_ratings[rated_mask]
        
        # Select top k similar items
        top_k_idx = np.argsort(item_sims)[::-1][:self.k_neighbors]
        
        top_sims = item_sims[top_k_idx]
        top_ratings = item_ratings[top_k_idx]
        
        # Weighted average prediction
        if np.sum(np.abs(top_sims)) == 0:
            return 3.0
        
        prediction = np.sum(top_sims * top_ratings) / np.sum(np.abs(top_sims))
        
        return np.clip(prediction, 1, 5)
    
    def recommend(self, user_id: int, n_recommendations: int = 10,
                  exclude_rated: bool = True) -> list:
        """
        Generate movie recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_ids:
            return self._get_popular_movies(n_recommendations)
        
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix.values[user_idx]
        
        predictions = []
        
        for movie_idx, movie_id in enumerate(self.movie_ids):
            if exclude_rated and user_ratings[movie_idx] > 0:
                continue
            
            predicted_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def _get_popular_movies(self, n: int) -> list:
        """Get most popular movies for cold start."""
        movie_ratings = self.user_item_matrix.values
        avg_ratings = []
        
        for movie_idx, movie_id in enumerate(self.movie_ids):
            ratings = movie_ratings[:, movie_idx]
            non_zero = ratings[ratings > 0]
            if len(non_zero) > 0:
                avg_ratings.append((movie_id, non_zero.mean(), len(non_zero)))
        
        avg_ratings.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        return [(movie_id, avg) for movie_id, avg, _ in avg_ratings[:n]]


class SVDRecommender:
    """
    SVD-based Matrix Factorization Recommender.
    
    Uses scikit-surprise library for SVD implementation.
    """
    
    def __init__(self, n_factors: int = 100, n_epochs: int = 20,
                 lr_all: float = 0.005, reg_all: float = 0.02):
        """
        Initialize SVD Recommender.
        
        Args:
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            lr_all: Learning rate
            reg_all: Regularization term
        """
        if not SURPRISE_AVAILABLE:
            raise ImportError("scikit-surprise is required for SVD. Install with: pip install scikit-surprise")
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        
        self.trainset = None
        self.ratings_df = None
        self.user_ids = None
        self.movie_ids = None
        self.is_fitted = False
        
    def fit(self, ratings_df: pd.DataFrame) -> 'SVDRecommender':
        """
        Fit the SVD model.
        
        Args:
            ratings_df: DataFrame with user_id, movie_id, rating columns
            
        Returns:
            self
        """
        self.ratings_df = ratings_df.copy()
        self.user_ids = ratings_df['user_id'].unique().tolist()
        self.movie_ids = ratings_df['movie_id'].unique().tolist()
        
        # Create surprise dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings_df[['user_id', 'movie_id', 'rating']], 
            reader
        )
        
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)
        
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est
    
    def recommend(self, user_id: int, n_recommendations: int = 10,
                  exclude_rated: bool = True) -> list:
        """
        Generate movie recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get movies user has already rated
        if exclude_rated:
            rated_movies = set(
                self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id']
            )
        else:
            rated_movies = set()
        
        predictions = []
        
        for movie_id in self.movie_ids:
            if movie_id in rated_movies:
                continue
            
            predicted_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def cross_validate(self, ratings_df: pd.DataFrame, 
                       cv: int = 5) -> dict:
        """
        Perform cross-validation on the model.
        
        Args:
            ratings_df: Ratings DataFrame
            cv: Number of folds
            
        Returns:
            Cross-validation results
        """
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings_df[['user_id', 'movie_id', 'rating']], 
            reader
        )
        
        results = cross_validate(
            self.model, data, 
            measures=['RMSE', 'MAE'], 
            cv=cv, 
            verbose=True
        )
        
        return {
            'rmse_mean': results['test_rmse'].mean(),
            'rmse_std': results['test_rmse'].std(),
            'mae_mean': results['test_mae'].mean(),
            'mae_std': results['test_mae'].std()
        }


def create_recommender(algorithm: str = 'user_based', **kwargs):
    """
    Factory function to create recommender instances.
    
    Args:
        algorithm: 'user_based', 'item_based', or 'svd'
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Recommender instance
    """
    algorithms = {
        'user_based': UserBasedCF,
        'item_based': ItemBasedCF,
        'svd': SVDRecommender
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(algorithms.keys())}")
    
    return algorithms[algorithm](**kwargs)


if __name__ == "__main__":
    # Test collaborative filtering
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.preprocessing import load_and_preprocess_data
    
    print("Loading data...")
    preprocessor, ratings, movies, users = load_and_preprocess_data()
    
    print("\nTesting User-Based CF...")
    user_cf = UserBasedCF(k_neighbors=50)
    user_cf.fit(ratings)
    
    test_user = ratings['user_id'].iloc[0]
    recs = user_cf.recommend(test_user, n_recommendations=5)
    print(f"Recommendations for user {test_user}:")
    for movie_id, rating in recs:
        title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"  {title}: {rating:.2f}")
    
    print("\nTesting Item-Based CF...")
    item_cf = ItemBasedCF(k_neighbors=50)
    item_cf.fit(ratings)
    
    recs = item_cf.recommend(test_user, n_recommendations=5)
    print(f"Recommendations for user {test_user}:")
    for movie_id, rating in recs:
        title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"  {title}: {rating:.2f}")
    
    if SURPRISE_AVAILABLE:
        print("\nTesting SVD...")
        svd = SVDRecommender(n_factors=50, n_epochs=10)
        svd.fit(ratings)
        
        recs = svd.recommend(test_user, n_recommendations=5)
        print(f"Recommendations for user {test_user}:")
        for movie_id, rating in recs:
            title = movies[movies['movie_id'] == movie_id]['title'].values[0]
            print(f"  {title}: {rating:.2f}")
