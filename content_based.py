"""
Content-Based Filtering Module for Movie Recommendation System.

This module implements:
- Movie feature extraction (genres, year, ratings)
- Movie-movie similarity computation
- User profile building
- Content-based recommendations
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class ContentBasedFiltering:
    """
    Content-Based Filtering Recommender.
    
    Recommends movies based on content similarity to user preferences.
    """
    
    GENRE_COLUMNS = [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western'
    ]
    
    def __init__(self, use_genres: bool = True, use_ratings: bool = True,
                 use_year: bool = True, genre_weight: float = 1.0,
                 rating_weight: float = 0.3, year_weight: float = 0.1):
        """
        Initialize Content-Based Filtering.
        
        Args:
            use_genres: Include genre features
            use_ratings: Include rating statistics
            use_year: Include release year
            genre_weight: Weight for genre features
            rating_weight: Weight for rating features
            year_weight: Weight for year feature
        """
        self.use_genres = use_genres
        self.use_ratings = use_ratings
        self.use_year = use_year
        self.genre_weight = genre_weight
        self.rating_weight = rating_weight
        self.year_weight = year_weight
        
        self.movies_df = None
        self.ratings_df = None
        self.movie_features = None
        self.movie_similarity = None
        self.movie_ids = None
        self.movie_id_to_idx = None
        self.is_fitted = False
        
        self.scaler = MinMaxScaler()
        
    def fit(self, movies_df: pd.DataFrame, 
            ratings_df: pd.DataFrame = None) -> 'ContentBasedFiltering':
        """
        Fit the content-based model.
        
        Args:
            movies_df: Movies DataFrame with genre columns
            ratings_df: Ratings DataFrame (optional, for user profiles)
            
        Returns:
            self
        """
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy() if ratings_df is not None else None
        
        # Get movie IDs
        self.movie_ids = self.movies_df['movie_id'].tolist()
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        
        # Build feature matrix
        self._build_feature_matrix()
        
        # Compute movie similarity
        self.movie_similarity = cosine_similarity(self.movie_features)
        
        self.is_fitted = True
        return self
    
    def _build_feature_matrix(self) -> None:
        """Build the movie feature matrix."""
        features_list = []
        
        # Genre features
        if self.use_genres:
            # Check which genre columns exist
            available_genres = [g for g in self.GENRE_COLUMNS if g in self.movies_df.columns]
            
            if available_genres:
                genre_features = self.movies_df[available_genres].fillna(0).values
                genre_features = genre_features * self.genre_weight
                features_list.append(genre_features)
        
        # Rating statistics
        if self.use_ratings and 'avg_rating' in self.movies_df.columns:
            rating_features = self.movies_df[['avg_rating', 'rating_count']].fillna(0).values
            rating_features = self.scaler.fit_transform(rating_features)
            rating_features = rating_features * self.rating_weight
            features_list.append(rating_features)
        
        # Year feature
        if self.use_year and 'year' in self.movies_df.columns:
            year_features = self.movies_df[['year']].fillna(1990).values
            year_features = self.scaler.fit_transform(year_features)
            year_features = year_features * self.year_weight
            features_list.append(year_features)
        
        # Combine all features
        if features_list:
            self.movie_features = np.hstack(features_list)
        else:
            # Fallback: use one-hot encoding of movie IDs
            n_movies = len(self.movie_ids)
            self.movie_features = np.eye(n_movies)
    
    def get_similar_movies(self, movie_id: int, n: int = 10) -> list:
        """
        Get movies similar to a given movie.
        
        Args:
            movie_id: Target movie ID
            n: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if movie_id not in self.movie_id_to_idx:
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        similarities = self.movie_similarity[movie_idx]
        
        # Get top n similar movies (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        
        similar_movies = [
            (self.movie_ids[idx], similarities[idx])
            for idx in similar_indices
        ]
        
        return similar_movies
    
    def build_user_profile(self, user_id: int, 
                           min_rating: float = 3.5) -> np.ndarray:
        """
        Build a user preference profile based on their ratings.
        
        Args:
            user_id: User ID
            min_rating: Minimum rating to consider as "liked"
            
        Returns:
            User profile vector
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data required for user profiles")
        
        # Get user's highly rated movies
        user_ratings = self.ratings_df[
            (self.ratings_df['user_id'] == user_id) & 
            (self.ratings_df['rating'] >= min_rating)
        ]
        
        if len(user_ratings) == 0:
            # Return average profile if no high ratings
            return self.movie_features.mean(axis=0)
        
        # Weight by rating
        profile = np.zeros(self.movie_features.shape[1])
        total_weight = 0
        
        for _, row in user_ratings.iterrows():
            movie_id = row['movie_id']
            rating = row['rating']
            
            if movie_id in self.movie_id_to_idx:
                movie_idx = self.movie_id_to_idx[movie_id]
                # Weight by rating (higher rating = more weight)
                weight = rating - min_rating + 1
                profile += self.movie_features[movie_idx] * weight
                total_weight += weight
        
        if total_weight > 0:
            profile /= total_weight
        
        return profile
    
    def recommend(self, user_id: int, n_recommendations: int = 10,
                  min_rating: float = 3.5, exclude_rated: bool = True) -> list:
        """
        Generate recommendations for a user based on their preferences.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            min_rating: Minimum rating threshold for profile building
            exclude_rated: Exclude already rated movies
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Build user profile
        user_profile = self.build_user_profile(user_id, min_rating)
        
        # Calculate similarity between user profile and all movies
        similarities = cosine_similarity(
            user_profile.reshape(1, -1), 
            self.movie_features
        )[0]
        
        # Get movies user has rated
        if exclude_rated and self.ratings_df is not None:
            rated_movies = set(
                self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id']
            )
        else:
            rated_movies = set()
        
        # Collect recommendations
        recommendations = []
        
        for movie_idx, movie_id in enumerate(self.movie_ids):
            if movie_id in rated_movies:
                continue
            
            recommendations.append((movie_id, similarities[movie_idx]))
        
        # Sort by similarity
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def recommend_from_movie(self, movie_id: int, n_recommendations: int = 10,
                              user_id: int = None, exclude_rated: bool = True) -> list:
        """
        Recommend movies similar to a specific movie.
        
        Args:
            movie_id: Reference movie ID
            n_recommendations: Number of recommendations
            user_id: Optional user ID to exclude their rated movies
            exclude_rated: Exclude user's rated movies
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        similar = self.get_similar_movies(movie_id, n=n_recommendations + 50)
        
        if user_id is not None and exclude_rated and self.ratings_df is not None:
            rated_movies = set(
                self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id']
            )
            similar = [(mid, score) for mid, score in similar if mid not in rated_movies]
        
        return similar[:n_recommendations]
    
    def get_user_genre_preferences(self, user_id: int) -> dict:
        """
        Get user's genre preferences based on their ratings.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary of genre->preference_score
        """
        if self.ratings_df is None:
            return {}
        
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        genre_scores = {genre: 0.0 for genre in self.GENRE_COLUMNS}
        genre_counts = {genre: 0 for genre in self.GENRE_COLUMNS}
        
        for _, row in user_ratings.iterrows():
            movie_id = row['movie_id']
            rating = row['rating']
            
            movie_data = self.movies_df[self.movies_df['movie_id'] == movie_id]
            if len(movie_data) == 0:
                continue
            
            movie_row = movie_data.iloc[0]
            
            for genre in self.GENRE_COLUMNS:
                if genre in movie_row and movie_row[genre] == 1:
                    genre_scores[genre] += rating
                    genre_counts[genre] += 1
        
        # Calculate average rating per genre
        genre_preferences = {}
        for genre in self.GENRE_COLUMNS:
            if genre_counts[genre] > 0:
                genre_preferences[genre] = genre_scores[genre] / genre_counts[genre]
        
        # Sort by preference
        genre_preferences = dict(
            sorted(genre_preferences.items(), key=lambda x: x[1], reverse=True)
        )
        
        return genre_preferences
    
    def get_movie_info(self, movie_id: int) -> dict:
        """
        Get information about a specific movie.
        
        Args:
            movie_id: Movie ID
            
        Returns:
            Dictionary with movie information
        """
        movie_data = self.movies_df[self.movies_df['movie_id'] == movie_id]
        
        if len(movie_data) == 0:
            return None
        
        movie = movie_data.iloc[0]
        
        # Get genres
        genres = [g for g in self.GENRE_COLUMNS if g in movie and movie[g] == 1]
        
        info = {
            'movie_id': movie_id,
            'title': movie.get('title', 'Unknown'),
            'genres': genres,
            'year': movie.get('year', 'Unknown'),
            'avg_rating': movie.get('avg_rating', 0),
            'rating_count': movie.get('rating_count', 0)
        }
        
        return info


class TFIDFContentRecommender:
    """
    TF-IDF based content recommender for text descriptions.
    
    Uses movie descriptions/plots if available.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Initialize TF-IDF recommender.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: N-gram range for TF-IDF
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        self.tfidf_matrix = None
        self.movie_ids = None
        self.movie_id_to_idx = None
        self.is_fitted = False
    
    def fit(self, movies_df: pd.DataFrame, 
            text_column: str = 'description') -> 'TFIDFContentRecommender':
        """
        Fit the TF-IDF model.
        
        Args:
            movies_df: Movies DataFrame
            text_column: Column containing text descriptions
            
        Returns:
            self
        """
        if text_column not in movies_df.columns:
            # Create text from available columns
            text_data = self._create_text_from_features(movies_df)
        else:
            text_data = movies_df[text_column].fillna('')
        
        self.movie_ids = movies_df['movie_id'].tolist()
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        
        self.tfidf_matrix = self.tfidf.fit_transform(text_data)
        self.is_fitted = True
        
        return self
    
    def _create_text_from_features(self, movies_df: pd.DataFrame) -> pd.Series:
        """Create text representation from movie features."""
        genre_cols = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
            'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western'
        ]
        
        texts = []
        
        for _, row in movies_df.iterrows():
            text_parts = []
            
            # Add title
            if 'title' in row:
                text_parts.append(str(row['title']))
            
            # Add genres
            for genre in genre_cols:
                if genre in row and row[genre] == 1:
                    text_parts.append(genre)
            
            texts.append(' '.join(text_parts))
        
        return pd.Series(texts)
    
    def get_similar_movies(self, movie_id: int, n: int = 10) -> list:
        """
        Get movies similar based on TF-IDF similarity.
        
        Args:
            movie_id: Target movie ID
            n: Number of similar movies
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if movie_id not in self.movie_id_to_idx:
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        
        # Compute similarity with this movie
        movie_vec = self.tfidf_matrix[movie_idx]
        similarities = cosine_similarity(movie_vec, self.tfidf_matrix).flatten()
        
        # Get top similar movies
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        
        similar_movies = [
            (self.movie_ids[idx], similarities[idx])
            for idx in similar_indices
        ]
        
        return similar_movies


if __name__ == "__main__":
    # Test content-based filtering
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.preprocessing import load_and_preprocess_data
    
    print("Loading data...")
    preprocessor, ratings, movies, users = load_and_preprocess_data()
    
    print("\nTesting Content-Based Filtering...")
    cb = ContentBasedFiltering()
    cb.fit(movies, ratings)
    
    # Test similar movies
    test_movie = movies['movie_id'].iloc[0]
    print(f"\nMovies similar to '{movies[movies['movie_id'] == test_movie]['title'].values[0]}':")
    similar = cb.get_similar_movies(test_movie, n=5)
    for movie_id, score in similar:
        title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"  {title}: {score:.3f}")
    
    # Test user recommendations
    test_user = ratings['user_id'].iloc[0]
    print(f"\nContent-based recommendations for user {test_user}:")
    recs = cb.recommend(test_user, n_recommendations=5)
    for movie_id, score in recs:
        title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"  {title}: {score:.3f}")
    
    # Test user genre preferences
    print(f"\nGenre preferences for user {test_user}:")
    prefs = cb.get_user_genre_preferences(test_user)
    for genre, score in list(prefs.items())[:5]:
        print(f"  {genre}: {score:.2f}")
