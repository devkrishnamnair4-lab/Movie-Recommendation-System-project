"""
Unit tests for the Movie Recommendation System.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPreprocessing:
    """Tests for data preprocessing module."""
    
    @pytest.fixture
    def sample_ratings(self):
        """Create sample ratings data."""
        return pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
            'movie_id': [1, 2, 3, 1, 4, 2, 3, 4, 5, 1],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 3.0, 4.0, 2.0, 5.0, 4.0],
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='D')
        })
    
    @pytest.fixture
    def sample_movies(self):
        """Create sample movies data."""
        return pd.DataFrame({
            'movie_id': [1, 2, 3, 4, 5],
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
            'Action': [1, 0, 1, 0, 1],
            'Comedy': [0, 1, 0, 1, 0],
            'Drama': [1, 1, 0, 0, 1],
            'avg_rating': [4.3, 3.5, 3.5, 3.5, 5.0],
            'rating_count': [3, 2, 2, 2, 1]
        })
    
    def test_ratings_structure(self, sample_ratings):
        """Test ratings data has correct columns."""
        required_cols = ['user_id', 'movie_id', 'rating']
        assert all(col in sample_ratings.columns for col in required_cols)
    
    def test_movies_structure(self, sample_movies):
        """Test movies data has correct columns."""
        required_cols = ['movie_id', 'title']
        assert all(col in sample_movies.columns for col in required_cols)
    
    def test_rating_range(self, sample_ratings):
        """Test ratings are in valid range."""
        assert sample_ratings['rating'].min() >= 1
        assert sample_ratings['rating'].max() <= 5
    
    def test_no_duplicate_ratings(self, sample_ratings):
        """Test no duplicate user-movie ratings."""
        duplicates = sample_ratings.duplicated(subset=['user_id', 'movie_id'])
        assert not duplicates.any()


class TestUserBasedCF:
    """Tests for User-Based Collaborative Filtering."""
    
    @pytest.fixture
    def sample_ratings(self):
        """Create sample ratings data."""
        return pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 4],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0]
        })
    
    def test_fit(self, sample_ratings):
        """Test model fitting."""
        from src.collaborative_filtering import UserBasedCF
        
        model = UserBasedCF(k_neighbors=2)
        model.fit(sample_ratings)
        
        assert model.is_fitted
        assert len(model.user_ids) == 3
        assert len(model.movie_ids) == 4
    
    def test_predict(self, sample_ratings):
        """Test rating prediction."""
        from src.collaborative_filtering import UserBasedCF
        
        model = UserBasedCF(k_neighbors=2)
        model.fit(sample_ratings)
        
        # Predict for user 1, movie 4 (not rated)
        pred = model.predict(1, 4)
        
        assert 1 <= pred <= 5
    
    def test_recommend(self, sample_ratings):
        """Test recommendation generation."""
        from src.collaborative_filtering import UserBasedCF
        
        model = UserBasedCF(k_neighbors=2)
        model.fit(sample_ratings)
        
        recs = model.recommend(1, n_recommendations=2)
        
        assert len(recs) <= 2
        assert all(isinstance(r, tuple) for r in recs)
        assert all(len(r) == 2 for r in recs)  # (movie_id, score)
    
    def test_similar_users(self, sample_ratings):
        """Test finding similar users."""
        from src.collaborative_filtering import UserBasedCF
        
        model = UserBasedCF(k_neighbors=2)
        model.fit(sample_ratings)
        
        similar = model.get_similar_users(1, k=2)
        
        assert len(similar) <= 2
        assert all(s[0] != 1 for s in similar)  # Shouldn't include self


class TestItemBasedCF:
    """Tests for Item-Based Collaborative Filtering."""
    
    @pytest.fixture
    def sample_ratings(self):
        """Create sample ratings data."""
        return pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 4],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0]
        })
    
    def test_fit(self, sample_ratings):
        """Test model fitting."""
        from src.collaborative_filtering import ItemBasedCF
        
        model = ItemBasedCF(k_neighbors=2)
        model.fit(sample_ratings)
        
        assert model.is_fitted
        assert model.item_similarity is not None
    
    def test_similar_items(self, sample_ratings):
        """Test finding similar items."""
        from src.collaborative_filtering import ItemBasedCF
        
        model = ItemBasedCF(k_neighbors=2)
        model.fit(sample_ratings)
        
        similar = model.get_similar_items(1, k=2)
        
        assert len(similar) <= 2
        assert all(s[0] != 1 for s in similar)  # Shouldn't include self


class TestContentBasedFiltering:
    """Tests for Content-Based Filtering."""
    
    @pytest.fixture
    def sample_movies(self):
        """Create sample movies data."""
        return pd.DataFrame({
            'movie_id': [1, 2, 3, 4, 5],
            'title': ['Action Movie', 'Comedy Film', 'Drama', 'Action Comedy', 'Pure Drama'],
            'Action': [1, 0, 0, 1, 0],
            'Comedy': [0, 1, 0, 1, 0],
            'Drama': [0, 0, 1, 0, 1],
            'avg_rating': [4.0, 3.5, 4.5, 4.0, 3.0],
            'rating_count': [100, 50, 75, 60, 40]
        })
    
    @pytest.fixture
    def sample_ratings(self):
        """Create sample ratings data."""
        return pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2],
            'movie_id': [1, 4, 3, 2, 5],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0]
        })
    
    def test_fit(self, sample_movies, sample_ratings):
        """Test model fitting."""
        from src.content_based import ContentBasedFiltering
        
        model = ContentBasedFiltering()
        model.fit(sample_movies, sample_ratings)
        
        assert model.is_fitted
        assert model.movie_features is not None
        assert model.movie_similarity is not None
    
    def test_similar_movies(self, sample_movies, sample_ratings):
        """Test finding similar movies."""
        from src.content_based import ContentBasedFiltering
        
        model = ContentBasedFiltering()
        model.fit(sample_movies, sample_ratings)
        
        # Movie 1 (Action) should be similar to Movie 4 (Action + Comedy)
        similar = model.get_similar_movies(1, n=2)
        
        assert len(similar) == 2
        similar_ids = [s[0] for s in similar]
        assert 4 in similar_ids  # Action Comedy similar to Action
    
    def test_user_profile(self, sample_movies, sample_ratings):
        """Test user profile building."""
        from src.content_based import ContentBasedFiltering
        
        model = ContentBasedFiltering()
        model.fit(sample_movies, sample_ratings)
        
        profile = model.build_user_profile(1, min_rating=4.0)
        
        assert profile is not None
        assert len(profile) == model.movie_features.shape[1]


class TestHybridModel:
    """Tests for Hybrid Recommendation Model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for hybrid model."""
        ratings = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 4],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0]
        })
        
        movies = pd.DataFrame({
            'movie_id': [1, 2, 3, 4],
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
            'Action': [1, 0, 1, 0],
            'Comedy': [0, 1, 0, 1],
            'avg_rating': [4.5, 4.0, 3.5, 4.5],
            'rating_count': [2, 3, 2, 2]
        })
        
        return ratings, movies
    
    def test_fit(self, sample_data):
        """Test hybrid model fitting."""
        from src.hybrid_model import HybridRecommender
        
        ratings, movies = sample_data
        
        model = HybridRecommender(strategy='weighted')
        model.fit(ratings, movies)
        
        assert model.is_fitted
        assert model.cf_model is not None
        assert model.cb_model is not None
    
    def test_recommend_weighted(self, sample_data):
        """Test weighted hybrid recommendations."""
        from src.hybrid_model import HybridRecommender
        
        ratings, movies = sample_data
        
        model = HybridRecommender(strategy='weighted')
        model.fit(ratings, movies)
        
        recs = model.recommend(1, n_recommendations=2)
        
        assert len(recs) <= 2
    
    def test_recommend_switching(self, sample_data):
        """Test switching hybrid recommendations."""
        from src.hybrid_model import HybridRecommender
        
        ratings, movies = sample_data
        
        model = HybridRecommender(strategy='switching')
        model.fit(ratings, movies)
        
        recs = model.recommend(1, n_recommendations=2)
        
        assert len(recs) <= 2


class TestModelEvaluator:
    """Tests for model evaluation metrics."""
    
    def test_rmse(self):
        """Test RMSE calculation."""
        from src.hybrid_model import ModelEvaluator
        
        actual = np.array([4.0, 3.0, 5.0, 2.0])
        predicted = np.array([3.5, 3.5, 4.5, 2.5])
        
        rmse = ModelEvaluator.rmse(actual, predicted)
        
        assert rmse > 0
        assert rmse < 1  # Small error
    
    def test_mae(self):
        """Test MAE calculation."""
        from src.hybrid_model import ModelEvaluator
        
        actual = np.array([4.0, 3.0, 5.0, 2.0])
        predicted = np.array([3.5, 3.5, 4.5, 2.5])
        
        mae = ModelEvaluator.mae(actual, predicted)
        
        assert mae == 0.5  # Average of [0.5, 0.5, 0.5, 0.5]
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        from src.hybrid_model import ModelEvaluator
        
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 6, 7]
        
        precision = ModelEvaluator.precision_at_k(recommended, relevant, k=5)
        
        assert precision == 0.4  # 2 hits in top 5
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        from src.hybrid_model import ModelEvaluator
        
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 6, 7]
        
        recall = ModelEvaluator.recall_at_k(recommended, relevant, k=5)
        
        assert recall == 0.5  # 2 out of 4 relevant items found
    
    def test_coverage(self):
        """Test catalog coverage calculation."""
        from src.hybrid_model import ModelEvaluator
        
        all_recommended = [[1, 2, 3], [2, 3, 4], [1, 4, 5]]
        total_items = 10
        
        coverage = ModelEvaluator.coverage(all_recommended, total_items)
        
        assert coverage == 0.5  # 5 unique items out of 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
