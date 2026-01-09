"""
Hybrid Recommendation System Module.

This module implements:
- Weighted Hybrid combining CF and Content-Based
- Switching Hybrid for cold start handling
- Cascade Hybrid for refined recommendations
- Model evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from src.collaborative_filtering import UserBasedCF, ItemBasedCF, SVDRecommender, SURPRISE_AVAILABLE
from src.content_based import ContentBasedFiltering


class HybridRecommender:
    """
    Hybrid Recommendation System combining multiple algorithms.
    
    Strategies:
    - Weighted: Combines CF and CB scores with weights
    - Switching: Uses different models based on user/item activity
    - Cascade: Uses CF first, then refines with CB
    """
    
    def __init__(self, 
                 cf_weight: float = 0.7, 
                 cb_weight: float = 0.3,
                 cf_algorithm: str = 'item_based',
                 cold_start_threshold: int = 5,
                 strategy: str = 'weighted'):
        """
        Initialize Hybrid Recommender.
        
        Args:
            cf_weight: Weight for collaborative filtering (0-1)
            cb_weight: Weight for content-based filtering (0-1)
            cf_algorithm: 'user_based', 'item_based', or 'svd'
            cold_start_threshold: Minimum ratings for non-cold-start
            strategy: 'weighted', 'switching', or 'cascade'
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_algorithm = cf_algorithm
        self.cold_start_threshold = cold_start_threshold
        self.strategy = strategy
        
        # Initialize models
        self.cf_model = None
        self.cb_model = None
        
        # Data
        self.ratings_df = None
        self.movies_df = None
        self.user_rating_counts = None
        self.movie_rating_counts = None
        
        self.is_fitted = False
        
    def fit(self, ratings_df: pd.DataFrame, 
            movies_df: pd.DataFrame) -> 'HybridRecommender':
        """
        Fit all component models.
        
        Args:
            ratings_df: Ratings DataFrame
            movies_df: Movies DataFrame
            
        Returns:
            self
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        # Calculate rating counts for cold start detection
        self.user_rating_counts = ratings_df.groupby('user_id').size().to_dict()
        self.movie_rating_counts = ratings_df.groupby('movie_id').size().to_dict()
        
        # Fit collaborative filtering model
        print(f"Fitting {self.cf_algorithm} collaborative filtering model...")
        if self.cf_algorithm == 'user_based':
            self.cf_model = UserBasedCF(k_neighbors=50)
        elif self.cf_algorithm == 'item_based':
            self.cf_model = ItemBasedCF(k_neighbors=50)
        elif self.cf_algorithm == 'svd' and SURPRISE_AVAILABLE:
            self.cf_model = SVDRecommender(n_factors=100, n_epochs=20)
        else:
            # Default to item-based
            self.cf_model = ItemBasedCF(k_neighbors=50)
        
        self.cf_model.fit(ratings_df)
        
        # Fit content-based model
        print("Fitting content-based model...")
        self.cb_model = ContentBasedFiltering()
        self.cb_model.fit(movies_df, ratings_df)
        
        self.is_fitted = True
        print("Hybrid model fitted successfully!")
        
        return self
    
    def _is_cold_start_user(self, user_id: int) -> bool:
        """Check if user is cold-start (few ratings)."""
        count = self.user_rating_counts.get(user_id, 0)
        return count < self.cold_start_threshold
    
    def _is_cold_start_item(self, movie_id: int) -> bool:
        """Check if movie is cold-start (few ratings)."""
        count = self.movie_rating_counts.get(movie_id, 0)
        return count < self.cold_start_threshold
    
    def _normalize_scores(self, scores: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores
        
        ratings = [s[1] for s in scores]
        min_r, max_r = min(ratings), max(ratings)
        
        if max_r - min_r == 0:
            return [(mid, 0.5) for mid, _ in scores]
        
        normalized = [
            (mid, (r - min_r) / (max_r - min_r))
            for mid, r in scores
        ]
        
        return normalized
    
    def recommend_weighted(self, user_id: int, 
                           n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Weighted hybrid recommendation.
        
        Combines CF and CB scores with specified weights.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        # Get more candidates than needed for better fusion
        n_candidates = n_recommendations * 5
        
        # Get CF recommendations
        cf_recs = self.cf_model.recommend(user_id, n_recommendations=n_candidates)
        cf_recs = self._normalize_scores(cf_recs)
        cf_dict = {mid: score for mid, score in cf_recs}
        
        # Get CB recommendations
        cb_recs = self.cb_model.recommend(user_id, n_recommendations=n_candidates)
        cb_recs = self._normalize_scores(cb_recs)
        cb_dict = {mid: score for mid, score in cb_recs}
        
        # Combine scores
        all_movies = set(cf_dict.keys()) | set(cb_dict.keys())
        
        combined_scores = []
        for movie_id in all_movies:
            cf_score = cf_dict.get(movie_id, 0)
            cb_score = cb_dict.get(movie_id, 0)
            
            combined = self.cf_weight * cf_score + self.cb_weight * cb_score
            combined_scores.append((movie_id, combined))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        return combined_scores[:n_recommendations]
    
    def recommend_switching(self, user_id: int,
                            n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Switching hybrid recommendation.
        
        Uses CB for cold-start users, CF otherwise.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        if self._is_cold_start_user(user_id):
            # Use content-based for cold-start users
            return self.cb_model.recommend(user_id, n_recommendations=n_recommendations)
        else:
            # Use CF with CB fallback
            cf_recs = self.cf_model.recommend(user_id, n_recommendations=n_recommendations)
            
            # If not enough recommendations, supplement with CB
            if len(cf_recs) < n_recommendations:
                cb_recs = self.cb_model.recommend(
                    user_id, 
                    n_recommendations=n_recommendations - len(cf_recs)
                )
                cf_movie_ids = {mid for mid, _ in cf_recs}
                for mid, score in cb_recs:
                    if mid not in cf_movie_ids:
                        cf_recs.append((mid, score))
            
            return cf_recs[:n_recommendations]
    
    def recommend_cascade(self, user_id: int,
                          n_recommendations: int = 10,
                          n_first_stage: int = 50) -> List[Tuple[int, float]]:
        """
        Cascade hybrid recommendation.
        
        Gets CF recommendations first, then re-ranks with CB.
        
        Args:
            user_id: User ID
            n_recommendations: Final number of recommendations
            n_first_stage: Candidates from first stage
            
        Returns:
            List of (movie_id, score) tuples
        """
        # First stage: CF recommendations
        cf_recs = self.cf_model.recommend(user_id, n_recommendations=n_first_stage)
        
        if not cf_recs:
            return self.cb_model.recommend(user_id, n_recommendations=n_recommendations)
        
        cf_recs = self._normalize_scores(cf_recs)
        
        # Second stage: Re-rank with CB
        user_profile = self.cb_model.build_user_profile(user_id)
        
        reranked = []
        for movie_id, cf_score in cf_recs:
            if movie_id in self.cb_model.movie_id_to_idx:
                movie_idx = self.cb_model.movie_id_to_idx[movie_id]
                movie_features = self.cb_model.movie_features[movie_idx]
                
                # Compute CB score
                cb_score = np.dot(user_profile, movie_features) / (
                    np.linalg.norm(user_profile) * np.linalg.norm(movie_features) + 1e-8
                )
                
                # Combine for final score
                final_score = self.cf_weight * cf_score + self.cb_weight * cb_score
                reranked.append((movie_id, final_score))
            else:
                reranked.append((movie_id, cf_score))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:n_recommendations]
    
    def recommend(self, user_id: int, 
                  n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations using the configured strategy.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.strategy == 'weighted':
            return self.recommend_weighted(user_id, n_recommendations)
        elif self.strategy == 'switching':
            return self.recommend_switching(user_id, n_recommendations)
        elif self.strategy == 'cascade':
            return self.recommend_cascade(user_id, n_recommendations)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
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
        
        # Get CF prediction
        cf_pred = self.cf_model.predict(user_id, movie_id)
        
        # For hybrid prediction, we use CF primarily
        # CB is more for ranking than rating prediction
        return cf_pred
    
    def get_similar_movies(self, movie_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get movies similar to a given movie.
        
        Uses content-based similarity.
        
        Args:
            movie_id: Reference movie ID
            n: Number of similar movies
            
        Returns:
            List of (movie_id, similarity) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.cb_model.get_similar_movies(movie_id, n)
    
    def explain_recommendation(self, user_id: int, 
                                movie_id: int) -> Dict:
        """
        Explain why a movie was recommended.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'movie_id': movie_id,
            'movie_info': self.cb_model.get_movie_info(movie_id),
            'reasons': []
        }
        
        # Get user's genre preferences
        genre_prefs = self.cb_model.get_user_genre_preferences(user_id)
        
        # Get movie genres
        movie_info = explanation['movie_info']
        if movie_info:
            movie_genres = movie_info.get('genres', [])
            
            # Check genre match
            matching_genres = [g for g in movie_genres if g in genre_prefs]
            if matching_genres:
                explanation['reasons'].append(
                    f"Matches your preferred genres: {', '.join(matching_genres)}"
                )
        
        # Check similar users (CF explanation)
        if hasattr(self.cf_model, 'get_similar_users'):
            similar_users = self.cf_model.get_similar_users(user_id, k=5)
            if similar_users:
                explanation['reasons'].append(
                    f"Users with similar taste also enjoyed this movie"
                )
        
        # Get similar movies user liked
        user_ratings = self.ratings_df[
            (self.ratings_df['user_id'] == user_id) & 
            (self.ratings_df['rating'] >= 4)
        ]['movie_id'].tolist()
        
        for rated_movie in user_ratings[:5]:
            similar = self.cb_model.get_similar_movies(rated_movie, n=20)
            similar_ids = [m[0] for m in similar]
            if movie_id in similar_ids:
                rated_title = self.movies_df[
                    self.movies_df['movie_id'] == rated_movie
                ]['title'].values
                if len(rated_title) > 0:
                    explanation['reasons'].append(
                        f"Similar to '{rated_title[0]}' which you rated highly"
                    )
                break
        
        return explanation


class ModelEvaluator:
    """
    Evaluate recommendation model performance.
    """
    
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    @staticmethod
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(actual - predicted))
    
    @staticmethod
    def precision_at_k(recommended: List[int], 
                       relevant: List[int], 
                       k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant item IDs (actually liked)
            k: Number of recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k <= 0:
            return 0.0
        
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        hits = len(recommended_k & relevant_set)
        
        return hits / k
    
    @staticmethod
    def recall_at_k(recommended: List[int], 
                    relevant: List[int], 
                    k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant item IDs
            k: Number of recommendations
            
        Returns:
            Recall@K score
        """
        if len(relevant) == 0:
            return 0.0
        
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        hits = len(recommended_k & relevant_set)
        
        return hits / len(relevant_set)
    
    @staticmethod
    def ndcg_at_k(recommended: List[int], 
                  relevant: List[int], 
                  k: int) -> float:
        """
        Calculate NDCG@K.
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant item IDs
            k: Number of recommendations
            
        Returns:
            NDCG@K score
        """
        relevant_set = set(relevant)
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed
        
        # Calculate ideal DCG
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        
        if ideal_dcg == 0:
            return 0.0
        
        return dcg / ideal_dcg
    
    @staticmethod
    def coverage(all_recommended: List[List[int]], 
                 total_items: int) -> float:
        """
        Calculate catalog coverage.
        
        Args:
            all_recommended: List of recommendation lists for all users
            total_items: Total number of items in catalog
            
        Returns:
            Coverage score (0-1)
        """
        unique_items = set()
        for recs in all_recommended:
            unique_items.update(recs)
        
        return len(unique_items) / total_items if total_items > 0 else 0.0
    
    def evaluate_model(self, model, test_df: pd.DataFrame, 
                       k: int = 10, min_rating: float = 4.0) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Recommendation model with predict() and recommend() methods
            test_df: Test DataFrame
            k: K for precision/recall/NDCG
            min_rating: Threshold for relevant items
            
        Returns:
            Dictionary with evaluation metrics
        """
        users = test_df['user_id'].unique()
        
        # Rating prediction metrics
        predictions = []
        actuals = []
        
        # Recommendation metrics
        all_precisions = []
        all_recalls = []
        all_ndcgs = []
        all_recommendations = []
        
        for user_id in users:
            user_test = test_df[test_df['user_id'] == user_id]
            
            # Rating predictions
            for _, row in user_test.iterrows():
                try:
                    pred = model.predict(row['user_id'], row['movie_id'])
                    predictions.append(pred)
                    actuals.append(row['rating'])
                except:
                    pass
            
            # Recommendation metrics
            try:
                recs = model.recommend(user_id, n_recommendations=k)
                recommended_ids = [r[0] for r in recs]
                
                # Get relevant items (high ratings in test set)
                relevant = user_test[user_test['rating'] >= min_rating]['movie_id'].tolist()
                
                all_precisions.append(self.precision_at_k(recommended_ids, relevant, k))
                all_recalls.append(self.recall_at_k(recommended_ids, relevant, k))
                all_ndcgs.append(self.ndcg_at_k(recommended_ids, relevant, k))
                all_recommendations.append(recommended_ids)
            except:
                pass
        
        # Calculate final metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        total_movies = test_df['movie_id'].nunique()
        
        return {
            'rmse': self.rmse(actuals, predictions) if len(predictions) > 0 else None,
            'mae': self.mae(actuals, predictions) if len(predictions) > 0 else None,
            f'precision@{k}': np.mean(all_precisions) if all_precisions else 0.0,
            f'recall@{k}': np.mean(all_recalls) if all_recalls else 0.0,
            f'ndcg@{k}': np.mean(all_ndcgs) if all_ndcgs else 0.0,
            'coverage': self.coverage(all_recommendations, total_movies)
        }


if __name__ == "__main__":
    # Test hybrid model
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.preprocessing import load_and_preprocess_data
    
    print("Loading data...")
    preprocessor, ratings, movies, users = load_and_preprocess_data()
    
    print("\nTesting Hybrid Recommender (weighted strategy)...")
    hybrid = HybridRecommender(
        cf_weight=0.7, 
        cb_weight=0.3, 
        cf_algorithm='item_based',
        strategy='weighted'
    )
    hybrid.fit(ratings, movies)
    
    test_user = ratings['user_id'].iloc[0]
    print(f"\nHybrid recommendations for user {test_user}:")
    recs = hybrid.recommend(test_user, n_recommendations=5)
    
    for movie_id, score in recs:
        title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"  {title}: {score:.3f}")
    
    # Test explanation
    if recs:
        print(f"\nExplanation for first recommendation:")
        explanation = hybrid.explain_recommendation(test_user, recs[0][0])
        print(f"  Movie: {explanation['movie_info']['title']}")
        print(f"  Reasons: {explanation['reasons']}")
    
    print("\n" + "="*50)
    print("Testing Model Evaluation...")
    
    # Split data for evaluation
    train_df, test_df = preprocessor.train_test_split(test_size=0.2)
    
    # Fit on training data
    hybrid_eval = HybridRecommender(strategy='weighted')
    hybrid_eval.fit(train_df, movies)
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(hybrid_eval, test_df, k=10)
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
