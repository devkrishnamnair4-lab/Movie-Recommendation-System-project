"""
Data Preprocessing Module for Movie Recommendation System.

This module handles:
- Loading MovieLens dataset
- Data cleaning and preparation
- Feature engineering
- Creating user-item matrices
- Train-test splitting
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')


class MovieLensDataLoader:
    """
    Loads and preprocesses the MovieLens dataset.
    Supports both MovieLens 100K and 1M datasets.
    """
    
    def __init__(self, data_path: str = None, dataset: str = '100k'):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to store/load data
            dataset: '100k' or '1m'
        """
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'data', 'raw')
        
        self.data_path = data_path
        self.dataset = dataset
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        
        # Create directories if they don't exist
        os.makedirs(data_path, exist_ok=True)
        
    def download_dataset(self) -> None:
        """Download the MovieLens dataset if not already present."""
        if self.dataset == '100k':
            url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
            folder_name = 'ml-100k'
        else:
            url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
            folder_name = 'ml-1m'
        
        zip_path = os.path.join(self.data_path, f'{folder_name}.zip')
        extract_path = os.path.join(self.data_path, folder_name)
        
        if os.path.exists(extract_path):
            print(f"Dataset already exists at {extract_path}")
            return
        
        print(f"Downloading MovieLens {self.dataset} dataset...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_path)
        
        # Clean up zip file
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully!")
    
    def load_100k_data(self) -> tuple:
        """Load MovieLens 100K dataset."""
        folder = os.path.join(self.data_path, 'ml-100k')
        
        # Load ratings
        ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.ratings_df = pd.read_csv(
            os.path.join(folder, 'u.data'),
            sep='\t',
            names=ratings_cols,
            encoding='latin-1'
        )
        
        # Load movies
        movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date',
                       'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.movies_df = pd.read_csv(
            os.path.join(folder, 'u.item'),
            sep='|',
            names=movies_cols,
            encoding='latin-1'
        )
        
        # Load users
        users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        self.users_df = pd.read_csv(
            os.path.join(folder, 'u.user'),
            sep='|',
            names=users_cols,
            encoding='latin-1'
        )
        
        return self.ratings_df, self.movies_df, self.users_df
    
    def load_1m_data(self) -> tuple:
        """Load MovieLens 1M dataset."""
        folder = os.path.join(self.data_path, 'ml-1m')
        
        # Load ratings
        self.ratings_df = pd.read_csv(
            os.path.join(folder, 'ratings.dat'),
            sep='::',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python',
            encoding='latin-1'
        )
        
        # Load movies
        self.movies_df = pd.read_csv(
            os.path.join(folder, 'movies.dat'),
            sep='::',
            names=['movie_id', 'title', 'genres'],
            engine='python',
            encoding='latin-1'
        )
        
        # Load users
        self.users_df = pd.read_csv(
            os.path.join(folder, 'users.dat'),
            sep='::',
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            engine='python',
            encoding='latin-1'
        )
        
        return self.ratings_df, self.movies_df, self.users_df
    
    def load_data(self) -> tuple:
        """Load the appropriate dataset."""
        self.download_dataset()
        
        if self.dataset == '100k':
            return self.load_100k_data()
        else:
            return self.load_1m_data()


class DataPreprocessor:
    """
    Preprocesses the MovieLens data for recommendation algorithms.
    """
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                 users_df: pd.DataFrame = None, dataset: str = '100k'):
        """
        Initialize the preprocessor.
        
        Args:
            ratings_df: Ratings dataframe
            movies_df: Movies dataframe
            users_df: Users dataframe (optional)
            dataset: '100k' or '1m'
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        self.users_df = users_df.copy() if users_df is not None else None
        self.dataset = dataset
        
        self.genre_columns = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
            'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western'
        ]
    
    def clean_data(self) -> None:
        """Clean the data by handling missing values and duplicates."""
        # Remove duplicate ratings
        self.ratings_df = self.ratings_df.drop_duplicates(
            subset=['user_id', 'movie_id'], 
            keep='last'
        )
        
        # Ensure rating is in valid range
        self.ratings_df = self.ratings_df[
            (self.ratings_df['rating'] >= 1) & 
            (self.ratings_df['rating'] <= 5)
        ]
        
        # Convert timestamp to datetime
        self.ratings_df['timestamp'] = pd.to_datetime(
            self.ratings_df['timestamp'], 
            unit='s'
        )
        
        # Clean movie titles
        if 'title' in self.movies_df.columns:
            self.movies_df['title'] = self.movies_df['title'].fillna('Unknown Title')
            # Extract year from title if present (e.g., "Toy Story (1995)")
            self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)')
            self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce')
            self.movies_df['year'] = self.movies_df['year'].fillna(1990).astype(int)
        
        # Process genres for 1M dataset
        if self.dataset == '1m' and 'genres' in self.movies_df.columns:
            self._process_genres_1m()
    
    def _process_genres_1m(self) -> None:
        """Process genres for MovieLens 1M format."""
        # Split genres string into list
        self.movies_df['genre_list'] = self.movies_df['genres'].str.split('|')
        
        # Create binary columns for each genre
        mlb = MultiLabelBinarizer(classes=self.genre_columns)
        genre_matrix = mlb.fit_transform(self.movies_df['genre_list'].fillna(''))
        
        for i, genre in enumerate(self.genre_columns):
            self.movies_df[genre] = genre_matrix[:, i]
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create new features for the recommendation system.
        
        Returns:
            DataFrame with engineered features
        """
        # Rating statistics per movie
        movie_stats = self.ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        movie_stats.columns = ['movie_id', 'rating_count', 'avg_rating', 'rating_std']
        movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
        
        # User statistics
        user_stats = self.ratings_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        user_stats.columns = ['user_id', 'user_rating_count', 'user_avg_rating', 'user_rating_std']
        user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
        
        # Merge movie stats with movies dataframe
        self.movies_df = self.movies_df.merge(movie_stats, on='movie_id', how='left')
        self.movies_df['rating_count'] = self.movies_df['rating_count'].fillna(0).astype(int)
        self.movies_df['avg_rating'] = self.movies_df['avg_rating'].fillna(0)
        self.movies_df['rating_std'] = self.movies_df['rating_std'].fillna(0)
        
        # Merge user stats with ratings
        self.ratings_df = self.ratings_df.merge(user_stats, on='user_id', how='left')
        
        # Normalized rating (remove user bias)
        self.ratings_df['rating_normalized'] = (
            self.ratings_df['rating'] - self.ratings_df['user_avg_rating']
        )
        
        # Popularity score (log-scaled rating count)
        self.movies_df['popularity'] = np.log1p(self.movies_df['rating_count'])
        
        # Bayesian average rating
        C = self.movies_df['avg_rating'].mean()
        m = self.movies_df['rating_count'].quantile(0.1)
        self.movies_df['weighted_rating'] = (
            (self.movies_df['rating_count'] * self.movies_df['avg_rating'] + m * C) /
            (self.movies_df['rating_count'] + m)
        )
        
        return self.ratings_df
    
    def create_user_item_matrix(self, fill_value: float = 0) -> pd.DataFrame:
        """
        Create user-item rating matrix.
        
        Args:
            fill_value: Value to fill for missing ratings
            
        Returns:
            User-item matrix as DataFrame
        """
        user_item_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=fill_value
        )
        return user_item_matrix
    
    def create_sparse_matrix(self) -> tuple:
        """
        Create sparse user-item matrix for efficient computation.
        
        Returns:
            Tuple of (sparse_matrix, user_ids, movie_ids)
        """
        # Create mappings
        user_ids = self.ratings_df['user_id'].unique()
        movie_ids = self.ratings_df['movie_id'].unique()
        
        user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
        
        # Create sparse matrix
        rows = self.ratings_df['user_id'].map(user_to_idx)
        cols = self.ratings_df['movie_id'].map(movie_to_idx)
        values = self.ratings_df['rating']
        
        sparse_matrix = csr_matrix(
            (values, (rows, cols)), 
            shape=(len(user_ids), len(movie_ids))
        )
        
        return sparse_matrix, user_ids, movie_ids, user_to_idx, movie_to_idx
    
    def get_movie_features(self, normalize: bool = True) -> np.ndarray:
        """
        Get movie feature vectors for content-based filtering.
        
        Args:
            normalize: Whether to normalize features
            
        Returns:
            Feature matrix (n_movies, n_features)
        """
        # Genre features
        genre_features = self.movies_df[self.genre_columns].values
        
        # Numerical features
        numerical_features = self.movies_df[['avg_rating', 'popularity', 'weighted_rating']].copy()
        numerical_features = numerical_features.fillna(0).values
        
        if normalize:
            scaler = StandardScaler()
            numerical_features = scaler.fit_transform(numerical_features)
        
        # Combine features
        features = np.hstack([genre_features, numerical_features])
        
        return features
    
    def train_test_split(self, test_size: float = 0.2, 
                         random_state: int = 42) -> tuple:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df, test_df = train_test_split(
            self.ratings_df,
            test_size=test_size,
            random_state=random_state,
            stratify=None  # Can stratify by user if needed
        )
        
        return train_df, test_df
    
    def get_statistics(self) -> dict:
        """
        Calculate dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        n_users = self.ratings_df['user_id'].nunique()
        n_movies = self.ratings_df['movie_id'].nunique()
        n_ratings = len(self.ratings_df)
        
        # Sparsity
        sparsity = 1 - (n_ratings / (n_users * n_movies))
        
        # Rating distribution
        rating_dist = self.ratings_df['rating'].value_counts().sort_index()
        
        # Cold start analysis
        user_counts = self.ratings_df['user_id'].value_counts()
        movie_counts = self.ratings_df['movie_id'].value_counts()
        
        cold_start_users = (user_counts < 5).sum()
        cold_start_movies = (movie_counts < 5).sum()
        
        stats = {
            'n_users': n_users,
            'n_movies': n_movies,
            'n_ratings': n_ratings,
            'sparsity': sparsity,
            'avg_ratings_per_user': n_ratings / n_users,
            'avg_ratings_per_movie': n_ratings / n_movies,
            'rating_distribution': rating_dist.to_dict(),
            'cold_start_users': cold_start_users,
            'cold_start_movies': cold_start_movies,
            'rating_mean': self.ratings_df['rating'].mean(),
            'rating_std': self.ratings_df['rating'].std()
        }
        
        return stats


def load_and_preprocess_data(data_path: str = None, 
                              dataset: str = '100k') -> tuple:
    """
    Convenience function to load and preprocess data in one call.
    
    Args:
        data_path: Path to data directory
        dataset: '100k' or '1m'
        
    Returns:
        Tuple of (preprocessor, ratings_df, movies_df, users_df)
    """
    # Load data
    loader = MovieLensDataLoader(data_path, dataset)
    ratings_df, movies_df, users_df = loader.load_data()
    
    # Preprocess
    preprocessor = DataPreprocessor(ratings_df, movies_df, users_df, dataset)
    preprocessor.clean_data()
    preprocessor.engineer_features()
    
    return preprocessor, preprocessor.ratings_df, preprocessor.movies_df, users_df


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Loading and preprocessing MovieLens 100K dataset...")
    preprocessor, ratings, movies, users = load_and_preprocess_data()
    
    print("\nDataset Statistics:")
    stats = preprocessor.get_statistics()
    for key, value in stats.items():
        if key != 'rating_distribution':
            print(f"  {key}: {value}")
    
    print("\nRating Distribution:")
    for rating, count in stats['rating_distribution'].items():
        print(f"  {rating} stars: {count} ratings")
    
    print("\nSample of processed movies:")
    print(movies[['movie_id', 'title', 'avg_rating', 'rating_count', 'popularity']].head(10))
