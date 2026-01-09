"""
Configuration settings for the Movie Recommendation System.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# MovieLens dataset URLs
MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

# Model parameters
DEFAULT_N_RECOMMENDATIONS = 10
DEFAULT_N_SIMILAR_USERS = 50
DEFAULT_N_SIMILAR_ITEMS = 50
DEFAULT_MIN_RATING = 3.5

# SVD parameters
SVD_N_FACTORS = 100
SVD_N_EPOCHS = 20
SVD_LEARNING_RATE = 0.005
SVD_REGULARIZATION = 0.02

# Hybrid model weights
HYBRID_CF_WEIGHT = 0.7
HYBRID_CB_WEIGHT = 0.3

# Cold start thresholds
COLD_START_USER_THRESHOLD = 5
COLD_START_ITEM_THRESHOLD = 5

# UI settings
PAGE_TITLE = "🎬 Movie Recommender"
PAGE_ICON = "🎬"
LAYOUT = "wide"

# Color scheme for UI
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'background': '#0e1117',
    'card': '#1e2130',
    'text': '#ffffff',
    'success': '#00d4aa',
    'warning': '#ffc107'
}

# Genre list
GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]
