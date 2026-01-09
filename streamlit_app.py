"""
Movie Recommender - Streamlit Web Application

A modern, interactive movie recommendation system with multiple algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_and_preprocess_data
from src.collaborative_filtering import UserBasedCF, ItemBasedCF, SVDRecommender, SURPRISE_AVAILABLE
from src.content_based import ContentBasedFiltering
from src.hybrid_model import HybridRecommender
from app.utils import (
    create_rating_distribution_chart,
    create_genre_distribution_chart,
    create_top_movies_chart,
    create_user_stats_chart,
    create_genre_preference_chart,
    create_recommendation_scores_chart,
    format_movie_card
)

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2130 0%, #252a3d 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Cards */
    .movie-card {
        background: linear-gradient(135deg, #1e2130 0%, #2d3250 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #1e2130;
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e2130;
        border-radius: 8px;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e2130;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Genre columns
GENRE_COLUMNS = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]


@st.cache_data(show_spinner=False)
def load_data():
    """Load and preprocess the MovieLens dataset."""
    with st.spinner('🎬 Loading movie database...'):
        preprocessor, ratings, movies, users = load_and_preprocess_data()
    return preprocessor, ratings, movies, users


@st.cache_resource(show_spinner=False)
def load_models(_ratings, _movies):
    """Load and train recommendation models."""
    models = {}
    
    with st.spinner('🧠 Training recommendation models...'):
        # User-Based CF
        user_cf = UserBasedCF(k_neighbors=50)
        user_cf.fit(_ratings)
        models['User-Based CF'] = user_cf
        
        # Item-Based CF
        item_cf = ItemBasedCF(k_neighbors=50)
        item_cf.fit(_ratings)
        models['Item-Based CF'] = item_cf
        
        # Content-Based
        content = ContentBasedFiltering()
        content.fit(_movies, _ratings)
        models['Content-Based'] = content
        
        # Hybrid
        hybrid = HybridRecommender(cf_weight=0.7, cb_weight=0.3, strategy='weighted')
        hybrid.fit(_ratings, _movies)
        models['Hybrid (Recommended)'] = hybrid
        
        # SVD if available
        if SURPRISE_AVAILABLE:
            svd = SVDRecommender(n_factors=50, n_epochs=10)
            svd.fit(_ratings)
            models['SVD'] = svd
    
    return models


def get_user_stats(user_id, ratings_df, movies_df):
    """Get statistics for a specific user."""
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    if len(user_ratings) == 0:
        return None
    
    stats = {
        'total_ratings': len(user_ratings),
        'avg_rating': user_ratings['rating'].mean(),
        'ratings_df': user_ratings
    }
    
    # Get favorite genres
    genre_counts = {}
    for _, row in user_ratings[user_ratings['rating'] >= 4].iterrows():
        movie = movies_df[movies_df['movie_id'] == row['movie_id']]
        if len(movie) > 0:
            for genre in GENRE_COLUMNS:
                if genre in movie.columns and movie[genre].values[0] == 1:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    stats['favorite_genres'] = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return stats


def render_recommendations(recommendations, movies_df, cb_model):
    """Render recommendation cards."""
    if not recommendations:
        st.warning("No recommendations available. Try different settings.")
        return
    
    for rank, (movie_id, score) in enumerate(recommendations, 1):
        movie_info = cb_model.get_movie_info(movie_id) if hasattr(cb_model, 'get_movie_info') else None
        
        if movie_info is None:
            movie_data = movies_df[movies_df['movie_id'] == movie_id]
            if len(movie_data) > 0:
                movie = movie_data.iloc[0]
                genres = [g for g in GENRE_COLUMNS if g in movie and movie[g] == 1]
                movie_info = {
                    'title': movie.get('title', 'Unknown'),
                    'genres': genres,
                    'year': movie.get('year', 'N/A'),
                    'avg_rating': movie.get('avg_rating', 0),
                    'rating_count': movie.get('rating_count', 0)
                }
            else:
                continue
        
        card_html = format_movie_card(movie_info, score, rank)
        st.markdown(card_html, unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3rem; margin-bottom: 10px;">🎬 Movie Recommender</h1>
        <p style="color: #888; font-size: 1.1rem;">Discover your next favorite movie with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        preprocessor, ratings, movies, users = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Make sure the MovieLens dataset is downloaded. It will be downloaded automatically on first run.")
        return
    
    # Load models
    try:
        models = load_models(ratings, movies)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # User selection
        user_ids = sorted(ratings['user_id'].unique())
        
        user_input_method = st.radio(
            "Select user by:",
            ["Dropdown", "Enter ID"],
            horizontal=True
        )
        
        if user_input_method == "Dropdown":
            selected_user = st.selectbox(
                "Select User",
                user_ids,
                index=0
            )
        else:
            selected_user = st.number_input(
                "Enter User ID",
                min_value=min(user_ids),
                max_value=max(user_ids),
                value=min(user_ids)
            )
        
        st.markdown("---")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "🧠 Recommendation Algorithm",
            list(models.keys()),
            index=list(models.keys()).index('Hybrid (Recommended)') if 'Hybrid (Recommended)' in models else 0
        )
        
        # Number of recommendations
        n_recommendations = st.slider(
            "📊 Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10
        )
        
        st.markdown("---")
        
        # Genre filter
        st.markdown("### 🎭 Genre Filter")
        filter_genres = st.multiselect(
            "Show only these genres:",
            GENRE_COLUMNS
        )
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Algorithms:**
            - **User-Based CF**: Finds similar users
            - **Item-Based CF**: Finds similar movies
            - **Content-Based**: Uses movie features
            - **Hybrid**: Combines all approaches
            - **SVD**: Matrix factorization
            
            **Data**: MovieLens 100K Dataset
            """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Recommendations", 
        "👤 User Profile", 
        "🔍 Find Similar Movies",
        "📊 Dataset Insights"
    ])
    
    # Tab 1: Recommendations
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 🎬 Top Picks for User {selected_user}")
            
            # Get recommendations
            model = models[algorithm]
            
            try:
                if hasattr(model, 'recommend'):
                    recs = model.recommend(selected_user, n_recommendations=n_recommendations * 2)
                else:
                    st.warning("Selected model doesn't support recommendations.")
                    recs = []
                
                # Apply genre filter if specified
                if filter_genres and recs:
                    filtered_recs = []
                    for movie_id, score in recs:
                        movie_data = movies[movies['movie_id'] == movie_id]
                        if len(movie_data) > 0:
                            movie = movie_data.iloc[0]
                            movie_genres = [g for g in GENRE_COLUMNS if g in movie and movie[g] == 1]
                            if any(g in movie_genres for g in filter_genres):
                                filtered_recs.append((movie_id, score))
                    recs = filtered_recs[:n_recommendations]
                else:
                    recs = recs[:n_recommendations]
                
                # Get content-based model for movie info
                cb_model = models.get('Content-Based') or models.get('Hybrid (Recommended)')
                
                render_recommendations(recs, movies, cb_model)
                
                # Export button
                if recs:
                    rec_data = []
                    for movie_id, score in recs:
                        movie_data = movies[movies['movie_id'] == movie_id]
                        if len(movie_data) > 0:
                            rec_data.append({
                                'Movie ID': movie_id,
                                'Title': movie_data['title'].values[0],
                                'Score': score
                            })
                    
                    if rec_data:
                        export_df = pd.DataFrame(rec_data)
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Recommendations",
                            csv,
                            "recommendations.csv",
                            "text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
        
        with col2:
            st.markdown("### 📈 Score Distribution")
            if recs:
                fig = create_recommendation_scores_chart(recs[:10], movies)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: User Profile
    with tab2:
        user_stats = get_user_stats(selected_user, ratings, movies)
        
        if user_stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎬 Movies Rated", user_stats['total_ratings'])
            with col2:
                st.metric("⭐ Average Rating", f"{user_stats['avg_rating']:.2f}")
            with col3:
                favorite = user_stats['favorite_genres'][0][0] if user_stats['favorite_genres'] else "N/A"
                st.metric("❤️ Favorite Genre", favorite)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Your Rating Pattern")
                fig = create_user_stats_chart(user_stats['ratings_df'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎭 Genre Preferences")
                cb_model = models.get('Content-Based')
                if cb_model:
                    genre_prefs = cb_model.get_user_genre_preferences(selected_user)
                    fig = create_genre_preference_chart(genre_prefs)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recent ratings
            st.markdown("### 🕐 Recent Ratings")
            recent = user_stats['ratings_df'].sort_values('timestamp', ascending=False).head(10)
            
            display_data = []
            for _, row in recent.iterrows():
                movie_data = movies[movies['movie_id'] == row['movie_id']]
                if len(movie_data) > 0:
                    display_data.append({
                        'Title': movie_data['title'].values[0],
                        'Your Rating': '⭐' * int(row['rating']),
                        'Date': row['timestamp'].strftime('%Y-%m-%d') if pd.notna(row['timestamp']) else 'N/A'
                    })
            
            if display_data:
                st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
        else:
            st.info("No rating history found for this user.")
    
    # Tab 3: Find Similar Movies
    with tab3:
        st.markdown("### 🔍 Find Movies Similar to Your Favorites")
        
        # Movie search
        movie_titles = movies['title'].tolist()
        selected_movie_title = st.selectbox(
            "Select or search for a movie:",
            movie_titles,
            index=0
        )
        
        selected_movie = movies[movies['title'] == selected_movie_title]
        
        if len(selected_movie) > 0:
            movie_id = selected_movie['movie_id'].values[0]
            
            cb_model = models.get('Content-Based')
            if cb_model:
                similar = cb_model.get_similar_movies(movie_id, n=10)
                
                st.markdown(f"### Movies similar to *{selected_movie_title}*")
                
                render_recommendations(similar, movies, cb_model)
    
    # Tab 4: Dataset Insights
    with tab4:
        st.markdown("### 📊 MovieLens Dataset Overview")
        
        # Statistics
        stats = preprocessor.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Total Users", f"{stats['n_users']:,}")
        with col2:
            st.metric("🎬 Total Movies", f"{stats['n_movies']:,}")
        with col3:
            st.metric("⭐ Total Ratings", f"{stats['n_ratings']:,}")
        with col4:
            st.metric("📉 Sparsity", f"{stats['sparsity']*100:.1f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⭐ Rating Distribution")
            fig = create_rating_distribution_chart(ratings)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎭 Genre Distribution")
            fig = create_genre_distribution_chart(movies, GENRE_COLUMNS)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🏆 Top Rated Movies")
        fig = create_top_movies_chart(movies, n=15)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
