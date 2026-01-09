"""
Helper utilities for the Streamlit application.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_rating_distribution_chart(ratings_df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive rating distribution histogram.
    
    Args:
        ratings_df: Ratings DataFrame
        
    Returns:
        Plotly figure
    """
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=rating_counts.index.astype(str),
            y=rating_counts.values,
            marker=dict(
                color=rating_counts.values,
                colorscale='Viridis',
                showscale=False
            ),
            hovertemplate='Rating: %{x}<br>Count: %{y:,}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Rating Distribution',
        xaxis_title='Rating',
        yaxis_title='Count',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=350
    )
    
    return fig


def create_genre_distribution_chart(movies_df: pd.DataFrame, 
                                     genre_columns: list) -> go.Figure:
    """
    Create a genre distribution pie chart.
    
    Args:
        movies_df: Movies DataFrame
        genre_columns: List of genre column names
        
    Returns:
        Plotly figure
    """
    # Count movies per genre
    genre_counts = {}
    for genre in genre_columns:
        if genre in movies_df.columns:
            genre_counts[genre] = movies_df[genre].sum()
    
    if not genre_counts:
        return go.Figure()
    
    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())
    
    fig = go.Figure(data=[
        go.Pie(
            labels=genres,
            values=counts,
            hole=0.4,
            textposition='inside',
            textinfo='percent+label',
            marker=dict(
                colors=px.colors.qualitative.Set3
            )
        )
    ])
    
    fig.update_layout(
        title='Genre Distribution',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        showlegend=False
    )
    
    return fig


def create_top_movies_chart(movies_df: pd.DataFrame, n: int = 20) -> go.Figure:
    """
    Create a horizontal bar chart of top-rated movies.
    
    Args:
        movies_df: Movies DataFrame with rating info
        n: Number of movies to show
        
    Returns:
        Plotly figure
    """
    if 'rating_count' not in movies_df.columns:
        return go.Figure()
    
    # Filter for movies with enough ratings
    popular = movies_df[movies_df['rating_count'] >= 50].copy()
    popular = popular.nlargest(n, 'avg_rating')
    
    # Truncate titles for display
    popular['title_short'] = popular['title'].str[:30]
    
    fig = go.Figure(data=[
        go.Bar(
            y=popular['title_short'],
            x=popular['avg_rating'],
            orientation='h',
            marker=dict(
                color=popular['avg_rating'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Rating')
            ),
            hovertemplate='%{y}<br>Rating: %{x:.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'Top {n} Highest Rated Movies',
        xaxis_title='Average Rating',
        yaxis_title='',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500,
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def create_user_stats_chart(user_ratings: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing user's rating history.
    
    Args:
        user_ratings: User's ratings DataFrame
        
    Returns:
        Plotly figure
    """
    if len(user_ratings) == 0:
        return go.Figure()
    
    rating_counts = user_ratings['rating'].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=rating_counts.index.astype(str),
            y=rating_counts.values,
            marker=dict(
                color=['#ff6b6b', '#ffa06b', '#ffd06b', '#b8e986', '#6bcf63'],
                line=dict(width=0)
            )
        )
    ])
    
    fig.update_layout(
        title='Your Rating Pattern',
        xaxis_title='Rating',
        yaxis_title='Count',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=250
    )
    
    return fig


def create_genre_preference_chart(genre_prefs: dict) -> go.Figure:
    """
    Create a radar chart of user's genre preferences.
    
    Args:
        genre_prefs: Dictionary of genre->preference score
        
    Returns:
        Plotly figure
    """
    if not genre_prefs:
        return go.Figure()
    
    genres = list(genre_prefs.keys())[:10]  # Top 10 genres
    values = [genre_prefs[g] for g in genres]
    
    # Close the radar chart
    genres = genres + [genres[0]]
    values = values + [values[0]]
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=values,
            theta=genres,
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=2),
            marker=dict(size=8, color='#667eea')
        )
    ])
    
    fig.update_layout(
        title='Your Genre Preferences',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=350,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        )
    )
    
    return fig


def create_recommendation_scores_chart(recommendations: list, 
                                        movies_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart of recommendation scores.
    
    Args:
        recommendations: List of (movie_id, score) tuples
        movies_df: Movies DataFrame
        
    Returns:
        Plotly figure
    """
    if not recommendations:
        return go.Figure()
    
    movie_ids = [r[0] for r in recommendations]
    scores = [r[1] for r in recommendations]
    
    titles = []
    for mid in movie_ids:
        movie = movies_df[movies_df['movie_id'] == mid]
        if len(movie) > 0:
            title = movie['title'].values[0][:25]
            titles.append(title)
        else:
            titles.append(f"Movie {mid}")
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=titles,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale=[[0, '#667eea'], [1, '#764ba2']],
                line=dict(width=0)
            ),
            hovertemplate='%{y}<br>Score: %{x:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Recommendation Scores',
        xaxis_title='Score',
        yaxis_title='',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def format_movie_card(movie_info: dict, score: float = None, 
                       rank: int = None) -> str:
    """
    Format movie information as an HTML card.
    
    Args:
        movie_info: Movie information dictionary
        score: Recommendation score
        rank: Rank in recommendations
        
    Returns:
        HTML string
    """
    title = movie_info.get('title', 'Unknown')
    genres = movie_info.get('genres', [])
    year = movie_info.get('year', 'N/A')
    avg_rating = movie_info.get('avg_rating', 0)
    rating_count = movie_info.get('rating_count', 0)
    
    genres_str = ' • '.join(genres[:3]) if genres else 'Unknown'
    
    # Generate star rating display
    full_stars = int(avg_rating)
    stars = '★' * full_stars + '☆' * (5 - full_stars)
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #1e2130 0%, #2d3250 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    ">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                {"<span style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;'>#{rank}</span>" if rank else ""}
                <h3 style="margin: 8px 0 5px 0; color: #fff;">{title}</h3>
                <p style="color: #999; font-size: 13px; margin: 0;">{genres_str} • {year}</p>
            </div>
            {"<div style='text-align: right;'><span style='color: #667eea; font-size: 18px; font-weight: bold;'>" + f"{score:.2f}" + "</span><br><span style='color: #666; font-size: 11px;'>match score</span></div>" if score else ""}
        </div>
        <div style="margin-top: 12px; display: flex; align-items: center; gap: 15px;">
            <span style="color: #ffc107; font-size: 16px;">{stars}</span>
            <span style="color: #999; font-size: 13px;">{avg_rating:.1f}/5 ({rating_count:,} ratings)</span>
        </div>
    </div>
    """
    
    return card_html


def get_movie_poster_placeholder(genres: list) -> str:
    """
    Generate a placeholder gradient based on genres.
    
    Args:
        genres: List of movie genres
        
    Returns:
        CSS gradient string
    """
    genre_colors = {
        'Action': '#ff4757',
        'Adventure': '#ffa502',
        'Animation': '#2ed573',
        'Comedy': '#1e90ff',
        'Crime': '#2f3542',
        'Documentary': '#57606f',
        'Drama': '#5352ed',
        'Fantasy': '#a55eea',
        'Horror': '#e74c3c',
        'Musical': '#ff6b81',
        'Mystery': '#3d3d3d',
        'Romance': '#ff4757',
        'Sci-Fi': '#00d2d3',
        'Thriller': '#2c3e50',
        'War': '#636e72',
        'Western': '#d35400'
    }
    
    if not genres:
        return 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    
    colors = [genre_colors.get(g, '#667eea') for g in genres[:2]]
    if len(colors) == 1:
        colors.append('#764ba2')
    
    return f'linear-gradient(135deg, {colors[0]} 0%, {colors[1]} 100%)'
