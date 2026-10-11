import pandas as pd
from .config import MOVIES_PATH_RAW, MOVIES_FINAL_PATH

# Builds the movie dataset only. The music catalog is built offline from the
# 278k dataset by train_music_278k.py (see DATA.md).

def assign_mood_to_movie(genres):
    """Assign mood based on movie genres"""
    if pd.isna(genres):
        return "happy"  # default
    
    genres_lower = str(genres).lower()
    
    # Define genre to mood mapping
    if any(word in genres_lower for word in ['horror', 'thriller', 'mystery', 'crime']):
        return "stressed"
    elif any(word in genres_lower for word in ['drama', 'war', 'history']):
        return "sad"
    elif any(word in genres_lower for word in ['action', 'adventure', 'science fiction', 'fantasy']):
        return "energetic"
    elif any(word in genres_lower for word in ['comedy', 'family', 'animation', 'music']):
        return "happy"
    elif any(word in genres_lower for word in ['romance']):
        return "chill"
    else:
        return "happy"  # default

def build_movies_final():
    try:
        movies = pd.read_csv(MOVIES_PATH_RAW)
        
        rename_map = {
            "movie title": "title",
            "plot": "overview", 
            "rating": "rating",
            "votes": "popularity",
            "year": "release_year",
            "poster": "poster_path"
        }
        
        movies = movies.rename(columns=rename_map)
        movies['mood_tag'] = movies['genres'].apply(assign_mood_to_movie)
        
        keep_cols = [c for c in [
            "title", "genres", "overview", "rating", "popularity",
            "release_year", "poster_path", "mood_tag"
        ] if c in movies.columns]
        
        movies_final = movies[keep_cols].copy()
        movies_final = movies_final.drop_duplicates(subset=['title', 'release_year'], keep='first')
        movies_final = movies_final.dropna(subset=['title'])
        
        # Filter quality movies (rating > 5.0, popularity > 100)
        if 'rating' in movies_final.columns:
            movies_final = movies_final[movies_final['rating'] > 5.0]
        if 'popularity' in movies_final.columns:
            movies_final = movies_final[movies_final['popularity'] > 100]
        
        movies_final['poster_path'] = movies_final['poster_path'].fillna('https://via.placeholder.com/300x450?text=No+Image')
        
        movies_final.to_csv(MOVIES_FINAL_PATH, index=False)
        print(f"Saved movies_final.csv with {len(movies_final)} movies")
        print(f"Mood distribution: {movies_final['mood_tag'].value_counts().to_dict()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Building movie dataset...")
    build_movies_final()
    print("\n✅ Dataset building complete!")
