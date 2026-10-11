#!/usr/bin/env python3
"""
Build complete movie dataset from TMDB API
This will replace your existing movie dataset with fresh TMDB data
"""

import requests
import pandas as pd
import time
from src.config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE_URL, MOVIES_FINAL_PATH, REQUEST_TIMEOUT

def fetch_tmdb_movies(total_pages=100):
    """Fetch comprehensive movie dataset from TMDB"""
    movies = []
    
    print(f"Fetching {total_pages} pages of movies from TMDB...")
    
    # Fetch popular movies globally
    for page in range(1, total_pages + 1):
        print(f"Fetching global popular page {page}/{total_pages}...")
        
        url = f"{TMDB_BASE_URL}/movie/popular"
        params = {"api_key": TMDB_API_KEY, "page": page}
        
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                for movie in data["results"]:
                    processed_movie = process_movie(movie)
                    if processed_movie:
                        movies.append(processed_movie)
            
            time.sleep(0.25)
            
        except Exception as e:
            print(f"Error on page {page}: {type(e).__name__} (details suppressed: request URLs contain the API key)")
            continue
    
    # Fetch Indian movies specifically
    print("Fetching Indian movies...")
    indian_movies = fetch_indian_movies()
    movies.extend(indian_movies)
    
    return movies

def fetch_indian_movies():
    """Fetch Indian movies from TMDB"""
    movies = []
    
    # Search for Bollywood and regional Indian cinema
    indian_searches = [
        "bollywood", "hindi movies", "tamil movies", "telugu movies", 
        "malayalam movies", "kannada movies", "punjabi movies", "marathi movies",
        "bengali movies", "gujarati movies", "rajasthani movies"
    ]
    
    for search_term in indian_searches:
        print(f"Searching for: {search_term}")
        
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": search_term,
            "page": 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                for movie in data["results"][:10]:  # Limit to top 10 per search
                    processed_movie = process_movie(movie)
                    if processed_movie:
                        movies.append(processed_movie)
            
            time.sleep(0.25)
            
        except Exception as e:
            print(f"Error searching {search_term}: {type(e).__name__} (details suppressed: request URLs contain the API key)")
            continue
    
    # Also fetch movies from India region
    print("Fetching movies by Indian region...")
    url = f"{TMDB_BASE_URL}/discover/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "with_origin_country": "IN",  # India
        "sort_by": "popularity.desc",
        "page": 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            for movie in data["results"]:
                processed_movie = process_movie(movie)
                if processed_movie:
                    movies.append(processed_movie)
    except Exception as e:
        print(f"Error fetching Indian region movies: {type(e).__name__} (details suppressed: request URLs contain the API key)")
    
    return movies

def process_movie(movie):
    """Process individual movie data"""
    try:
        # Get detailed movie info including genres
        movie_id = movie["id"]
        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY}
        
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            print(f"Failed to get details for {movie.get('title', 'Unknown')}: {response.status_code}")
            return None
            
        details = response.json()
        
        # Extract genres
        genres = [genre["name"] for genre in details.get("genres", [])]
        
        # Assign mood based on genres
        mood = assign_mood_from_genres(genres)
        
        processed = {
            "title": movie["title"],
            "genres": "|".join(genres),
            "rating": movie["vote_average"],
            "popularity": movie["popularity"],
            "poster_path": f"{TMDB_IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else "",
            "mood": mood,
            "release_date": movie.get("release_date", ""),
            "overview": movie.get("overview", "")[:500] if movie.get("overview") else "",
            "runtime": details.get("runtime", 0),
            "vote_count": movie.get("vote_count", 0)
        }
        
        print(f"Processed: {processed['title']}")
        return processed
        
    except Exception as e:
        print(f"Error processing movie {movie.get('title', 'Unknown')}: {type(e).__name__} (details suppressed: request URLs contain the API key)")
        return None

def assign_mood_from_genres(genres):
    """Assign mood based on movie genres"""
    genre_mood_map = {
        "Horror": "stressed",
        "Thriller": "stressed",
        "Mystery": "stressed",
        "Drama": "sad",
        "War": "sad",
        "History": "sad",
        "Action": "energetic",
        "Adventure": "energetic",
        "Science Fiction": "energetic",
        "Comedy": "happy",
        "Family": "happy",
        "Animation": "happy",
        "Romance": "chill",
        "Music": "chill",
        "Documentary": "chill",
        "Fantasy": "chill",
        # Indian genre mappings
        "Bollywood": "happy",  # Default Bollywood to happy
        "Musical": "happy",
        "Dance": "energetic"
    }
    
    # Priority order for mood assignment
    mood_priority = ["stressed", "energetic", "happy", "sad", "chill"]
    
    for mood in mood_priority:
        for genre in genres:
            if genre in genre_mood_map and genre_mood_map[genre] == mood:
                return mood
    
    return "chill"  # Default mood

def fetch_top_rated_movies(pages=20):
    """Fetch top rated movies for better quality dataset"""
    movies = []
    
    print(f"Fetching {pages} pages of top-rated movies...")
    
    for page in range(1, pages + 1):
        url = f"{TMDB_BASE_URL}/movie/top_rated"
        params = {"api_key": TMDB_API_KEY, "page": page}
        
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                for movie in data["results"]:
                    processed_movie = process_movie(movie)
                    if processed_movie:
                        movies.append(processed_movie)
            
            time.sleep(0.25)
            
        except Exception as e:
            print(f"Error on page {page}: {type(e).__name__} (details suppressed: request URLs contain the API key)")
            continue
    
    return movies

def main():
    print("🎬 Building comprehensive movie dataset from TMDB...")
    
    # Check API key
    if TMDB_API_KEY == "your_api_key_here":
        print("❌ Please set your TMDB API key in src/config.py")
        return
    
    all_movies = []
    
    # Fetch popular movies (broader selection)
    popular_movies = fetch_tmdb_movies(total_pages=10)  # Reduced for testing
    all_movies.extend(popular_movies)
    
    # Fetch top-rated movies (higher quality)
    top_rated_movies = fetch_top_rated_movies(pages=5)  # Reduced for testing
    all_movies.extend(top_rated_movies)
    
    print(f"Total movies fetched: {len(all_movies)}")
    
    if not all_movies:
        print("❌ No movies were fetched. Check your API key and internet connection.")
        return
    
    # Remove duplicates based on title
    seen_titles = set()
    unique_movies = []
    for movie in all_movies:
        if movie and movie["title"].lower() not in seen_titles:
            seen_titles.add(movie["title"].lower())
            unique_movies.append(movie)
    
    print(f"Unique movies after deduplication: {len(unique_movies)}")
    
    # Create DataFrame and save
    df = pd.DataFrame(unique_movies)
    
    if df.empty:
        print("❌ No valid movies to save")
        return
    
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"DataFrame shape: {df.shape}")
    
    # Filter out movies with very low ratings or vote counts (if columns exist)
    if "rating" in df.columns and "vote_count" in df.columns:
        df = df[(df["rating"] >= 4.0) & (df["vote_count"] >= 10)]
    
    # Sort by popularity and rating (if columns exist)
    if "popularity" in df.columns and "rating" in df.columns:
        df = df.sort_values(["popularity", "rating"], ascending=[False, False])
    
    # Save to CSV
    df.to_csv(MOVIES_FINAL_PATH, index=False)
    
    print(f"✅ Successfully created dataset with {len(df)} movies!")
    if "mood" in df.columns:
        print(f"📊 Mood distribution:")
        print(df["mood"].value_counts())
    print(f"💾 Saved to: {MOVIES_FINAL_PATH}")

if __name__ == "__main__":
    main()