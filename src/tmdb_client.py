import requests
import pandas as pd
from datetime import datetime, timedelta
from .config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE_URL, MOVIES_FINAL_PATH, REQUEST_TIMEOUT

class TMDBClient:
    def __init__(self):
        self.api_key = TMDB_API_KEY
        self.base_url = TMDB_BASE_URL
        self.session = requests.Session()
    
    def get_popular_movies(self, pages=5):
        """Fetch popular movies from TMDB"""
        movies = []
        for page in range(1, pages + 1):
            url = f"{self.base_url}/movie/popular"
            params = {"api_key": self.api_key, "page": page}
            
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                movies.extend(data["results"])
        
        return self._process_movies(movies)
    
    def get_new_releases(self, days_back=30):
        """Fetch movies released in the last N days"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/discover/movie"
        params = {
            "api_key": self.api_key,
            "primary_release_date.gte": start_date,
            "primary_release_date.lte": end_date,
            "sort_by": "popularity.desc"
        }
        
        response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            movies = response.json()["results"]
            return self._process_movies(movies)
        return []
    
    def _process_movies(self, movies):
        """Convert TMDB movie data to our format"""
        processed = []
        for movie in movies:
            # Map genres to mood categories
            genres = self._get_movie_genres(movie["id"])
            mood = self._assign_mood_from_genres(genres)
            
            processed.append({
                "title": movie["title"],
                "genres": "|".join(genres),
                "rating": movie["vote_average"],
                "popularity": movie["popularity"],
                "poster_path": f"{TMDB_IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else "",
                "mood": mood,
                "release_date": movie.get("release_date", ""),
                "overview": movie.get("overview", "")
            })
        
        return processed
    
    def _get_movie_genres(self, movie_id):
        """Get detailed genre names for a movie"""
        url = f"{self.base_url}/movie/{movie_id}"
        params = {"api_key": self.api_key}
        
        response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return [genre["name"] for genre in data.get("genres", [])]
        return []
    
    def _assign_mood_from_genres(self, genres):
        """Assign mood based on movie genres"""
        genre_mood_map = {
            "Horror": "stressed",
            "Thriller": "stressed", 
            "Drama": "sad",
            "War": "sad",
            "Action": "energetic",
            "Adventure": "energetic",
            "Comedy": "happy",
            "Family": "happy",
            "Romance": "chill",
            "Documentary": "chill"
        }
        
        for genre in genres:
            if genre in genre_mood_map:
                return genre_mood_map[genre]
        
        return "chill"  # Default mood
    
    def update_movie_dataset(self):
        """Update the existing movie dataset with new releases"""
        try:
            # Load existing dataset
            existing_df = pd.read_csv(MOVIES_FINAL_PATH)
            existing_titles = set(existing_df["title"].str.lower())
        except FileNotFoundError:
            existing_df = pd.DataFrame()
            existing_titles = set()
        
        # Get new movies
        new_movies = self.get_new_releases()
        
        # Filter out duplicates
        unique_movies = [
            movie for movie in new_movies 
            if movie["title"].lower() not in existing_titles
        ]
        
        if unique_movies:
            new_df = pd.DataFrame(unique_movies)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_csv(MOVIES_FINAL_PATH, index=False)
            print(f"Added {len(unique_movies)} new movies to dataset")
        else:
            print("No new movies to add")
        
        return len(unique_movies)