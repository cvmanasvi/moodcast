#!/usr/bin/env python3
"""
Script to update movie dataset with new releases from TMDB API
Run this script weekly/daily to keep your movie recommendations fresh
"""

from src.tmdb_client import TMDBClient

def main():
    print("Updating movie dataset with TMDB API...")
    
    client = TMDBClient()
    new_count = client.update_movie_dataset()
    
    if new_count > 0:
        print(f"✅ Successfully added {new_count} new movies!")
    else:
        print("ℹ️ No new movies found")

if __name__ == "__main__":
    main()