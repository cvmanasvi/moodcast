import logging
import os

import pandas as pd
from .config import MOVIES_FINAL_PATH, MOVIES_PATH_RAW, MUSIC_CATALOG_278K_PATH, MUSIC_METADATA_PATH
from .music_metadata import MetadataBuildError, load_metadata_store

logger = logging.getLogger("moodcast")

_movies_df = None
_movie_imdb_ids = None
_music_catalog_df = None
_music_metadata = None
_music_metadata_loaded = False


class DatasetUnavailable(Exception):
    """A dataset a feature needs is not present on this machine. These files are
    not distributed with the repository (see DATA.md); the message says why and
    how to provide one, and is safe to show to users."""


MOVIES_UNAVAILABLE = ("Movie recommendations are unavailable: they need data/movies_final.csv, which is "
                      "not distributed with this repository because its source and licence are "
                      "unverified (see DATA.md).")
MUSIC_UNAVAILABLE = ("Music recommendations are unavailable: the music catalog (data/music_catalog_278k.csv) "
                     "has not been built on this machine. Download the 278k dataset and run "
                     "python train_music_278k.py (see DATA.md).")


def load_movie_data():
    global _movies_df
    if _movies_df is None:
        if not os.path.exists(MOVIES_FINAL_PATH):
            raise DatasetUnavailable(MOVIES_UNAVAILABLE)
        _movies_df = pd.read_csv(MOVIES_FINAL_PATH)
    return _movies_df

def load_movie_imdb_ids():
    """
    {movies_final row index: IMDb numeric id}, recovered from the raw dataset
    the way build_datasets.py built movies_final (first raw row per title +
    year). Used only to find the right TMDB movie or TV series for display
    details; empty if the raw file is missing or unreadable.
    """
    global _movie_imdb_ids
    if _movie_imdb_ids is None:
        try:
            raw = pd.read_csv(MOVIES_PATH_RAW, usecols=["movie title", "year", "imdb id"])
            raw = raw.drop_duplicates(subset=["movie title", "year"], keep="first")
            ids = {(t, y): int(i) for t, y, i in raw[["movie title", "year", "imdb id"]].itertuples(index=False)
                   if pd.notna(i) and pd.notna(y)}
            movies = load_movie_data()
            _movie_imdb_ids = {idx: ids[(t, y)] for idx, t, y in movies[["title", "release_year"]].itertuples()
                               if (t, y) in ids}
        except (OSError, ValueError, KeyError):
            logger.exception("Ignoring unreadable raw movie data at %s; matching TMDB by title only", MOVIES_PATH_RAW)
            _movie_imdb_ids = {}
    return _movie_imdb_ids

def load_music_catalog():
    """The 278k Spotify-track-ID-keyed catalog built by train_music_278k.py:
    the app's only music catalog."""
    global _music_catalog_df
    if _music_catalog_df is None:
        if not os.path.exists(MUSIC_CATALOG_278K_PATH):
            raise DatasetUnavailable(MUSIC_UNAVAILABLE)
        _music_catalog_df = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str})
    return _music_catalog_df

def load_music_metadata():
    """
    Optional local display metadata (a MetadataStore keyed by track_id), read
    once per process. Returns None if data/metadata/music_metadata.csv hasn't
    been built (python build_music_metadata.py) or is unreadable -- callers
    must then fall back to placeholder titles. Recommendations never depend
    on it. Built after the app started? Restart to pick it up.
    """
    global _music_metadata, _music_metadata_loaded
    if not _music_metadata_loaded:
        try:
            _music_metadata = load_metadata_store(MUSIC_METADATA_PATH)
        except (MetadataBuildError, OSError, ValueError):
            logger.exception("Ignoring unreadable music metadata at %s; serving without it", MUSIC_METADATA_PATH)
            _music_metadata = None
        _music_metadata_loaded = True
    return _music_metadata
