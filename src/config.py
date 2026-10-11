import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Only the project's own .env (no search of parent folders, which could pick up
# an unrelated file). Variables already set in the environment take precedence.
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Raw data paths
RAW_MOVIE_DIR = os.path.join(BASE_DIR, "data", "movie_dataset2")
MOVIES_PATH_RAW = os.path.join(RAW_MOVIE_DIR, "Movies (1970-2023).csv")

# Final cleaned movie file (created by build_datasets.py)
PROCESSED_DIR = os.path.join(BASE_DIR, "data")
MOVIES_FINAL_PATH = os.path.join(PROCESSED_DIR, "movies_final.csv")

# Mood config
MOODS = ["happy", "sad", "angry", "chill", "energetic", "stressed"]

# Label mapping for the 278k-row labeled audio-feature dataset
# (data/music_dataset2/). These four labels are the dataset's ground truth;
# used by train_all_models.py and src/music_pipeline.py.
LABEL_TO_MOOD = {
    0: "sad",
    1: "happy",
    2: "energetic",
    3: "calm",
}

# 278k source dataset (read-only: nothing in the pipeline ever writes to it)
# and the Spotify-track-ID-keyed catalog built from it by train_music_278k.py.
MUSIC_278K_SOURCE_PATH = os.path.join(PROCESSED_DIR, "music_dataset2", "278k_labelled_uri.csv")
MUSIC_CATALOG_278K_PATH = os.path.join(PROCESSED_DIR, "music_catalog_278k.csv")

# Local-only display metadata (title/artist/year) keyed by track_id, built by
# build_music_metadata.py from a downloaded third-party dataset. Gitignored,
# never used for ML, and the recommender works without it. See METADATA.md.
METADATA_DIR = os.path.join(PROCESSED_DIR, "metadata")
MUSIC_METADATA_PATH = os.path.join(METADATA_DIR, "music_metadata.csv")
METADATA_MANIFEST_PATH = os.path.join(METADATA_DIR, "metadata_sources.json")
EXTERNAL_DATA_DIR = os.path.join(PROCESSED_DIR, "external")

# The app offers 6 moods, the 278k music dataset only has the 4 labels above.
# Music targets for each app mood (recommend.py weights earlier ones higher).
# A target is either one of the 4 real labels, or a derived mood from
# DERIVED_MUSIC_MOODS below -- derived moods are feature rules, never labels.
MOOD_TO_MUSIC_TARGETS = {
    "happy": ["happy"],
    "sad": ["sad"],
    "energetic": ["energetic"],
    "chill": ["calm"],
    "angry": ["angry"],
    "stressed": ["stressed"],
}

# Derived music moods for app moods the 278k dataset has no label for.
# `rule`: (feature, op, percentile) -- the percentile is taken over the whole
# 278k catalog and turned into a concrete cutoff at build time (recorded in
# models/music_mood_classifier_278k_eval.json).
# `guards` (optional): named groups of absolute (feature, op, value) checks;
# a track matching any of them is never a candidate.
# Ranking, one of:
#   `score`   -- "extreme": mean of the listed features' percentile ranks
#                (+1 = higher is more on-mood, -1 = lower is).
#   `profile` -- "closeness": 1 - mean absolute distance, in percentile-rank
#                space, to the median percentile of the (guarded) candidates
#                on the listed features; the median profile is recorded at
#                build time. Prefers typical members over extreme outliers.
# The dataset's labels are never changed.
DERIVED_MUSIC_MOODS = {
    "angry": {
        "rule": [("valence", "<=", 0.25), ("energy", ">=", 0.75), ("loudness", ">=", 0.75)],
        "score": {"valence": -1, "energy": +1, "loudness": +1},
        "experimental": False,
        "note": "Negative, high-arousal, loud. Mostly energetic-labelled tracks; "
                "cannot separate aggressive from merely dark/intense music.",
    },
    "stressed": {
        "rule": [("valence", "<=", 0.40), ("energy", ">=", 0.50), ("energy", "<", 0.75),
                 ("danceability", "<=", 0.50)],
        "guards": {
            # danceability, tempo and speechiness are exactly 0 together on the
            # same 90 tracks (short, mostly sad/calm) -- the audio analysis
            # failed; valence == 0 adds 45 more.
            "failed_audio_analysis": [("danceability", "==", 0.0), ("tempo", "==", 0.0), ("valence", "==", 0.0)],
            # Spotify's documented threshold for "probably made entirely of spoken words".
            "spoken_word": [("speechiness", ">", 0.66)],
        },
        "profile": ["valence", "energy", "danceability", "loudness"],
        "experimental": True,
        "note": "EXPERIMENTAL, ambiguous: matching tracks span the sad, happy and "
                "energetic labels. Not a ground-truth class. Ranked by closeness to "
                "the typical candidate, because ranking by extremeness surfaced "
                "failed-analysis and near-silent outliers. Tempo is not a rule gate "
                "(weak separation, octave errors); tempo == 0 only marks failed analysis.",
    },
}

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ordered target mood(s) for "Match my mood". recommend.py always tries the
# mood itself first (works automatically once ML-labeled data for that mood
# exists, e.g. music now has genuine "angry"-tagged tracks); these lists are
# secondary fallback moods, weighted lower, used when a dataset has no
# signal for the mood itself (e.g. movies have no "angry" category at all,
# so movie recommendations for "angry" lean on energetic/stressed content).
MOOD_TO_TAGS_MATCH = {
    "happy": ["happy"],
    "sad": ["sad"],
    "angry": ["energetic", "stressed"],
    "chill": ["chill"],
    "energetic": ["energetic"],
    "stressed": ["stressed"],
}

# Ordered target mood(s) for "Cheer me up" -- deliberately distinct per
# source mood (not just [happy, chill] for everything) so the uplifting
# recommendation actually reflects *what* you're being cheered up from:
# stressed needs calming down, angry needs cooling off before cheer, sad
# gets warmth, chill gets a gentle nudge toward energy. Order matters --
# recommend.py weights earlier moods higher, so distinct orderings (not
# just distinct sets) produce genuinely different rankings.
MOOD_TO_TAGS_CHEER = {
    "happy": ["happy", "energetic"],
    "energetic": ["energetic", "happy"],
    "chill": ["energetic", "chill"],   # nudge chill toward more energy, keep some chill flavor
    "sad": ["happy", "chill"],         # warmth + comfort
    "angry": ["chill", "happy"],       # cool off first, then lift
    "stressed": ["chill"],             # pure calm-down, no "happy party" content
}
# every entry above is a distinct ordered list -- no two source moods route
# to the same ranking (verified in the training/eval smoke tests)

# TMDB API Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # Get from https://www.themoviedb.org/settings/api
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Seconds before an outbound HTTP request (TMDB dataset builder) is abandoned.
REQUEST_TIMEOUT = 15
