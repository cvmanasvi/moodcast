import ast
import logging
import os
import re

import pandas as pd
from flask import Flask, render_template, jsonify, request
from werkzeug.exceptions import HTTPException

from src.config import MOODS
from src.data_loader import (DatasetUnavailable, load_movie_data, load_movie_imdb_ids, load_music_catalog,
                             load_music_metadata)
from src import artist_priority, movie_details, popular_artists, spotify_oembed
from src.music_metadata import missing_display, spotify_track_url
from src.recommend import recommend_music, recommend_movies
from src.mood_utils import text_to_mood_with_confidence

logger = logging.getLogger("moodcast")

app = Flask(__name__, template_folder='templates', static_folder='static')

# ---- API input constraints ----
VALID_MOODS = set(MOODS)
MODE_TO_RECOMMENDER_MODE = {'match': "Match my mood", 'uplift': "Cheer me up"}
VALID_TYPES = {'music', 'movies'}
DEFAULT_LIMIT = 30
MAX_LIMIT = 100
MAX_TEXT_LENGTH = 1000
MAX_QUERY_LENGTH = 100
MAX_EXCLUDE = 600          # ids a refresh may ask to avoid (recently shown items)
MAX_REFRESH_DRAWS = 6      # extra recommender draws a refresh may use to find unseen items
MAX_DETAIL_IDS = 60        # ids per /api/track-details or /api/movie-details call
_TRACK_ID_RE = re.compile(r'^[0-9A-Za-z]{22}$')


class ApiError(Exception):
    """Client-side error in an /api request; rendered as a JSON 4xx response."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


@app.errorhandler(ApiError)
def handle_api_error(e):
    return jsonify({'error': e.message}), e.status


@app.errorhandler(DatasetUnavailable)
def handle_dataset_unavailable(e):
    # A dataset that isn't distributed with the repository hasn't been provided
    # locally: say so plainly (the message names the file and DATA.md).
    return jsonify({'error': str(e), 'dataset_unavailable': True}), 503


@app.errorhandler(HTTPException)
def handle_http_error(e):
    # JSON for API routes so the frontend can read the error; normal HTML
    # error pages everywhere else.
    if request.path.startswith('/api/'):
        return jsonify({'error': e.description}), e.code
    return e


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    # Full traceback goes to the server log; the client only gets a generic
    # message so internals (paths, data, exception text) aren't leaked.
    logger.exception("Unhandled error on %s %s", request.method, request.path)
    return jsonify({'error': 'Internal server error'}), 500


def _json_object_body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("Request body must be a JSON object (Content-Type: application/json).")
    return data


def _validate_choice(data, field, allowed, default=None):
    value = data.get(field, default)
    if value is None:
        raise ApiError(f"'{field}' is required. Allowed values: {sorted(allowed)}.")
    if not isinstance(value, str) or value not in allowed:
        raise ApiError(f"Invalid '{field}'. Allowed values: {sorted(allowed)}.")
    return value


def _validate_limit(data):
    limit = data.get('limit', DEFAULT_LIMIT)
    # bool is a subclass of int, so reject it explicitly
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ApiError(f"'limit' must be an integer between 1 and {MAX_LIMIT}.")
    return limit


def _validate_exclude(data, media_type):
    """Optional list of recently shown items to avoid on a refresh: track_ids for music,
    movie_ids for movies."""
    exclude = data.get('exclude', [])
    if not isinstance(exclude, list) or len(exclude) > MAX_EXCLUDE:
        raise ApiError(f"'exclude' must be a list of at most {MAX_EXCLUDE} ids.")
    if media_type == 'music':
        if not all(isinstance(t, str) and _TRACK_ID_RE.match(t) for t in exclude):
            raise ApiError("'exclude' must contain Spotify track ids for music.")
    elif not all(isinstance(m, int) and not isinstance(m, bool) and m >= 0 for m in exclude):
        raise ApiError("'exclude' must contain movie ids (non-negative integers) for movies.")
    return set(exclude)


def _fresh_results(draw, key, exclude, limit):
    """
    Up to `limit` results not in `exclude`, for a "refresh". Each draw is an
    ordinary, unmodified call to the recommender for the same mood and mode;
    items already shown are skipped and the list is topped up from further
    draws, in the order the recommender returned them. Stops after
    MAX_REFRESH_DRAWS, so a small eligible pool yields fewer (reported) results
    instead of repeats.
    """
    seen, out, draws = set(exclude), [], 0
    while len(out) < limit and draws < MAX_REFRESH_DRAWS:
        draws += 1
        for item in draw():
            if item[key] in seen:
                continue
            seen.add(item[key])
            out.append(item)
            if len(out) == limit:
                break
    return out, {'excluded': len(exclude), 'draws': draws, 'returned': len(out), 'exhausted': len(out) < limit}


def _id_list(values, valid, what):
    if not isinstance(values, list) or not values or len(values) > MAX_DETAIL_IDS or not all(valid(v) for v in values):
        raise ApiError(f"Provide 1-{MAX_DETAIL_IDS} valid {what}.")
    return list(dict.fromkeys(values))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/movies')
def movies():
    return render_template('movies.html')

@app.route('/music')
def music():
    return render_template('music.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/detect-mood', methods=['POST'])
def detect_mood():
    """Text -> mood, for the landing page's 'describe your mood' box."""
    data = _json_object_body()
    text = data.get('text')
    if not isinstance(text, str) or not text.strip():
        raise ApiError("'text' is required and must be a non-empty string.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ApiError(f"'text' must be at most {MAX_TEXT_LENGTH} characters.")
    mood, confidence, source = text_to_mood_with_confidence(text.strip())
    # `uncertain`: the mood is only the model's best guess (low confidence or no known words)
    return jsonify({'mood': mood, 'confidence': confidence, 'source': source,
                    'uncertain': source != 'ml_classifier'})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = _json_object_body()
    mood = _validate_choice(data, 'mood', VALID_MOODS)
    mode = _validate_choice(data, 'mode', set(MODE_TO_RECOMMENDER_MODE), default='match')
    media_type = _validate_choice(data, 'type', VALID_TYPES)
    limit = _validate_limit(data)

    exclude = _validate_exclude(data, media_type)

    recommendation_mode = MODE_TO_RECOMMENDER_MODE[mode]
    if media_type == 'movies':
        draw, key = (lambda: get_movie_recommendations(mood, recommendation_mode, top_n=limit)), 'movie_id'
    else:
        draw, key = (lambda: get_music_recommendations(mood, recommendation_mode, top_n=limit)), 'track_id'

    body = {'mood': mood, 'mode': mode}
    if exclude:
        body['results'], body['refresh'] = _fresh_results(draw, key, exclude, limit)
    else:
        body['results'] = draw()
    return jsonify(body)


@app.route('/api/popular', methods=['POST'])
def popular():
    """Popular Artists mode: its own local dataset and pipeline (src/popular_artists.py),
    separate from /api/recommend. Reports `available: false` when the data isn't prepared."""
    data = _json_object_body()
    mood = _validate_choice(data, 'mood', VALID_MOODS)
    mode = _validate_choice(data, 'mode', set(MODE_TO_RECOMMENDER_MODE), default='match')
    limit = min(_validate_limit(data), popular_artists.MAX_RESULTS)
    exclude = _validate_exclude(data, 'music')

    body = {'mood': mood, 'mode': mode, 'limit': limit}
    df = popular_artists.load_prepared()
    if df is None:
        body.update(available=False, results=[], message=(
            "Popular Artists needs its local dataset, which hasn't been prepared on this server "
            "(see DATA.md: python prepare_popular_artists.py)."))
        return jsonify(body)
    rows, info = popular_artists.recommend_popular(df, mood, MODE_TO_RECOMMENDER_MODE[mode], limit, exclude)
    body.update(available=True, results=[_popular_track_fields(track, mood) for _, track in rows.iterrows()],
                eligible=info['eligible'], limited=info['limited'], limited_by=info['limited_by'])
    if exclude:
        body['refresh'] = {'excluded': len(exclude), 'returned': info['returned'], 'exhausted': info['limited']}
    return jsonify(body)


@app.route('/api/track-details')
def track_details():
    """Real title + cover art for displayed tracks from Spotify's public oEmbed
    endpoint (no login/Premium; no artist field). Display only."""
    ids = _id_list([i for i in (request.args.get('ids') or '').split(',') if i],
                   lambda t: bool(_TRACK_ID_RE.match(t)), 'Spotify track ids in ?ids=')
    found = spotify_oembed.lookup_many(ids)
    return jsonify({
        'enabled': spotify_oembed.enabled(),
        'source': 'Spotify oEmbed',
        'tracks': {t: {'title': r['title'], 'artwork': r['artwork']} for t, r in found.items()},
    })


@app.route('/api/movie-details', methods=['POST'])
def movie_details_route():
    """TMDB poster and India watch providers for dataset titles (films or TV
    series), found by IMDb id or else exact title + year (see
    src/movie_details.py). The API key stays server-side."""
    data = _json_object_body()
    movie_df = load_movie_data()
    ids = _id_list(data.get('movie_ids'),
                   lambda m: isinstance(m, int) and not isinstance(m, bool) and m in movie_df.index, "'movie_ids'")
    imdb_ids = load_movie_imdb_ids()
    items = []
    for movie_id in ids:
        row = movie_df.loc[movie_id]
        year = int(row['release_year']) if pd.notna(row.get('release_year')) else None
        items.append((movie_id, clean_movie_title(row.get('title', '')), year, imdb_ids.get(movie_id)))
    found = movie_details.lookup_many(items)
    return jsonify({
        'region': movie_details.REGION,
        'enabled': movie_details.enabled(),
        'attribution': movie_details.ATTRIBUTION,
        'movies': {str(m): r for m, r in found.items()},
    })

_all_genres_cache = None

def _get_all_genres(movie_df):
    """Lazily-built, cached set of individual movie genres (genres is
    stored as a Python-list-repr string)."""
    global _all_genres_cache
    if _all_genres_cache is None:
        genres = set()
        for val in movie_df['genres'].dropna():
            try:
                parsed = ast.literal_eval(str(val))
                if isinstance(parsed, (list, tuple)):
                    for g in parsed:
                        genres.add(str(g))
            except (ValueError, SyntaxError):
                continue
        _all_genres_cache = sorted(genres)
    return _all_genres_cache

@app.route('/api/search')
def search():
    """Universal search across movies, songs, artists and genres for the
    navbar search widget: case-insensitive substring matching. Songs and
    artists come from the local music metadata (keyed by track_id); with no
    metadata built, those two groups are simply empty."""
    q = (request.args.get('q') or '').strip()
    if len(q) > MAX_QUERY_LENGTH:
        raise ApiError(f"'q' must be at most {MAX_QUERY_LENGTH} characters.")
    if len(q) < 2:
        return jsonify({'movies': [], 'songs': [], 'artists': [], 'genres': []})
    ql = q.lower()
    try:
        movie_df = load_movie_data()
    except DatasetUnavailable:
        movie_df = None  # no local movie data: the movie and genre groups are empty

    movie_matches = (movie_df[movie_df['title'].str.lower().str.contains(ql, na=False, regex=False)]
                     if movie_df is not None else pd.DataFrame())
    if 'popularity' in movie_matches.columns:
        movie_matches = movie_matches.sort_values('popularity', ascending=False)
    movies_out = []
    for _, mv in movie_matches.head(6).iterrows():
        release_year = mv.get('release_year')
        rating = mv.get('rating', 0)
        movies_out.append({
            'title': clean_movie_title(mv.get('title', 'Unknown')),
            'year': str(int(release_year)) if pd.notna(release_year) else 'N/A',
            'image': poster_url(mv.get('poster_path')),
            'rating': f"{float(rating):.1f}" if pd.notna(rating) and rating > 0 else 'N/A',
        })

    metadata = load_music_metadata()
    songs_out, artists_out = [], []
    if metadata is not None:
        for hit in metadata.search_tracks(q, limit=6):
            songs_out.append({
                'track_id': hit['track_id'],
                'title': hit['title'],
                'artist': hit['artist'],
                'artists': hit['artists'],
                'year': hit['release_year'],
                'release_year': hit['release_year'],
                'image': '',  # no artwork source yet
                'spotify_url': spotify_track_url(hit['track_id']),
            })
        # whole artist names from artist_names_json -- never split on commas
        artists_out = [{'name': a} for a in metadata.search_artists(q, limit=6)]
    genres_out = ([{'name': g} for g in _get_all_genres(movie_df) if ql in g.lower()][:6]
                  if movie_df is not None else [])

    return jsonify({'movies': movies_out, 'songs': songs_out, 'artists': artists_out, 'genres': genres_out})

def _extract_audio_features(track):
    """Pull the per-track audio features (valence, energy, danceability,
    tempo, acousticness -- all 0-1 scale except tempo which is BPM) out of a
    278k catalog row. Returns None for any field that's missing."""
    def safe_float(val, ndigits=3):
        return round(float(val), ndigits) if pd.notna(val) else None
    return {
        'valence': safe_float(track.get('valence')),
        'energy': safe_float(track.get('energy')),
        'danceability': safe_float(track.get('danceability')),
        'tempo': safe_float(track.get('tempo'), 1),
        'acousticness': safe_float(track.get('acousticness')),
    }

def _catalog_track_fields(track, metadata):
    """Display fields for a 278k-catalog track: looked up in the optional
    local metadata by exact track_id, with placeholders when it has none.
    Everything else (duration, mood, audio features) stays from the catalog."""
    track_id = str(track.get('track_id'))
    display = metadata.display(track_id) if metadata is not None else missing_display()
    return {
        'track_id': track_id,
        'title': display['title'],
        'artist': display['artist'],
        'artists': display['artists'],
        'album': None,  # the metadata source has no album field
        'image': '',    # no artwork source yet
        'year': display['release_year'],
        'release_year': display['release_year'],
        'has_metadata': display['has_metadata'],
        'spotify_url': spotify_track_url(track_id),
        'duration_ms': int(track['duration_ms']) if pd.notna(track.get('duration_ms')) else None,
    }


def _popular_track_fields(track, mood):
    """A Popular Artists track in the music card's shape. Title and artist string are
    the source dataset's, verbatim (the artist string is never split); the Spotify link
    is built from the validated track_id. No artist category is sent."""
    track_id = str(track['track_id'])
    year = int(track['year']) if pd.notna(track.get('year')) else None
    duration_ms = int(track['duration_ms']) if pd.notna(track.get('duration_ms')) else None
    return {
        'track_id': track_id,
        'title': track['track_name'],
        'artist': track['artist_name'],
        'album': None,
        'image': '',    # covers come only from the oEmbed lookup (/api/track-details)
        'year': year,
        'release_year': year,
        'has_metadata': True,
        'spotify_url': spotify_track_url(track_id),
        'duration_ms': duration_ms,
        'duration': format_duration(duration_ms),
        'mood': mood,
        # the classifier probability for happy/sad/energetic/calm; angry/stressed are rule-based
        'mood_confidence': float(track['score']) if track['target'] in popular_artists.BASE_TARGETS else None,
        'color': get_mood_color(mood),
        'audio_features': _extract_audio_features(track),
    }


def _prioritize_artists(music_df, recs, metadata):
    """Optional preferred-artist swaps (src/artist_priority.py) on the recommender's list;
    the list is served unchanged if the layer is off, has no metadata, or fails."""
    config = artist_priority.load_config()
    if config is None or metadata is None:
        return recs
    try:
        return artist_priority.prioritize(music_df, recs, artist_priority.get_index(music_df, metadata, config), config)
    except Exception as exc:  # never fail a request over the optional layer
        logger.warning("Artist prioritization skipped (%s); serving the recommender's list", type(exc).__name__)
        return recs


def get_music_recommendations(mood, mode, top_n):
    music_df = load_music_catalog()
    recs = recommend_music(music_df, mood, mode, top_n=top_n)
    metadata = load_music_metadata()
    recs = _prioritize_artists(music_df, recs, metadata)
    # Enrich with the optional local metadata by track_id. Ranking and
    # membership are already fixed here.

    results = []
    for _, track in recs.iterrows():
        fields = _catalog_track_fields(track, metadata)
        results.append({
            **fields,
            'duration': format_duration(fields['duration_ms']),  # null if missing, never a placeholder
            'genre': None,
            'mood': str(track.get('mood', mood)),
            'mood_confidence': float(track.get('mood_confidence', 0)) if pd.notna(track.get('mood_confidence')) else None,
            'color': get_mood_color(mood),
            'audio_features': _extract_audio_features(track),
            # set only for tracks whose local-metadata artist is in config/artist_priority.json
            'artist_category': track.get('artist_category') if isinstance(track.get('artist_category'), str) else None,
            'priority_artist': track.get('priority_artist') if isinstance(track.get('priority_artist'), str) else None,
        })

    logger.debug("Returning %d music tracks for mood=%s, mode=%s", len(results), mood, mode)
    return results

def get_movie_recommendations(mood, mode, top_n):
    movie_df = load_movie_data()
    recs = recommend_movies(movie_df, mood, mode, top_n=top_n)

    results = []
    for movie_id, movie in recs.iterrows():
        title = clean_movie_title(movie.get('title', 'Unknown Movie'))
        overview = movie.get('overview')
        poster = movie.get('poster_path', '')
        rating = movie.get('rating', 0)
        release_year = movie.get('release_year')
        year = str(int(release_year)) if pd.notna(release_year) else 'N/A'
        runtime = movie.get('runtime_min')
        genres = parse_genre_list(movie.get('genres'))

        results.append({
            'movie_id': int(movie_id),  # row of the movie dataset: stable id for refresh/details
            'title': title,
            'description': format_overview(overview),
            'overview': full_overview(overview),
            'image': poster_url(poster),
            'rating': f"{float(rating):.1f}" if pd.notna(rating) and rating > 0 else 'N/A',
            'year': year,
            'duration': f"{int(runtime)} min" if pd.notna(runtime) and runtime > 0 else 'N/A',
            'genre': genres,
            'mood_confidence': float(movie.get('mood_confidence', 0)) if pd.notna(movie.get('mood_confidence')) else None,
        })

    logger.debug("Returning %d movies for mood=%s, mode=%s", len(results), mood, mode)
    return results

def clean_movie_title(title):
    """Titles in the source data carry a baked-in ': (YYYY)' suffix (e.g.
    'Jeevan Mrityu: (1970)') even though release_year is already a separate
    column -- strip it so the frontend shows a clean title."""
    return re.sub(r":\s*\(\d{4}\)\s*$", "", str(title)).strip()


def format_overview(overview):
    """movie.get('overview') returns NaN (a float), not None, for the ~36
    movies with a missing plot -- str(nan) renders as the literal text 'nan'
    if not handled explicitly."""
    if overview is None or pd.isna(overview) or not str(overview).strip():
        return "No description available."
    text = str(overview)
    return text[:200] + "..." if len(text) > 200 else text


def full_overview(overview):
    """The complete plot text, or '' when the dataset has none (never a placeholder)."""
    if overview is None or pd.isna(overview) or not str(overview).strip():
        return ''
    return str(overview).strip()


def parse_genre_list(genres_raw):
    """genres is stored as a Python-list repr string, e.g. "['Action', 'Drama']" --
    parse it into a clean comma-separated string instead of showing the raw repr.
    Missing genres stay empty ('') rather than being given a made-up default."""
    if genres_raw is None or pd.isna(genres_raw):
        return ""
    try:
        parsed = ast.literal_eval(str(genres_raw))
        if isinstance(parsed, (list, tuple)):
            return ", ".join(str(g) for g in parsed)
    except (ValueError, SyntaxError):
        pass
    return str(genres_raw).strip()


# Hosts that serve generic "no image" placeholders, not a film's poster.
_PLACEHOLDER_IMAGE_HOSTS = ('via.placeholder.com', 'placehold.co', 'placeholder.com')


def poster_url(poster):
    """The movie's poster URL from the dataset, or '' if it is missing, not an
    absolute http(s) URL, or a generic placeholder image. Never substitutes
    another image. (Whether the URL still loads is checked by the browser.)"""
    if poster is None or (not isinstance(poster, str) and pd.isna(poster)):
        return ''
    url = str(poster).strip()
    match = re.match(r'^https?://([^/?#]+)', url, re.IGNORECASE)
    if not match:
        return ''
    host = match.group(1).lower()
    if any(host == h or host.endswith('.' + h) for h in _PLACEHOLDER_IMAGE_HOSTS):
        return ''
    return url


def format_duration(ms):
    """MM:SS for a real duration in milliseconds, or None if it's missing."""
    if not ms:
        return None
    seconds = int(ms / 1000)
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"

def get_mood_color(mood):
    """Get color for mood"""
    colors = {
        'happy': '#FFD700',
        'sad': '#5C6BC0',
        'energetic': '#FF6F00',
        'chill': '#A5D6A7',
        'stressed': '#80CBC4',
        'angry': '#FF5252'
    }
    return colors.get(mood, '#1db954')


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None or not value.strip():  # unset or blank (e.g. "NAME=" in .env)
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


if __name__ == '__main__':
    # src.config has already loaded .env, so these can be set there too.
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    debug = _env_flag('FLASK_DEBUG')
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', '5000'))

    # Load the datasets at startup instead of on the first request. A dataset that
    # isn't present only disables its feature (those requests get a 503 that says
    # why); one that is present but unreadable still stops the server.
    for name, loader in (("music catalog", load_music_catalog), ("movie dataset", load_movie_data)):
        try:
            data = loader()
        except DatasetUnavailable as exc:
            logger.warning("%s", exc)
        except Exception:
            logger.exception("Could not load the %s from data/", name)
            raise SystemExit(1)
        else:
            logger.info("Loaded the %s: %d rows", name, len(data))
    metadata = load_music_metadata()
    if metadata is None:
        logger.info("No local music metadata (python build_music_metadata.py); song search is empty "
                    "and 278k tracks show placeholder titles")
    else:
        logger.info("Local music metadata loaded for %d tracks", len(metadata))

    if debug and host not in ('127.0.0.1', 'localhost'):
        logger.warning("FLASK_DEBUG is on while listening on %s -- the debugger allows "
                       "arbitrary code execution; never expose it on a network.", host)
    logger.info("Starting MoodCast on http://%s:%d (debug=%s)", host, port, debug)
    app.run(debug=debug, host=host, port=port, load_dotenv=False)
