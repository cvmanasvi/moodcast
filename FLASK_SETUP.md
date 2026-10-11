# Flask app: configuration and API

See [README.md](README.md) for installation and [QUICKSTART.md](QUICKSTART.md)
for a short start.

## Pages

| Route | Page |
|---|---|
| `/` | Home: mood and mode selection, text mood detection |
| `/music` | Music recommendations (needs the locally built music catalog) |
| `/movies` | Movie recommendations (needs `data/movies_final.csv`, not distributed) |
| `/dashboard` | Combined music and movie view |

## Configuration

Optional settings go in `.env` (copy `.env.example`; `.env` is gitignored) or
in the environment. Environment variables take precedence.

| Variable | Default | Purpose |
|---|---|---|
| `FLASK_HOST` | `127.0.0.1` | Listen address |
| `FLASK_PORT` | `5000` | Port |
| `FLASK_DEBUG` | `0` | Werkzeug debugger. For local development only; never expose it on a network |
| `LOG_LEVEL` | `INFO` | Log level |
| `TMDB_API_KEY` | unset | Movie poster fallback and India watch providers |
| `MOODCAST_ARTIST_PRIORITY` | off | `1` enables the preferred-artist layer |
| `MOODCAST_SPOTIFY_OEMBED` | on | `0` disables Spotify oEmbed titles and covers |

## JSON API

Moods are `happy`, `sad`, `angry`, `chill`, `energetic` and `stressed`. Modes
are `match` and `uplift`.

| Endpoint | Purpose |
|---|---|
| `POST /api/detect-mood` | `{"text": "..."}` returns `{mood, confidence, source, uncertain}` |
| `POST /api/recommend` | `{"mood", "mode", "type": "music" or "movies", "limit", "exclude"}` returns `{mood, mode, results, refresh?}` |
| `POST /api/popular` | Popular Artists mode; returns `available: false` with a message when its dataset isn't prepared |
| `GET /api/track-details?ids=...` | Titles and covers for Spotify track IDs (oEmbed) |
| `POST /api/movie-details` | `{"movie_ids": [...]}` returns TMDB poster and India providers |
| `GET /api/search?q=...` | Movies, songs, artists and genres (empty groups when their data is absent) |

Invalid input returns a JSON `4xx` with an `error` message. When the dataset a
request needs is not present, the API returns `503` with
`{"error": "<explanation>", "dataset_unavailable": true}`. Unexpected errors
return a generic `500` and are logged server-side.

Flask is the only maintained UI; the original Streamlit app was retired in
Phase 2H.4.
