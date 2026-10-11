"""
Track title and cover art from Spotify's public oEmbed endpoint
(https://open.spotify.com/oembed?url=https://open.spotify.com/track/<id>).

oEmbed needs no account, token or Premium subscription, and answers for the
exact track ID asked about. It returns the track's title and a cover image,
but no artist. Used only to display recommendations the recommender has
already chosen; nothing here affects ranking, mood labels or training.

Server-side and on demand, with an in-memory cache; nothing is written to
disk. Responses are validated: a title must be a non-empty string and the
cover must be a Spotify image-CDN URL, otherwise the field is left empty.
Disable with MOODCAST_SPOTIFY_OEMBED=0.
"""
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger(__name__)

OEMBED_URL = "https://open.spotify.com/oembed"
TIMEOUT = 5
TTL_OK = 24 * 3600
TTL_ERROR = 120
MAX_WORKERS = 8
MAX_TITLE_LENGTH = 300

_TRACK_ID_RE = re.compile(r"^[0-9A-Za-z]{22}$")
_ART_RE = re.compile(r"^https://(i\.scdn\.co|image-cdn-[a-z0-9-]+\.spotifycdn\.com|mosaic\.scdn\.co)/image/[0-9a-f]+$")

_cache = {}
_lock = threading.Lock()


def enabled():
    return os.getenv("MOODCAST_SPOTIFY_OEMBED", "1").strip().lower() not in {"0", "false", "no", "off"}


def parse(payload):
    """{'title', 'artwork'} from an oEmbed JSON payload; each None unless valid."""
    payload = payload if isinstance(payload, dict) else {}
    title = payload.get("title")
    title = title.strip() if isinstance(title, str) else ""
    art = payload.get("thumbnail_url")
    return {
        "title": title if 0 < len(title) <= MAX_TITLE_LENGTH else None,
        "artwork": art if isinstance(art, str) and _ART_RE.match(art) else None,
    }


def _fetch(track_id, session):
    try:
        response = session.get(OEMBED_URL, params={"url": f"https://open.spotify.com/track/{track_id}"}, timeout=TIMEOUT)
        if response.status_code == 404:
            return {"title": None, "artwork": None, "status": "not_found"}
        response.raise_for_status()
        return {**parse(response.json()), "status": "ok"}
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Spotify oEmbed lookup failed (%s)", type(exc).__name__)
        return {"title": None, "artwork": None, "status": "error"}


def lookup(track_id, session=None):
    """Cached oEmbed fields for one valid track id, or None for an invalid id."""
    if not isinstance(track_id, str) or not _TRACK_ID_RE.match(track_id):
        return None
    if not enabled():
        return {"title": None, "artwork": None, "status": "disabled"}
    now = time.monotonic()
    with _lock:
        hit = _cache.get(track_id)
        if hit and hit[0] > now:
            return hit[1]
    result = _fetch(track_id, session or requests.Session())
    with _lock:
        _cache[track_id] = (now + (TTL_ERROR if result["status"] == "error" else TTL_OK), result)
    return result


def lookup_many(track_ids):
    """{track_id: lookup(track_id)} for the valid, de-duplicated ids, in parallel."""
    ids = list(dict.fromkeys(t for t in track_ids if isinstance(t, str) and _TRACK_ID_RE.match(t)))
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(lambda t: lookup(t, session), ids))
    return dict(zip(ids, results))


def clear_cache():
    with _lock:
        _cache.clear()
