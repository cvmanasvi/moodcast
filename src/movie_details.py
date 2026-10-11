"""
Verified TMDB details for a title from the MoodCast dataset: its TMDB poster
and where it can be watched in India.

The dataset (IMDb-sourced) mixes films with TV series such as Friends (1994),
and has no TMDB id. A title is resolved in this order:
  1. by its IMDb id via TMDB /find, which says whether it is a movie or a TV
     series -- accepted only when exactly one movie-or-TV result comes back;
  2. otherwise by searching TMDB movies AND TV series for its title and
     release (first-air) year. A match is accepted only when exactly one
     result across both has the same normalized title (or original title) and
     the same year (or, failing that, one off).
Anything else returns nothing for it -- no poster and no providers are guessed.

Watch-provider data comes from TMDB's /movie/{id} or /tv/{id} with
append_to_response=watch/providers (JustWatch data, which TMDB requires to be
attributed). It lists provider names and logos per region plus one TMDB
"watch" page link per region; TMDB gives no per-provider deep links, so none
are made up. `providers.verified` is True only when TMDB actually returned
watch-provider data, so "no listed options" is never inferred from a missing
or failed response.

Runs server-side only (the API key never reaches the browser), with an
in-memory cache. Nothing is written to disk. Ranking is unaffected.
"""
import logging
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import requests

from .config import TMDB_API_KEY, TMDB_BASE_URL

logger = logging.getLogger(__name__)

REGION = "IN"
TIMEOUT = 6                       # seconds per TMDB request
TTL_OK = 6 * 3600                 # cache lifetime for answered lookups
TTL_ERROR = 60                    # retry network/server failures after a minute
MAX_WORKERS = 8
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
LOGO_BASE = "https://image.tmdb.org/t/p/w92"
ATTRIBUTION = "Streaming data from JustWatch via TMDB"
PROVIDER_GROUPS = (("flatrate", "Stream"), ("free", "Free"), ("ads", "Free with ads"), ("rent", "Rent"), ("buy", "Buy"))

_IMAGE_PATH_RE = re.compile(r"^/[A-Za-z0-9_-]+\.(jpg|jpeg|png|svg)$")
_WATCH_LINK_RE = re.compile(r"^https://www\.themoviedb\.org/(movie|tv)/[^\s\"'<>]+$")

_cache = {}
_lock = threading.Lock()


def enabled():
    return bool(TMDB_API_KEY)


def normalize_title(title):
    """Case-, accent-, punctuation- and whitespace-insensitive form used only for matching."""
    text = unicodedata.normalize("NFKD", str(title or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower().replace("&", " and "))
    return re.sub(r"\s+", " ", text).strip()


def _get(session, path, params):
    response = session.get(f"{TMDB_BASE_URL}{path}", params={"api_key": TMDB_API_KEY, **params}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def _as_movie_result(tv):
    """A /search/tv result in /search/movie's field names, so one matcher serves both."""
    if not isinstance(tv, dict):
        return tv
    return {**tv, "title": tv.get("name"), "original_title": tv.get("original_name"),
            "release_date": tv.get("first_air_date")}


def match_movie(results, title, year):
    """
    The single TMDB search result with the same normalized title (or original
    title) and release year, as (result, None); or (None, "unmatched"/"ambiguous").
    Only if no result has the exact year is a one-year difference accepted
    (festival vs. theatrical release dates), and it must still be unique.
    """
    want = normalize_title(title)
    same_title = [r for r in results or []
                  if isinstance(r, dict) and want
                  and want in {normalize_title(r.get("title")), normalize_title(r.get("original_title"))}
                  and str(r.get("release_date") or "")[:4].isdigit()]
    for tolerance in (0, 1):
        hits = [r for r in same_title if abs(int(str(r["release_date"])[:4]) - int(year)) <= tolerance]
        if hits:
            return (hits[0], None) if len({r.get("id") for r in hits}) == 1 else (None, "ambiguous")
    return None, "unmatched"


def parse_providers(payload, region=REGION):
    """
    Provider groups for one region from a watch/providers payload, keeping only
    well-formed entries. `verified` is True only for a real payload (one with a
    "results" object): then an absent region means TMDB lists no options there.
    A missing payload (None, or no "results") stays unverified.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    verified = isinstance(results, dict)
    region_data = results.get(region) if verified else None
    if not isinstance(region_data, dict):
        return {"region": region, "verified": verified, "link": None, "groups": []}
    link = region_data.get("link")
    groups = []
    for key, label in PROVIDER_GROUPS:
        providers, seen = [], set()
        for p in region_data.get(key) or []:
            name = p.get("provider_name") if isinstance(p, dict) else None
            if not isinstance(name, str) or not name.strip() or name in seen:
                continue
            seen.add(name)
            logo = p.get("logo_path")
            providers.append({"name": name.strip(),
                              "logo": f"{LOGO_BASE}{logo}" if isinstance(logo, str) and _IMAGE_PATH_RE.match(logo) else None})
        if providers:
            groups.append({"type": key, "label": label, "providers": providers})
    return {"region": region,
            "verified": True,
            "link": link if isinstance(link, str) and _WATCH_LINK_RE.match(link) else None,
            "groups": groups}


def _empty(status):
    return {"status": status, "media_type": None, "tmdb_id": None, "poster": None, "providers": parse_providers(None)}


def _find_by_imdb(session, imdb_id):
    """(media_type, tmdb_id) for an IMDb id, or (None, reason) -- "unmatched" if TMDB doesn't know it."""
    found = _get(session, f"/find/tt{int(imdb_id):07d}", {"external_source": "imdb_id"})
    hits = [("movie", r) for r in found.get("movie_results") or [] if isinstance(r, dict)]
    hits += [("tv", r) for r in found.get("tv_results") or [] if isinstance(r, dict)]
    if len({(m, r.get("id")) for m, r in hits}) == 1:
        return hits[0][0], int(hits[0][1]["id"])
    return None, "ambiguous" if hits else "unmatched"


def _search_by_title(session, title, year):
    """(media_type, tmdb_id) for the one movie or TV series with this title and year, or (None, reason)."""
    movies = _get(session, "/search/movie", {"query": title, "year": year, "include_adult": "false"})
    shows = _get(session, "/search/tv", {"query": title, "first_air_date_year": year, "include_adult": "false"})
    movie, movie_reason = match_movie(movies.get("results"), title, year)
    show, show_reason = match_movie([_as_movie_result(r) for r in shows.get("results") or []], title, year)
    if movie is not None and show is not None:
        return None, "ambiguous"
    if movie is not None:
        return "movie", int(movie["id"])
    if show is not None:
        return "tv", int(show["id"])
    return None, "ambiguous" if "ambiguous" in (movie_reason, show_reason) else "unmatched"


def _lookup_uncached(title, year, session, imdb_id=None):
    try:
        media_type, found = _find_by_imdb(session, imdb_id) if imdb_id else (None, "unmatched")
        if media_type is None and found == "unmatched" and title and year is not None:
            media_type, found = _search_by_title(session, title, year)
        if media_type is None:
            return _empty(found)
        details = _get(session, f"/{media_type}/{found}", {"append_to_response": "watch/providers"})
        poster_path = details.get("poster_path")
        return {
            "status": "matched",
            "media_type": media_type,
            "tmdb_id": found,
            "poster": f"{POSTER_BASE}{poster_path}" if isinstance(poster_path, str) and _IMAGE_PATH_RE.match(poster_path) else None,
            "providers": parse_providers(details.get("watch/providers")),
        }
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning("TMDB lookup failed for a title (%s)", type(exc).__name__)
        return _empty("error")


def lookup(title, year, session=None, imdb_id=None):
    """Cached TMDB details for one dataset title (by IMDb id when known, else title + year)."""
    if not enabled():
        return _empty("disabled")
    if not imdb_id and (not title or year is None):
        return _empty("unmatched")
    key = ("imdb", int(imdb_id)) if imdb_id else (normalize_title(title), int(year))
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    result = _lookup_uncached(title, None if year is None else int(year), session or requests.Session(), imdb_id)
    ttl = TTL_ERROR if result["status"] == "error" else TTL_OK
    with _lock:
        _cache[key] = (now + ttl, result)
    return result


def lookup_many(items):
    """{key: lookup(...)} for an iterable of (key, title, year[, imdb_id]), in parallel."""
    items = list(items)
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(lambda it: lookup(it[1], it[2], session, it[3] if len(it) > 3 else None), items))
    return {it[0]: r for it, r in zip(items, results)}


def clear_cache():
    with _lock:
        _cache.clear()
