"""
Popular Artists: a separate music mode built on its own local dataset.

Lists songs by a curated set of popular international, K-pop and Indian
artists (config/popular_artists.json) that suit the selected mood. It does
not use the 278k catalog, src/recommend.py's ranking or the artist-priority
layer; it only reuses the app's mood -> music-target mapping (read-only) so
"Match my mood" and "Cheer me up" mean the same thing in both modes.

Data: data/external/popular_artists_2024/popular_tracks.csv, built locally
by prepare_popular_artists.py from the 2024 multilingual Spotify tracks
dataset (see DATA.md). It is local-only; when it is missing, load_prepared()
returns None and the mode reports that it is unavailable.

Mood eligibility is model-derived, not human-validated: happy, sad,
energetic and calm (chill) need the existing 278k classifier's probability
>= CONFIDENCE_MIN for that target; angry and stressed use the existing
derived-mood rules with the cutoffs recorded for the 278k catalog, applied
by the preparation script.

A list has at most MAX_RESULTS tracks and MAX_PER_ARTIST per curated artist.
Picks rotate across the artist categories (internal only -- never sent to
the browser) and favour popular tracks with a strong mood signal. Draws use
NumPy's global random state, so a fixed seed gives a fixed list.
"""
import json
import logging
import os
import re
import threading

import numpy as np
import pandas as pd

from . import recommend
from .config import BASE_DIR

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(BASE_DIR, "config", "popular_artists.json")
DATA_DIR = os.path.join(BASE_DIR, "data", "external", "popular_artists_2024")
PREPARED_PATH = os.path.join(DATA_DIR, "popular_tracks.csv")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

CONFIDENCE_MIN = 0.6     # classifier probability a track needs for happy/sad/energetic/calm
MAX_RESULTS = 30
MAX_PER_ARTIST = 2
POOL = 40                # most popular eligible tracks per category a pick is drawn from

BASE_TARGETS = ("happy", "sad", "energetic", "calm")
DERIVED_TARGETS = ("angry", "stressed")
SPOTIFY_TRACK_URL = "https://open.spotify.com/track/"
_TRACK_ID_RE = re.compile(r"^[0-9A-Za-z]{22}$")

PREPARED_COLUMNS = [
    "track_id", "track_name", "artist_name", "matched_artists", "category", "year", "popularity",
    "duration_ms", "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
    "proba_sad", "proba_happy", "proba_energetic", "proba_calm",
    "derived_angry", "derived_angry_score", "derived_stressed", "derived_stressed_score", "track_url",
]


# ---------------------------------------------------------------- curated artists + matching

def load_config(path=None):
    """{"categories": {category: [artist names]}} from config/popular_artists.json."""
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cats = {k: [str(a) for a in v] for k, v in (cfg.get("categories") or {}).items() if v}
    if not cats:
        raise ValueError("config/popular_artists.json lists no artists")
    return {"categories": cats}


def artist_categories(config):
    """{artist name: category}, first category wins for a name listed twice."""
    out = {}
    for cat, names in config["categories"].items():
        for name in names:
            out.setdefault(name, cat)
    return out


def _parts(text):
    return [p.strip() for p in str(text).split(",")]


def match_artists(artist_string, names):
    """
    The curated names that appear in a source artist string as complete
    comma-separated entries, in the order they appear. A name that itself
    contains commas ("Tyler, The Creator") must appear as the same run of
    entries, so the string is never split into guessed artists, and a name
    only matches a whole entry ("Pritam" never matches "Pritam Chakraborty").
    """
    parts = _parts(artist_string)
    found = []
    for i in range(len(parts)):
        for name in names:
            sub = _parts(name)
            if parts[i:i + len(sub)] == sub and name not in found:
                found.append(name)
    return found


# ---------------------------------------------------------------- prepared data

_cache = {"key": None, "df": None}
_cache_lock = threading.Lock()


def load_prepared(path=None):
    """The prepared dataset (cached per file version), or None if it is missing or invalid."""
    path = path or PREPARED_PATH
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (path, stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        if _cache["key"] == key:
            return _cache["df"]
        try:
            df = pd.read_csv(path, dtype={"track_id": str, "track_name": str, "artist_name": str},
                             keep_default_na=False)
            missing = [c for c in PREPARED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"missing columns {missing}")
            df["matched_artists"] = df["matched_artists"].map(json.loads)
            for col in ("derived_angry", "derived_stressed"):
                df[col] = df[col].map(lambda v: str(v).strip().lower() == "true")
            valid = (df["track_id"].str.fullmatch(_TRACK_ID_RE.pattern)
                     & (df["track_url"] == SPOTIFY_TRACK_URL + df["track_id"]))
            if not valid.all() or df["track_id"].duplicated().any():
                raise ValueError("invalid or duplicate track ids")
            df = df.reset_index(drop=True)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Popular Artists data unusable (%s); the mode is unavailable", exc)
            df = None
        _cache["key"], _cache["df"] = key, df
        return df


# ---------------------------------------------------------------- recommendations

def target_candidates(df, target):
    """(eligible mask, ranking signal) for one music target on the prepared dataset."""
    if target in BASE_TARGETS:
        signal = df[f"proba_{target}"].to_numpy(dtype=float)
        member = signal >= CONFIDENCE_MIN
        if target == "calm":   # the calm speech-like guard, as on the main catalog
            member &= ~(df["speechiness"].to_numpy(dtype=float) > recommend.CALM_CELL_SPEECHINESS_MAX)
        return member, signal
    if target in DERIVED_TARGETS:
        return df[f"derived_{target}"].to_numpy(dtype=bool), df[f"derived_{target}_score"].to_numpy(dtype=float)
    raise ValueError(f"Unknown music target {target!r}")


def recommend_popular(df, mood, mode, limit=MAX_RESULTS, exclude=(), max_per_artist=MAX_PER_ARTIST):
    """
    (rows, info): up to `limit` (<= MAX_RESULTS) tracks for this mood and mode
    ("Match my mood" / "Cheer me up"), never one in `exclude`, at most
    `max_per_artist` per curated artist. Each music target gets its
    weight-proportional share of the list (missing slots go to the others);
    within a target, picks rotate to the category with the fewest picks that
    still has an eligible track, and each pick is drawn from that category's
    POOL most popular eligible tracks, weighted by popularity x mood signal.
    `rows` keeps the prepared columns plus `target` and `score`; `info`
    reports how many tracks were eligible and, for a short list, whether the
    eligible pool or the per-artist limit ran out, so the page can say which.
    """
    limit = max(0, min(int(limit), MAX_RESULTS))
    targets = recommend._target_music_weighted(mood, mode)
    if not targets:
        raise ValueError(f"No music targets for mood {mood!r}")
    excluded = df["track_id"].isin(set(exclude)).to_numpy()
    popularity = df["popularity"].to_numpy(dtype=float)
    categories = df["category"].to_numpy()
    cat_order = list(dict.fromkeys(categories))
    artist_count = {}
    taken = np.zeros(len(df), dtype=bool)
    picks = {t: [] for t, _ in targets}
    signals, members = {}, {}
    for target, _ in targets:
        members[target], signals[target] = target_candidates(df, target)
    eligible = np.logical_or.reduce([members[t] for t, _ in targets]) & ~excluded

    def fits(row):
        return all(artist_count.get(a, 0) < max_per_artist for a in df.at[row, "matched_artists"])

    def draw(target, n):
        member, signal = members[target], signals[target]
        tiebreak = {c: i for i, c in enumerate(np.random.permutation(cat_order))}
        for _ in range(n):
            free = member & ~taken & ~excluded
            by_cat = {}
            for cat in cat_order:
                rows = np.flatnonzero(free & (categories == cat))
                rows = [r for r in rows[np.lexsort((rows, -signal[rows], -popularity[rows]))] if fits(r)][:POOL]
                if rows:
                    by_cat[cat] = rows
            if not by_cat:
                return
            counts = {c: sum(categories[r] == c for t in picks for r in picks[t]) for c in by_cat}
            cat = min(by_cat, key=lambda c: (counts[c], tiebreak[c]))
            pool = np.array(by_cat[cat])
            weights = (popularity[pool] + 1.0) * np.maximum(signal[pool], 1e-6)
            row = int(np.random.choice(pool, p=weights / weights.sum()))
            taken[row] = True
            picks[target].append(row)
            for a in df.at[row, "matched_artists"]:
                artist_count[a] = artist_count.get(a, 0) + 1

    shares = recommend._allocate_shares([w for _, w in targets], limit) if limit else [0] * len(targets)
    for (target, _), share in zip(targets, shares):
        draw(target, share)
    for target, _ in targets:                       # top up from the other targets, in order
        missing = limit - int(taken.sum())
        if missing <= 0:
            break
        draw(target, missing)

    slots = []   # interleave targets through the list; each target's picks most popular first
    for t_i, (target, _) in enumerate(targets):
        rows = sorted(picks[target], key=lambda r: (-popularity[r], -signals[target][r], df.at[r, "track_id"]))
        for k, row in enumerate(rows):
            slots.append(((k + 0.5) / len(rows), t_i, row, target))
    slots.sort(key=lambda s: s[:2])
    order = [s[2] for s in slots]
    rows = df.iloc[order].assign(target=[s[3] for s in slots],
                                 score=[float(signals[s[3]][s[2]]) for s in slots])
    limited = len(rows) < limit
    info = {
        "targets": [t for t, _ in targets],
        "eligible": int(eligible.sum()),
        "returned": len(rows),
        "limited": limited,
        # why a list is short: too few eligible tracks left ("pool"), or eligible tracks remain
        # but every one would repeat an artist more than max_per_artist times ("artist_cap")
        "limited_by": None if not limited else ("artist_cap" if int(eligible.sum()) > len(rows) else "pool"),
    }
    return rows, info
