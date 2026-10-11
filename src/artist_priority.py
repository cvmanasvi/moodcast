"""
Artist prioritization: a small, optional layer on top of recommend_music.

recommend_music (Phase 2K) is called unchanged and decides the list: which
targets, how many slots each, and every track's relevance. This layer may then
swap some of those tracks for tracks by preferred artists (config/artist_priority.json:
international pop, K-pop, Indian), under rules that keep mood relevance intact:

- Same target only: a slot served for target T (e.g. "calm") can only go to a
  track that is itself an eligible T track (same membership rule; calm also
  keeps the Phase 2K.1 speech-like guard), never a track from another mood.
- Never less relevant: the replacement's ranking signal (the classifier
  probability / derived score the recommender ranked by) must be at least the
  signal of the track it replaces.
- Bounded: at most `max_share` of the list is swapped, weakest slots first,
  and no artist appears more than `max_per_artist` times in the final list.
- Balanced: picks rotate across categories that have eligible tracks.

Artist identity comes only from the local metadata (exact track_id match,
exact artist name). Tracks without metadata are never priority tracks. When
the layer is disabled, the metadata is missing, or nothing qualifies, the
recommender's list is returned unchanged.

On/off: config "enabled" (off by default), overridden by MOODCAST_ARTIST_PRIORITY
when set -- 1/true/yes/on switches it on, 0/false/no/off switches it off.
"""
import json
import logging
import os
import threading
import weakref

import numpy as np
import pandas as pd

from . import recommend
from .config import BASE_DIR

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(BASE_DIR, "config", "artist_priority.json")
_POOL = 20   # replacement drawn (signal-weighted) from the best _POOL eligible tracks of a category, like _sample_best


def load_config(path=None):
    """The prioritization config, or None when disabled/unreadable (the layer then does nothing)."""
    env = os.getenv("MOODCAST_ARTIST_PRIORITY", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return None
    try:
        with open(path or CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Artist priority config unavailable (%s); serving recommendations unchanged", type(exc).__name__)
        return None
    if not (cfg.get("enabled", False) or env in {"1", "true", "yes", "on"}):   # env can switch it on explicitly
        return None
    cats = {k: v for k, v in (cfg.get("categories") or {}).items() if isinstance(v, dict) and v.get("artists")}
    return {
        "max_share": min(max(float(cfg.get("max_share", 0.3)), 0.0), 1.0),
        "max_per_artist": max(int(cfg.get("max_per_artist", 2)), 1),
        "categories": {k: {"label": v.get("label", k), "artists": [str(a) for a in v["artists"]]} for k, v in cats.items()},
    }


class ArtistIndex:
    """
    Per catalog row: the priority category and the matched artist name
    (empty string when the track has no metadata or no configured artist).
    A track with several configured artists takes the first one in its own
    artist order; categories are tried in config order for that artist.

    Everything prioritize() needs per request is precomputed here once per
    catalog: integer category codes, every priority track's artist ids, and
    per target (lazily) its eligible priority tracks sorted best-first.
    """

    def __init__(self, catalog: pd.DataFrame, metadata, config):
        self.category = np.full(len(catalog), "", dtype=object)
        self.artist = np.full(len(catalog), "", dtype=object)
        self.labels = {k: v["label"] for k, v in config["categories"].items()}
        self.keys = list(self.labels)                       # config order; code = position + 1, 0 = none
        self.code = np.zeros(len(catalog), dtype=np.int16)
        self.metadata = metadata
        self._catalog = weakref.ref(catalog)
        self._targets = {}
        self._row_artists = {}                              # priority row -> tuple of artist ids
        self.artist_id = {}                                 # artist name -> int id
        lookup = {}
        for key, cat in config["categories"].items():
            for name in cat["artists"]:
                lookup.setdefault(name, key)
        if metadata is None:
            self.track_ids = np.array([], dtype=object)
            return
        self.track_ids = catalog["track_id"].astype(str).to_numpy()
        row_of = pd.Series(np.arange(len(catalog)), index=self.track_ids)
        for tid in metadata.track_ids():
            names = metadata.artists_of(tid)
            for name in names:
                if name in lookup:
                    pos = row_of.get(tid)
                    if pos is not None:
                        pos = int(pos)
                        self.category[pos] = lookup[name]
                        self.artist[pos] = name
                        self.code[pos] = self.keys.index(lookup[name]) + 1
                        self._row_artists[pos] = tuple(self._id(n) for n in names)
                    break
        self.available = {name: int((self.artist == name).sum()) for name in lookup}

    def _id(self, name):
        return self.artist_id.setdefault(name, len(self.artist_id))

    def artist_ids_at(self, pos):
        """Artist ids of any catalog row (for the per-artist cap); () without metadata."""
        pos = int(pos)
        if pos in self._row_artists:
            return self._row_artists[pos]
        if self.metadata is None:
            return ()
        return tuple(self._id(n) for n in self.metadata.artists_of(self.track_ids[pos]))

    def target(self, catalog, target):
        """
        (member, signal, rows, row_signal, artist_matrix) for a target: its
        eligibility as the recommender defines it (calm with the speech-like
        guard), and its eligible priority rows sorted by signal desc, then row
        -- the order the candidate pool is drawn in -- with their artist ids
        padded into a matrix (-1 = none). Cached per target for this catalog.
        """
        cached = self._targets.get(target)
        if cached is not None and self._catalog() is catalog:
            return cached
        member, signal = recommend._cell_candidates(catalog, target)
        rows = np.flatnonzero(member & (self.code > 0))
        rows = rows[np.lexsort((rows, -signal[rows]))]
        width = max((len(self._row_artists[r]) for r in rows), default=1)
        mat = np.full((len(rows), width), -1, dtype=np.int32)
        for i, r in enumerate(rows):
            ids = self._row_artists[r]
            mat[i, :len(ids)] = ids
        out = (member, signal, rows, signal[rows], mat)
        if self._catalog() is catalog:
            self._targets[target] = out
        return out

    def artists_at(self, catalog, pos):
        """All of a row's artist names; [] without metadata."""
        return self.metadata.artists_of(str(catalog["track_id"].iat[pos])) if self.metadata is not None else []


_index_cache = {"key": None, "index": None}
_index_lock = threading.Lock()


def get_index(catalog, metadata, config):
    """ArtistIndex for this catalog + metadata + config, built once and reused."""
    key = (id(catalog), id(metadata), json.dumps(config, sort_keys=True))
    with _index_lock:
        cached = _index_cache["key"]
        if cached and cached[0]() is catalog and cached[1:] == key[1:]:
            return _index_cache["index"]
        index = ArtistIndex(catalog, metadata, config)
        _index_cache["key"] = (weakref.ref(catalog),) + key[1:]
        _index_cache["index"] = index
        return index


def prioritize(catalog: pd.DataFrame, recs: pd.DataFrame, index: ArtistIndex, config) -> pd.DataFrame:
    """
    `recs` (recommend_music output: catalog rows plus `target` and `score`)
    with up to max_share of its slots swapped for preferred-artist tracks under
    the rules in the module docstring. Adds `artist_category` (label or None)
    and `priority_artist` (the configured artist that matched, or None), and
    keeps every other column, the list length and each slot's target. When
    tracks were swapped, each target's tracks are re-sorted best-first within
    that target's slots, so a stronger replacement moves up instead of staying
    in the weak slot it filled; a list with no swaps keeps its order.
    """
    out = recs.copy()
    out["artist_category"] = None
    out["priority_artist"] = None
    if config is None or index is None or index.metadata is None or len(recs) == 0 or "target" not in recs.columns:
        return out

    positions = catalog.index.get_indexer(recs.index)
    if (positions < 0).any():
        return out
    positions = positions.copy()
    targets = recs["target"].to_numpy()
    budget = int(np.floor(config["max_share"] * len(recs)))
    cap = config["max_per_artist"]

    elig = {t: index.target(catalog, t) for t in set(targets)}
    artist_count = {}
    for pos in positions:
        for a in index.artist_ids_at(pos):
            artist_count[a] = artist_count.get(a, 0) + 1
    picks_per_cat = {k: int((index.code[positions] == c + 1).sum()) for c, k in enumerate(index.keys)}

    # weakest non-priority slots first; a slot is only swapped for an equally or more relevant track
    order = sorted(range(len(positions)), key=lambda i: elig[targets[i]][1][positions[i]])
    swapped = 0
    for i in order:
        if swapped >= budget:
            break
        pos, target = positions[i], targets[i]
        if index.code[pos]:
            continue                                           # already a preferred-artist track
        _, signal, rows, row_signal, mat = elig[target]
        # candidates: eligible priority rows at least as relevant (rows are sorted by signal desc)
        k = int(np.searchsorted(-row_signal, -signal[pos], side="right"))
        if k == 0:
            continue
        cand, cmat = rows[:k], mat[:k]
        ok = ~np.isin(cand, positions)
        blocked = [a for a, n in artist_count.items() if n >= cap]
        if blocked:
            ok &= ~np.isin(cmat, blocked).any(axis=1)
        cand = cand[ok]
        if len(cand) == 0:
            continue
        # rotate categories: the one with the fewest picks so far that has an eligible track
        codes = index.code[cand]
        present = [index.keys[c - 1] for c in np.unique(codes)]
        key = sorted(present, key=lambda name: (picks_per_cat.get(name, 0), index.keys.index(name)))[0]
        pool = cand[codes == index.keys.index(key) + 1][:_POOL]
        weights = signal[pool] - signal[pool].min() + 1e-6
        choice = int(np.random.choice(pool, p=weights / weights.sum()))

        for a in index.artist_ids_at(pos):
            artist_count[a] -= 1
        positions[i] = choice
        for a in index.artist_ids_at(choice):
            artist_count[a] = artist_count.get(a, 0) + 1
        picks_per_cat[key] = picks_per_cat.get(key, 0) + 1
        swapped += 1

    if not swapped:
        out["artist_category"] = [index.labels.get(index.category[p]) or None for p in positions]
        out["priority_artist"] = [index.artist[p] or None for p in positions]
        return out
    result = catalog.iloc[positions].assign(
        score=[elig[t][1][p] for t, p in zip(targets, positions)],
        target=targets,
        artist_category=[index.labels.get(index.category[p]) or None for p in positions],
        priority_artist=[index.artist[p] or None for p in positions],
    )
    extra = [c for c in recs.columns if c not in result.columns]
    for c in extra:            # any other column the recommender added, kept for untouched slots only
        result[c] = recs[c].to_numpy()
    # a swap fills the replaced (weakest) slot; restore best-first order within each target,
    # keeping the slots each target occupies (the recommender's interleaving)
    return result.iloc[_order_within_targets(targets, result["score"].to_numpy())]


def _order_within_targets(targets, scores):
    """Row order that sorts each target's slots by score, highest first (ties keep slot
    order), while every slot keeps its target."""
    order = np.arange(len(targets))
    for target in dict.fromkeys(targets):
        slots = np.flatnonzero(targets == target)
        order[slots] = slots[np.argsort(-scores[slots], kind="stable")]
    return order
