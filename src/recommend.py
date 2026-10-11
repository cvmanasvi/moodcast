"""
Content-based recommenders for music and movies.

Both replace the old hand-weighted linear formulas (e.g.
`0.4*valence + 0.3*energy + ...` with made-up coefficients) with a score
grounded in trained ML models:

- Music, 278k catalog (data/music_catalog_278k.csv, keyed by Spotify track
  ID, built by train_music_278k.py): target-coverage ranking. Each app mood
  maps to one or more targets (MOOD_TO_MUSIC_TARGETS): a real dataset label,
  whose candidates are the tracks carrying that ground-truth label ranked by
  the 4-label classifier's probability, or a derived mood (angry, stressed),
  whose candidates meet a percentile feature rule and are ranked by its
  percentile score. Every target gets a weight-proportional share of the
  list. A request with a single dataset-label target (happy, sad, calm,
  energetic) instead draws its tracks from fixed feature cells spanning that
  label's eligible population (see FEATURE_CELL_TARGETS). That dataset has
  no popularity field, so popularity plays no part.
  This is the only music path: the legacy music_final.csv catalog and its
  K-Means + popularity ranking were retired in Phase 2H.4.
- Movies: the trained TF-IDF + softmax-regression text classifier's predicted
  probability that a movie's plot/genres match the target mood (precomputed
  into data/movies_final.csv as `proba_<mood>` columns by apply_ml_labels.py),
  blended with rating and popularity.

Function names/signatures (`recommend_music`, `recommend_movies`) are
unchanged so app.py / flask_app.py don't need to change how they call this.
"""
import logging
import threading
import weakref

import numpy as np
import pandas as pd
from .config import (DERIVED_MUSIC_MOODS, LABEL_TO_MOOD, MOOD_TO_MUSIC_TARGETS,
                     MOOD_TO_TAGS_CHEER, MOOD_TO_TAGS_MATCH)
from .music_pipeline import AUDIO_FEATURES

logger = logging.getLogger(__name__)


def _minmax(series: pd.Series) -> np.ndarray:
    values = series.astype(float).values
    lo, hi = np.nanmin(values), np.nanmax(values)
    if hi - lo < 1e-9:
        return np.zeros(len(values))
    return (values - lo) / (hi - lo)


def _diverse_top(df: pd.DataFrame, top_n: int, pool_multiplier: int = 3, pool_cap: int = 90) -> pd.DataFrame:
    """
    Pick top_n rows from df (already has a 'score' column), but instead of
    always returning the exact same top-N in the exact same order, weighted-
    sample them from a wider pool of good candidates (top pool_multiplier*N,
    capped at pool_cap). Higher-scoring rows are still much more likely to be
    picked -- this isn't random noise, it's "same quality tier, different
    pick and order each time" -- so results feel fresh across page loads /
    mood re-selections instead of being identical every single time.

    If top_n covers (or exceeds) the whole dataset, there's no pool to
    sample from, so we just shuffle the full sorted set instead.
    """
    ranked = df.sort_values("score", ascending=False)
    if top_n is None or top_n >= len(ranked):
        # nothing to trim -- still shuffle so repeat views aren't identical
        return ranked.sample(frac=1) if len(ranked) > 1 else ranked

    pool_size = min(len(ranked), max(top_n * pool_multiplier, top_n), pool_cap)
    pool = ranked.head(pool_size)
    if len(pool) <= top_n:
        return pool.sample(frac=1) if len(pool) > 1 else pool

    weights = pool["score"].to_numpy(dtype=float)
    weights = weights - weights.min() + 1e-6  # keep all weights positive
    probs = weights / weights.sum()
    chosen = np.random.choice(pool.index.to_numpy(), size=top_n, replace=False, p=probs)
    # re-sort the sampled set by score so the strongest pick still leads
    # (matters for e.g. the movies page "hero" slot, which is just results[0])
    return df.loc[chosen].sort_values("score", ascending=False)


def _target_moods_weighted(mood: str, mode: str):
    """
    Ordered, weighted list of (mood, weight) to rank content against.

    "Match my mood" always tries the mood itself first (weight 1.0) -- this
    matters now that ML-labeled data reliably includes moods like "angry"
    that the old hand-written tag map used to route around entirely --  then
    falls back to MOOD_TO_TAGS_MATCH's secondary moods at a lower weight, for
    datasets (like movies) that have no signal for the mood itself.

    "Cheer me up" uses MOOD_TO_TAGS_CHEER's ordered list directly, with
    earlier moods weighted higher, so two source moods that share target
    moods but in a different order (e.g. sad -> [happy, chill] vs.
    angry -> [chill, happy]) still produce genuinely different rankings
    instead of collapsing to the same top results.
    """
    if mode == "Cheer me up":
        ordered = MOOD_TO_TAGS_CHEER.get(mood, [mood])
    else:
        ordered = [mood] + [m for m in MOOD_TO_TAGS_MATCH.get(mood, []) if m != mood]

    return _rank_weighted(ordered)


def _rank_weighted(ordered):
    weights = [0.6 ** i for i in range(len(ordered))]  # 1.0, 0.6, 0.36, ...
    total = sum(weights)
    return [(m, w / total) for m, w in zip(ordered, weights)]


def _target_music_weighted(mood: str, mode: str):
    """
    Ordered, weighted list of (music target, weight).

    Same target moods as _target_moods_weighted ("Cheer me up" uses
    MOOD_TO_TAGS_CHEER, "Match my mood" the mood itself), each translated to
    music targets (real 278k labels or derived moods) through
    MOOD_TO_MUSIC_TARGETS.
    """
    app_moods = MOOD_TO_TAGS_CHEER.get(mood, [mood]) if mode == "Cheer me up" else [mood]
    ordered = []
    for app_mood in app_moods:
        for target in MOOD_TO_MUSIC_TARGETS.get(app_mood, []):
            if target not in ordered:
                ordered.append(target)
    return _rank_weighted(ordered)


_LABEL_BY_NAME = {name: label for label, name in LABEL_TO_MOOD.items()}


def _target_columns(target: str):
    """(membership column, ranking-signal column) for a music target."""
    if target in DERIVED_MUSIC_MOODS:
        return f"derived_{target}", f"derived_{target}_score"
    return "label", f"proba_{target}"


def _is_278k_catalog(df: pd.DataFrame, targets) -> bool:
    return "track_id" in df.columns and all(
        col in df.columns for target, _ in targets for col in _target_columns(target))


def _target_candidates(df: pd.DataFrame, target: str):
    """(bool mask of eligible tracks, ranking signal) for one target."""
    member_col, signal_col = _target_columns(target)
    if target in DERIVED_MUSIC_MOODS:
        member = df[member_col].to_numpy(dtype=bool)
    else:
        member = df[member_col].to_numpy() == _LABEL_BY_NAME[target]  # true dataset label
    return member, df[signal_col].to_numpy(dtype=float)


def _allocate_shares(weights, total: int):
    """Split `total` slots by weight (largest remainder); always sums to total."""
    raw = np.asarray(weights, dtype=float) * total / sum(weights)
    shares = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - shares), kind="stable")[: total - shares.sum()]:
        shares[i] += 1
    return shares.tolist()


def _sample_best(signal: np.ndarray, available: np.ndarray, n: int, pool_multiplier: int) -> np.ndarray:
    """
    Row positions of n available tracks, weighted-sampled (higher signal =
    likelier, as in _diverse_top) from the best n * pool_multiplier by
    signal; returned best-first. pool_multiplier=1 is deterministic.
    """
    idx = np.flatnonzero(available)
    if n <= 0 or len(idx) == 0:
        return np.array([], dtype=int)
    pool_size = min(len(idx), n * pool_multiplier)
    if pool_size < len(idx):
        idx = idx[np.argpartition(-signal[idx], pool_size - 1)[:pool_size]]
    if len(idx) > n:
        weights = signal[idx] - signal[idx].min() + 1e-6
        idx = np.random.choice(idx, size=n, replace=False, p=weights / weights.sum())
    return idx[np.argsort(-signal[idx], kind="stable")]


def _coverage_top(df: pd.DataFrame, targets, top_n: int, pool_multiplier: int = 20) -> pd.DataFrame:
    """
    Target-coverage ranking for the 278k catalog. Each target gets a
    weight-proportional share of top_n, filled only from its own candidates
    (see _target_candidates) by _sample_best; no track is used twice. If a
    target runs out of candidates, its missing slots go to the other targets
    in order. The per-target lists are then interleaved evenly (the k-th of a
    target's n picks sits at position (k + 0.5) / n), so every requested
    target shows up throughout the list instead of only after the first one
    is exhausted. Adds `score` (the target's ranking signal) and `target`.
    """
    shares = _allocate_shares([w for _, w in targets], top_n)
    taken = np.zeros(len(df), dtype=bool)
    picks = []
    for (target, _), share in zip(targets, shares):
        member, signal = _target_candidates(df, target)
        chosen = _sample_best(signal, member & ~taken, share, pool_multiplier)
        taken[chosen] = True
        picks.append([target, chosen, member, signal])

    missing = top_n - int(taken.sum())
    for pick in picks:
        if missing <= 0:
            break
        target, chosen, member, signal = pick
        extra = _sample_best(signal, member & ~taken, missing, pool_multiplier)
        taken[extra] = True
        missing -= len(extra)
        combined = np.concatenate([chosen, extra])
        pick[1] = combined[np.argsort(-signal[combined], kind="stable")]

    slots = []  # (interleave key, target index, row position, target, score)
    for t_i, (target, chosen, _, signal) in enumerate(picks):
        for k, row in enumerate(chosen):
            slots.append(((k + 0.5) / len(chosen), t_i, row, target, signal[row]))
    slots.sort(key=lambda slot: slot[:2])
    return df.iloc[[slot[2] for slot in slots]].assign(
        score=[slot[4] for slot in slots], target=[slot[3] for slot in slots])


# ---- feature cells for base-label single-target requests ----
# A request whose only target is a dataset label (happy/match, sad/match,
# chill/match and stressed/uplift -> calm, energetic/match) is served from 30
# fixed feature cells spanning that label's whole eligible population instead
# of _coverage_top's top-N pool: ranking by confidence alone picks the label's
# feature-space extreme, so those lists were narrow. Validated offline in
# Phases 2J/2J.1 (evaluate_music_balanced_quality.py,
# evaluate_music_cell_confirmation.py). Every other request (angry, stressed,
# all multi-target "Cheer me up" mixes) keeps _coverage_top unchanged.
# Calm cell candidates exclude speech-like tracks (Phase 2K.1): a few dozen of
# them tie at calm probability 1.0 and topped their cells, so about 1 in 5
# calm-only tracks served was speech-like. Only the calm cell route is guarded.
FEATURE_CELL_TARGETS = ("happy", "sad", "calm", "energetic")
CALM_CELL_SPEECHINESS_MAX = 0.66        # strictly greater is excluded, as in the stressed rule's spoken-word guard
_CELL_PC1_BINS, _CELL_PC2_BINS = 6, 5   # 30 cells
_CELL_TOP = 20                          # confident candidates per cell, like _coverage_top's 20 per slot
_cell_cache = {"catalog": None, "cells": None}   # one catalog per process: (weakref, cells or None if the build failed)
_cell_lock = threading.Lock()


def _uses_feature_cells(targets) -> bool:
    return len(targets) == 1 and targets[0][0] in FEATURE_CELL_TARGETS


def _cell_candidates(df: pd.DataFrame, target: str):
    """_target_candidates for the cell route: calm drops speechiness > CALM_CELL_SPEECHINESS_MAX."""
    member, signal = _target_candidates(df, target)
    if target == "calm":
        member = member & ~(df["speechiness"].to_numpy(dtype=float) > CALM_CELL_SPEECHINESS_MAX)
    return member, signal


def _build_feature_cells(df: pd.DataFrame) -> dict:
    """
    {target: 30 arrays of row positions, each best signal first}. A target's
    eligible tracks are split into equal-count cells: 6 quantile bins on the
    first principal component of their audio features (z-scored over the
    catalog), then 5 on the second within each. Deterministic for a given
    catalog and linear-algebra library (component signs only relabel cells).
    """
    X = df[AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    cells = {}
    for target in FEATURE_CELL_TARGETS:
        member, signal = _cell_candidates(df, target)
        rows = np.flatnonzero(member)
        if len(rows) == 0:
            cells[target] = [rows] * (_CELL_PC1_BINS * _CELL_PC2_BINS)
            continue
        f = z[rows] - z[rows].mean(axis=0)
        pc = f @ np.linalg.svd(f, full_matrices=False)[2][:2].T
        b1 = np.empty(len(rows), dtype=int)
        b1[np.argsort(pc[:, 0], kind="stable")] = np.arange(len(rows)) * _CELL_PC1_BINS // len(rows)
        cell = np.empty(len(rows), dtype=int)
        for b in range(_CELL_PC1_BINS):
            sel = np.flatnonzero(b1 == b)
            b2 = np.empty(len(sel), dtype=int)
            b2[np.argsort(pc[sel, 1], kind="stable")] = np.arange(len(sel)) * _CELL_PC2_BINS // len(sel)
            cell[sel] = b * _CELL_PC2_BINS + b2
        by_cell = []
        for c in range(_CELL_PC1_BINS * _CELL_PC2_BINS):
            r = rows[cell == c]
            by_cell.append(r[np.argsort(-signal[r], kind="stable")])
        cells[target] = by_cell
    return cells


def _get_feature_cells(df: pd.DataFrame):
    """The catalog's cells, built at most once per catalog object (the app
    loads one catalog per process). None if the build failed; not retried."""
    ref = _cell_cache["catalog"]
    if ref is not None and ref() is df:
        return _cell_cache["cells"]
    with _cell_lock:
        ref = _cell_cache["catalog"]
        if ref is not None and ref() is df:
            return _cell_cache["cells"]
        try:
            cells = _build_feature_cells(df)
        except Exception as exc:  # serve the standard ranking rather than fail the request
            logger.warning("Feature cells unavailable (%s); music requests use the standard ranking",
                           type(exc).__name__)
            cells = None
        _cell_cache["catalog"], _cell_cache["cells"] = weakref.ref(df), cells
        return cells


def _cell_select(signal: np.ndarray, available: np.ndarray, n: int, by_cell) -> np.ndarray:
    """
    Row positions of n available tracks, best-first: cells are visited in a
    random order and each contributes one pick, weighted as in _sample_best,
    among its _CELL_TOP most confident available tracks. More than one round
    is made only when n exceeds the cells that still have tracks.
    """
    available = available.copy()
    picks = []
    while len(picks) < n:
        before = len(picks)
        for c in np.random.permutation(len(by_cell)):
            top = by_cell[c][available[by_cell[c]]][:_CELL_TOP]
            if len(top) == 0:
                continue
            weights = signal[top] - signal[top].min() + 1e-6
            pick = int(np.random.choice(top, p=weights / weights.sum()))
            picks.append(pick)
            available[pick] = False
            if len(picks) == n:
                break
        if len(picks) == before:
            break
    picks = np.array(picks, dtype=int)
    return picks[np.argsort(-signal[picks], kind="stable")]


def _feature_cell_top(df: pd.DataFrame, target: str, top_n: int, by_cell):
    """Same columns and order convention as _coverage_top for one target, or
    None when fewer than top_n cell candidates exist (lists are never padded)."""
    member, signal = _cell_candidates(df, target)
    if int(member.sum()) < top_n:
        return None
    chosen = _cell_select(signal, member, top_n, by_cell)
    return df.iloc[chosen].assign(score=signal[chosen], target=[target] * len(chosen))


def recommend_music(df_music: pd.DataFrame, mood: str, mode: str, top_n: int = 10) -> pd.DataFrame:
    music_targets = _target_music_weighted(mood, mode)
    if not music_targets:
        raise ValueError(f"No music targets for mood {mood!r}")
    if not _is_278k_catalog(df_music, music_targets):
        raise ValueError("recommend_music needs the 278k catalog (data/music_catalog_278k.csv, "
                         "built by train_music_278k.py)")
    if _uses_feature_cells(music_targets):
        target = music_targets[0][0]
        state = np.random.get_state()
        try:
            cells = _get_feature_cells(df_music)
            if cells is not None:
                out = _feature_cell_top(df_music, target, top_n, cells[target])
                if out is not None:
                    return out
                logger.warning("Too few feature-cell candidates for %s; serving the standard ranking", target)
        except Exception as exc:  # fall back to the standard ranking, as if cells were never tried
            logger.warning("Feature-cell selection failed (%s); serving the standard ranking", type(exc).__name__)
            np.random.set_state(state)
    return _coverage_top(df_music, music_targets, top_n)


def recommend_movies(df_movies: pd.DataFrame, mood: str, mode: str, top_n: int = 10) -> pd.DataFrame:
    df = df_movies.copy()
    weighted_targets = _target_moods_weighted(mood, mode)

    available = [(t, w) for t, w in weighted_targets if f"proba_{t}" in df.columns]

    if available:
        total_w = sum(w for _, w in available)  # renormalize over only the moods this dataset actually has
        mood_score = sum((w / total_w) * df[f"proba_{t}"].values for t, w in available)
        rating_norm = _minmax(df["rating"]) if "rating" in df.columns else np.zeros(len(df))
        pop_norm = _minmax(df["popularity"]) if "popularity" in df.columns else np.zeros(len(df))
        df = df.assign(
            score=0.65 * mood_score + 0.20 * rating_norm + 0.15 * pop_norm,
        )
    elif "rating" in df.columns and "popularity" in df.columns:
        # fallback if the ML-scored columns aren't present for some reason
        df = df.assign(score=0.7 * _minmax(df["rating"]) + 0.3 * _minmax(df["popularity"]))
    elif "rating" in df.columns:
        df = df.assign(score=df["rating"])
    elif "popularity" in df.columns:
        df = df.assign(score=df["popularity"])
    else:
        df = df.assign(score=0.0)

    return _diverse_top(df, top_n)
