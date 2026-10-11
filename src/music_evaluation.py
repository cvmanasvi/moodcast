"""
Offline evaluation of the music *recommender* (not the classifier).

Classifier accuracy (models/music_mood_classifier_278k_eval.json) says how
well audio features predict the dataset's labels. It says nothing about the
lists users see. This module measures those lists, on the 278k catalog, with
relevance taken from what the catalog already defines -- no new ground truth:

- base targets (sad / happy / energetic / calm): relevant = the dataset's
  label. Those labels are inherited from the Moodify dataset, which describes
  them as model-generated (see DATA.md), not human-annotated per track.
- angry: relevant = `derived_angry` (a percentile feature rule).
- stressed: relevant = `derived_stressed` (an EXPERIMENTAL percentile rule with
  guards). Neither derived mood is a ground-truth class.
- app moods map to targets through MOOD_TO_MUSIC_TARGETS / MOOD_TO_TAGS_CHEER
  (e.g. chill -> calm; "Cheer me up" on sad -> happy + calm). A request's
  relevant set is the union of its targets' sets.

Because the recommender only ever picks from those same candidate sets,
precision against them is 1.0 by construction: it verifies the gating, not
quality. The comparisons that carry information are against the baselines
(uniform random over the catalog; random over the same eligible candidates,
i.e. the recommender without its classifier ranking), plus coverage,
diversity and stability.

Nothing here changes ranking. Recommendations come from
src.recommend.recommend_music exactly as the API calls it.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from .config import DERIVED_MUSIC_MOODS, LABEL_TO_MOOD, MOOD_TO_MUSIC_TARGETS, MOODS
from .music_pipeline import AUDIO_FEATURES
from . import recommend

MODES = {"match": "Match my mood", "uplift": "Cheer me up"}
KS = (5, 10, 20, 30)
LIST_SIZE = 30


# ---- metrics (binary relevance) ----

def precision_at_k(rel: np.ndarray, k: int) -> float:
    return float(np.mean(rel[:k])) if k > 0 and len(rel) else 0.0


def recall_at_k(rel: np.ndarray, k: int, n_relevant: int) -> float:
    return float(np.sum(rel[:k]) / n_relevant) if n_relevant else 0.0


def ndcg_at_k(rel: np.ndarray, k: int, n_relevant: int) -> float:
    """Binary-relevance NDCG: DCG of the list over the DCG of an ideal list
    with min(k, n_relevant) relevant items at the top."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel[:k] * discounts[:len(rel[:k])]))
    idcg = float(np.sum(discounts[:min(k, n_relevant)]))
    return dcg / idcg if idcg else 0.0


def intra_list_diversity(features: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between the list's tracks, in the
    catalog-standardized (z-scored) 9-feature audio space."""
    if len(features) < 2:
        return 0.0
    diffs = features[:, None, :] - features[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    return float(dists[np.triu_indices(len(features), k=1)].mean())


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


# ---- relevance ----

def target_members(catalog: pd.DataFrame, target: str) -> np.ndarray:
    """Bool mask of the tracks relevant to one target (same definition the
    recommender uses to choose candidates)."""
    member, _ = recommend._target_candidates(catalog, target)
    return member


def request_targets(mood: str, mode: str):
    return recommend._target_music_weighted(mood, MODES[mode])


def target_kind(target: str) -> str:
    if target in DERIVED_MUSIC_MOODS:
        return "derived (experimental rule)" if DERIVED_MUSIC_MOODS[target].get("experimental") else "derived rule"
    return "dataset label"


# ---- baselines (evaluation only; the app never uses them) ----

def random_catalog_list(n_rows: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Uniform random tracks from the whole catalog."""
    return rng.choice(n_rows, size=k, replace=False)


def random_eligible_list(members, weights, k: int, rng: np.random.RandomState) -> np.ndarray:
    """The recommender's target shares, filled uniformly at random from each
    target's candidates -- i.e. the recommender without its ranking signal."""
    taken, picks = set(), []
    for member, share in zip(members, recommend._allocate_shares(weights, k)):
        pool = np.setdiff1d(np.flatnonzero(member), list(taken), assume_unique=True)
        chosen = rng.choice(pool, size=min(share, len(pool)), replace=False)
        taken.update(chosen.tolist())
        picks.append(chosen)
    return np.concatenate(picks)


# ---- evaluation ----

def _list_metrics(rows: np.ndarray, relevant: np.ndarray, n_relevant: int, z: np.ndarray) -> dict:
    rel = relevant[rows].astype(float)
    out = {}
    for k in KS:
        out[f"precision@{k}"] = precision_at_k(rel, k)
        out[f"recall@{k}"] = recall_at_k(rel, k, n_relevant)
        out[f"ndcg@{k}"] = ndcg_at_k(rel, k, n_relevant)
    out["diversity"] = intra_list_diversity(z[rows])
    out["duplicates"] = int(len(rows) - len(set(rows.tolist())))
    return out


def _mean(dicts):
    return {key: round(float(np.mean([d[key] for d in dicts])), 6) for key in dicts[0]}


def catalog_checks(catalog: pd.DataFrame) -> dict:
    """Structural checks of the canonical catalog the evaluation runs on."""
    derived_cols = [c for m in DERIVED_MUSIC_MOODS for c in (f"derived_{m}", f"derived_{m}_score")]
    return {
        "rows": int(len(catalog)),
        "unique_track_ids": int(catalog["track_id"].nunique()),
        "audio_features_missing_values": int(catalog[AUDIO_FEATURES].isna().sum().sum()),
        "labels": sorted(catalog["mood"].unique().tolist()),
        "label_counts": {k: int(v) for k, v in catalog["mood"].value_counts().items()},
        "derived_columns_present": all(c in catalog.columns for c in derived_cols),
        "derived_counts": {m: int(catalog[f"derived_{m}"].sum()) for m in DERIVED_MUSIC_MOODS},
        "has_popularity_column": "popularity" in catalog.columns,
    }


def evaluate(catalog: pd.DataFrame, runs: int = 50, seed: int = 2026, moods=MOODS) -> dict:
    """
    For every (app mood, mode) request: `runs` recommendation lists of
    LIST_SIZE tracks from recommend_music (np.random seeded per run), plus the
    same number of lists from each baseline. Returns per-request and overall
    metrics; no track IDs are included in the output.
    """
    catalog = catalog.reset_index(drop=True)
    X = catalog[AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    labels = catalog["mood"].to_numpy()
    predicted = catalog["predicted_mood"].to_numpy()
    n = len(catalog)

    requests, latencies = {}, []
    served_by_mood = {m: set() for m in moods}
    all_served, total_slots = set(), 0
    for mood in moods:
        for mode in MODES:
            targets = request_targets(mood, mode)
            names, weights = [t for t, _ in targets], [w for _, w in targets]
            members = [target_members(catalog, t) for t in names]
            relevant = np.logical_or.reduce(members)
            n_relevant = int(relevant.sum())

            lists, per_list, per_target_hits = [], [], {t: 0 for t in names}
            for r in range(runs):
                np.random.seed(seed + r)
                t0 = time.perf_counter()
                recs = recommend.recommend_music(catalog, mood, MODES[mode], top_n=LIST_SIZE)
                latencies.append((time.perf_counter() - t0) * 1000)
                rows = recs.index.to_numpy()
                lists.append(rows)
                per_list.append(_list_metrics(rows, relevant, n_relevant, z))
                for t, member in zip(names, members):
                    per_target_hits[t] += int(member[rows].sum())

            rng = np.random.RandomState(seed)
            rand_cat = [_list_metrics(random_catalog_list(n, LIST_SIZE, rng), relevant, n_relevant, z)
                        for _ in range(runs)]
            rand_elig = [_list_metrics(random_eligible_list(members, weights, LIST_SIZE, rng), relevant, n_relevant, z)
                         for _ in range(runs)]

            served = set(itertools.chain.from_iterable(l.tolist() for l in lists))
            served_by_mood[mood] |= served
            all_served |= served
            total_slots += runs * LIST_SIZE
            overlaps = [jaccard(a, b) for a, b in itertools.combinations(lists, 2)]
            base_rows = [r for l in lists for r in l if labels[r] in names]
            np.random.seed(seed)
            again = recommend.recommend_music(catalog, mood, MODES[mode], top_n=LIST_SIZE).index.to_numpy()

            requests[f"{mood} / {mode}"] = {
                "targets": {t: {"weight": round(w, 4), "relevance": target_kind(t),
                                "relevant_tracks_in_catalog": int(m.sum())}
                            for t, w, m in zip(names, weights, members)},
                "relevant_tracks_for_request": n_relevant,
                "moodcast": _mean(per_list),
                "random_catalog_baseline": _mean(rand_cat),
                "random_eligible_baseline": _mean(rand_elig),
                "mood_consistency": {t: round(per_target_hits[t] / (runs * LIST_SIZE), 4) for t in names},
                "label_mix_of_served": {k: round(v / (runs * LIST_SIZE), 4)
                                        for k, v in pd.Series(labels[np.concatenate(lists)]).value_counts().items()},
                "classifier_agreement_on_label_targets": (
                    round(float(np.mean(predicted[base_rows] == labels[base_rows])), 4) if base_rows else None),
                "stability": {
                    "runs": runs,
                    "unique_tracks_over_runs": len(served),
                    "mean_pairwise_jaccard": round(float(np.mean(overlaps)), 4),
                    "repeat_rate_across_runs": round(1 - len(served) / (runs * LIST_SIZE), 4),
                    "same_seed_identical": bool(np.array_equal(again, lists[0])),
                },
            }

    def overall(key):
        return _mean([req[key] for req in requests.values()])

    return {
        "list_size": LIST_SIZE,
        "ks": list(KS),
        "runs_per_request": runs,
        "seed": seed,
        "requests": requests,
        "overall_mean_over_requests": {
            "moodcast": overall("moodcast"),
            "random_catalog_baseline": overall("random_catalog_baseline"),
            "random_eligible_baseline": overall("random_eligible_baseline"),
        },
        "catalog_coverage": {
            "unique_tracks_served": len(all_served),
            "catalog_tracks": n,
            "coverage": round(len(all_served) / n, 6),
            "recommendation_slots": total_slots,
            "by_mood": {m: {"unique_tracks_served": len(s), "coverage": round(len(s) / n, 6)}
                        for m, s in served_by_mood.items()},
        },
        "latency_ms": {"median": round(float(np.median(latencies)), 2),
                       "p95": round(float(np.percentile(latencies, 95)), 2),
                       "calls": len(latencies)},
    }


DEFINITIONS = {
    "relevance": {
        "dataset label": "track's `label` equals the target (sad/happy/energetic/calm). Labels come from the "
                         "Moodify dataset, which describes them as model-generated; not human per-track ground truth.",
        "derived rule": "track's `derived_angry` is true: valence <= 25th pct, energy >= 75th pct, loudness >= 75th "
                        "pct of the catalog. A feature rule, not a label.",
        "derived (experimental rule)": "track's `derived_stressed` is true: the stressed percentile rule after the "
                                       "failed-analysis and spoken-word guards. Experimental, not a label.",
        "request": "a request's relevant set is the union of its targets' sets (app mood -> targets via "
                   "MOOD_TO_MUSIC_TARGETS; 'Cheer me up' first maps the mood through MOOD_TO_TAGS_CHEER).",
        "app_mood_targets": MOOD_TO_MUSIC_TARGETS,
        "label_names": LABEL_TO_MOOD,
    },
    "precision@k": "share of the first k tracks that are relevant",
    "recall@k": "relevant tracks in the first k / all relevant tracks in the 278k catalog (tens of thousands per "
                "target, so recall@30 is tiny by construction)",
    "ndcg@k": "binary-relevance NDCG; ideal DCG has min(k, |relevant|) relevant tracks at the top",
    "diversity": "mean pairwise Euclidean distance between a list's tracks in the 9 audio features, each z-scored "
                 "over the catalog (no genre: the catalog has none)",
    "duplicates": "repeated track IDs within one list",
    "mood_consistency": "per target: share of served slots whose track satisfies that target's definition",
    "classifier_agreement_on_label_targets": "for served tracks whose label is a requested label target, share where "
                                             "the 278k classifier's predicted mood equals the label. Circular: the "
                                             "ranking itself prefers high classifier probability.",
    "catalog_coverage": "unique tracks served across all evaluation requests / catalog tracks",
    "stability": "over `runs` requests with different seeds: unique tracks, mean pairwise Jaccard overlap of the "
                 "lists, and whether the same seed reproduces the same list",
    "random_catalog_baseline": "uniform random tracks from the whole catalog",
    "random_eligible_baseline": "the recommender's per-target shares, filled uniformly at random from each target's "
                                "candidates (the recommender minus its classifier ranking)",
}

LIMITATIONS = [
    "Precision/NDCG against the label/rule definitions are 1.0 by construction: the recommender only selects "
    "candidates from those sets. They verify the gating, not whether users would like the tracks.",
    "No human relevance judgments, listening data or user feedback exist for this catalog; this is an offline, "
    "label-consistency evaluation only.",
    "Base-mood labels are inherited from the Moodify dataset and described there as model-generated.",
    "angry is a derived feature rule; stressed is a derived, experimental rule; chill is an application mapping to "
    "the calm label. None of them is ground truth.",
    "Recall@K over a 278k catalog is bounded by K / |relevant| and is near zero for any list of 30.",
    "Classifier agreement is circular (the ranking prefers high classifier probability).",
    "The classifier's 83.5% held-out accuracy measures label prediction, not recommendation quality.",
]
