"""
ANALYSIS ONLY -- offline sweep of candidate-pool size and sampling weights for
the music recommender. It changes nothing in production: the app calls
src.recommend.recommend_music, which uses _coverage_top's default
pool_multiplier (3 when this sweep was committed in Phase 2H.5, 20 since
Phase 2H.6) and classifier-signal weighting.

Run from the project root:
    python evaluate_music_recommender_analysis.py [--runs N] [--output PATH]

Reads   data/music_catalog_278k.csv (never modified)
Writes  models/music_recommender_sweep.json, or --output (aggregate metrics
        only: no track IDs or catalog rows). models/music_recommender_eval.json
        is not touched.

Two separate experiments, same 12 (mood, mode) requests, same seeds and list
size as evaluate_music_recommender.py:

A. Pool size: the production function recommend._coverage_top called with
   pool_multiplier in POOL_MULTIPLIERS (production weighting).
B. Sampling weights: a replica of _coverage_top/_sample_best whose only
   difference is the weight function (WEIGHTINGS). With "current" weighting
   the replica reproduces production lists exactly (checked at run time and in
   tests/test_music_sweep.py).
"""
import argparse
import datetime
import inspect
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend

SWEEP_PATH = os.path.join(MODELS_DIR, "music_recommender_sweep.json")
POOL_MULTIPLIERS = (1, 2, 3, 5, 10, 20, 50, 100)
PRODUCTION_POOL_MULTIPLIER = inspect.signature(recommend._coverage_top).parameters["pool_multiplier"].default
SAMPLING_POOL_MULTIPLIERS = (3, 10, 20)

# Weight given to each pool candidate when sampling without replacement.
# "current" is _sample_best's formula; the others only flatten it.
WEIGHTINGS = {
    "current": lambda s: s - s.min() + 1e-6,                 # linear in signal above the pool minimum
    "sqrt": lambda s: np.sqrt(s - s.min()) + 1e-6,           # flatter, same order
    "uniform": lambda s: np.ones(len(s)),                     # every pool candidate equally likely
}


# ---- replica of the production sampler with a pluggable weight function ----

def _sample_best_variant(signal, available, n, pool_multiplier, weight_fn):
    """recommend._sample_best with `weights` computed by weight_fn; identical otherwise."""
    idx = np.flatnonzero(available)
    if n <= 0 or len(idx) == 0:
        return np.array([], dtype=int)
    pool_size = min(len(idx), n * pool_multiplier)
    if pool_size < len(idx):
        idx = idx[np.argpartition(-signal[idx], pool_size - 1)[:pool_size]]
    if len(idx) > n:
        weights = weight_fn(signal[idx])
        idx = np.random.choice(idx, size=n, replace=False, p=weights / weights.sum())
    return idx[np.argsort(-signal[idx], kind="stable")]


def coverage_top_variant(df, targets, top_n, pool_multiplier, weight_fn):
    """recommend._coverage_top with _sample_best_variant; identical otherwise.
    Returns row positions in list order."""
    shares = recommend._allocate_shares([w for _, w in targets], top_n)
    taken = np.zeros(len(df), dtype=bool)
    picks = []
    for (target, _), share in zip(targets, shares):
        member, signal = recommend._target_candidates(df, target)
        chosen = _sample_best_variant(signal, member & ~taken, share, pool_multiplier, weight_fn)
        taken[chosen] = True
        picks.append([target, chosen, member, signal])
    missing = top_n - int(taken.sum())
    for pick in picks:
        if missing <= 0:
            break
        target, chosen, member, signal = pick
        extra = _sample_best_variant(signal, member & ~taken, missing, pool_multiplier, weight_fn)
        taken[extra] = True
        missing -= len(extra)
        combined = np.concatenate([chosen, extra])
        pick[1] = combined[np.argsort(-signal[combined], kind="stable")]
    slots = []
    for t_i, (target, chosen, _, signal) in enumerate(picks):
        for k, row in enumerate(chosen):
            slots.append(((k + 0.5) / len(chosen), t_i, row))
    slots.sort(key=lambda slot: slot[:2])
    return np.array([slot[2] for slot in slots], dtype=int)


def production_rows(df, targets, top_n, pool_multiplier):
    return recommend._coverage_top(df, targets, top_n, pool_multiplier=pool_multiplier).index.to_numpy()


# ---- one configuration ----

def run_config(catalog, z, recommender, pool_multiplier, runs, seed, moods=MOODS):
    """recommender(targets, seed) -> row positions. Same request set and seeds as ev.evaluate."""
    n = len(catalog)
    requests, latencies, all_served, consistency = {}, [], set(), []
    for mood in moods:
        for mode in ev.MODES:
            targets = ev.request_targets(mood, mode)
            names, weights = [t for t, _ in targets], [w for _, w in targets]
            members = [ev.target_members(catalog, t) for t in names]
            signals = [recommend._target_candidates(catalog, t)[1] for t in names]
            relevant = np.logical_or.reduce(members)
            n_relevant = int(relevant.sum())
            shares = dict(zip(names, recommend._allocate_shares(weights, ev.LIST_SIZE)))

            lists, per_list, shares_exact, served_signal = [], [], 0, []
            for r in range(runs):
                np.random.seed(seed + r)
                t0 = time.perf_counter()
                rows = recommender(targets, seed + r)
                latencies.append((time.perf_counter() - t0) * 1000)
                lists.append(rows)
                per_list.append(ev._list_metrics(rows, relevant, n_relevant, z))
                counts = {t: int(m[rows].sum()) for t, m in zip(names, members)}
                shares_exact += counts == shares
                consistency.append(float(relevant[rows].mean()))
                for t, m, s in zip(names, members, signals):
                    served_signal.extend(s[rows[m[rows]]].tolist())

            np.random.seed(seed)
            again = recommender(targets, seed)
            served = set(itertools.chain.from_iterable(l.tolist() for l in lists))
            all_served |= served
            pool = sum(min(int(m.sum()), share * pool_multiplier) for m, share in zip(members, shares.values()))
            requests[f"{mood} / {mode}"] = {
                "metrics": ev._mean(per_list),
                "target_consistency": round(float(np.mean([relevant[l].mean() for l in lists])), 6),
                "exact_target_shares_fraction": round(shares_exact / runs, 4),
                "effective_candidate_pool": pool,
                "unique_tracks_over_runs": len(served),
                "repeat_rate": round(1 - len(served) / (runs * ev.LIST_SIZE), 4),
                "mean_pairwise_jaccard": round(float(np.mean([ev.jaccard(a, b)
                                                              for a, b in itertools.combinations(lists, 2)])), 4),
                "same_seed_identical": bool(np.array_equal(again, lists[0])),
                "mean_ranking_signal_of_served": round(float(np.mean(served_signal)), 4),
            }
    reqs = list(requests.values())
    return {
        "requests": requests,
        "summary": {
            "target_consistency": round(float(np.mean(consistency)), 6),
            "exact_target_shares_all_lists": all(r["exact_target_shares_fraction"] == 1.0 for r in reqs),
            "catalog_unique_tracks": len(all_served),
            "catalog_coverage": round(len(all_served) / n, 6),
            "mean_repeat_rate": round(float(np.mean([r["repeat_rate"] for r in reqs])), 4),
            "mean_unique_tracks_per_request": round(float(np.mean([r["unique_tracks_over_runs"] for r in reqs])), 1),
            "mean_diversity": round(float(np.mean([r["metrics"]["diversity"] for r in reqs])), 4),
            "mean_pairwise_jaccard": round(float(np.mean([r["mean_pairwise_jaccard"] for r in reqs])), 4),
            "all_same_seed_identical": all(r["same_seed_identical"] for r in reqs),
            "mean_effective_candidate_pool": round(float(np.mean([r["effective_candidate_pool"] for r in reqs])), 1),
            "mean_ranking_signal_of_served": round(float(np.mean([r["mean_ranking_signal_of_served"] for r in reqs])), 4),
            "precision@10": round(float(np.mean([r["metrics"]["precision@10"] for r in reqs])), 6),
            "ndcg@10": round(float(np.mean([r["metrics"]["ndcg@10"] for r in reqs])), 6),
            "recall@30": round(float(np.mean([r["metrics"]["recall@30"] for r in reqs])), 6),
            "latency_ms": {"median": round(float(np.median(latencies)), 2),
                           "p95": round(float(np.percentile(latencies, 95)), 2),
                           "max": round(float(np.max(latencies)), 2)},
        },
    }


def random_eligible_reference(catalog, z, runs, seed, moods=MOODS):
    """The existing random-eligible baseline, per request (same RNG scheme as ev.evaluate)."""
    out = {}
    for mood in moods:
        for mode in ev.MODES:
            targets = ev.request_targets(mood, mode)
            names, weights = [t for t, _ in targets], [w for _, w in targets]
            members = [ev.target_members(catalog, t) for t in names]
            signals = [recommend._target_candidates(catalog, t)[1] for t in names]
            relevant = np.logical_or.reduce(members)
            rng = np.random.RandomState(seed)
            _ = [ev.random_catalog_list(len(catalog), ev.LIST_SIZE, rng) for _ in range(runs)]  # keep RNG aligned
            lists = [ev.random_eligible_list(members, weights, ev.LIST_SIZE, rng) for _ in range(runs)]
            served = set(itertools.chain.from_iterable(l.tolist() for l in lists))
            sig = [s[l[m[l]]] for l in lists for m, s in zip(members, signals)]
            out[f"{mood} / {mode}"] = {
                "diversity": round(float(np.mean([ev.intra_list_diversity(z[l]) for l in lists])), 4),
                "unique_tracks_over_runs": len(served),
                "repeat_rate": round(1 - len(served) / (runs * ev.LIST_SIZE), 4),
                "mean_pairwise_jaccard": round(float(np.mean([ev.jaccard(a, b)
                                                              for a, b in itertools.combinations(lists, 2)])), 4),
                "target_consistency": round(float(np.mean([relevant[l].mean() for l in lists])), 6),
                "mean_ranking_signal_of_served": round(float(np.mean(np.concatenate(sig))), 4),
            }
    vals = list(out.values())
    return {"requests": out, "summary": {
        "mean_diversity": round(float(np.mean([v["diversity"] for v in vals])), 4),
        "mean_repeat_rate": round(float(np.mean([v["repeat_rate"] for v in vals])), 4),
        "mean_pairwise_jaccard": round(float(np.mean([v["mean_pairwise_jaccard"] for v in vals])), 4),
        "mean_ranking_signal_of_served": round(float(np.mean([v["mean_ranking_signal_of_served"] for v in vals])), 4),
    }}


def compare_to_reference(config, reference):
    per = {}
    for name, req in config["requests"].items():
        ref = reference["requests"][name]
        per[name] = {"diversity_minus_random_eligible": round(req["metrics"]["diversity"] - ref["diversity"], 4),
                     "diversity_at_least_random_eligible": req["metrics"]["diversity"] >= ref["diversity"]}
    return {
        "requests_with_diversity_at_least_random_eligible": sum(p["diversity_at_least_random_eligible"]
                                                                for p in per.values()),
        "mean_diversity_minus_random_eligible": round(float(np.mean([p["diversity_minus_random_eligible"]
                                                                     for p in per.values()])), 4),
        "per_request": per,
    }


def pareto_front(configs):
    """Non-dominated configs: maximize consistency, coverage, diversity; minimize repeat rate, p95 latency."""
    def vec(s):
        return (s["target_consistency"], s["catalog_coverage"], s["mean_diversity"],
                -s["mean_repeat_rate"], -s["latency_ms"]["p95"])
    items = [(name, vec(c["summary"])) for name, c in configs.items()]
    front = []
    for name, v in items:
        dominated = any(all(o >= x for o, x in zip(w, v)) and any(o > x for o, x in zip(w, v))
                        for other, w in items if other != name)
        if not dominated:
            front.append(name)
    return front


def main():
    parser = argparse.ArgumentParser(description="Offline pool-size / sampling sweep (analysis only)")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=SWEEP_PATH, help="report path (default: %(default)s)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    catalog_hash = mp.file_sha256(MUSIC_CATALOG_278K_PATH)
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)

    # fidelity: the replica with production weighting reproduces production lists exactly
    for mood, mode in [(m, md) for m in MOODS for md in ev.MODES]:
        targets = ev.request_targets(mood, mode)
        for s in range(3):
            np.random.seed(args.seed + s)
            a = production_rows(catalog, targets, ev.LIST_SIZE, PRODUCTION_POOL_MULTIPLIER)
            np.random.seed(args.seed + s)
            b = coverage_top_variant(catalog, targets, ev.LIST_SIZE, PRODUCTION_POOL_MULTIPLIER, WEIGHTINGS["current"])
            if not np.array_equal(a, b):
                raise SystemExit(f"Replica diverges from production for {mood}/{mode} -- sampling sweep invalid")
    print("Replica fidelity check passed (identical to production for all 12 requests x 3 seeds)")

    reference = random_eligible_reference(catalog, z, args.runs, args.seed)
    print(f"random-eligible: {reference['summary']}")

    pool_sweep = {}
    for pm in POOL_MULTIPLIERS:
        cfg = run_config(catalog, z, lambda t, s, pm=pm: production_rows(catalog, t, ev.LIST_SIZE, pm),
                         pm, args.runs, args.seed)
        cfg["vs_random_eligible"] = compare_to_reference(cfg, reference)
        pool_sweep[f"pool_multiplier={pm}"] = cfg
        s = cfg["summary"]
        print(f"pm={pm:>3}: cons={s['target_consistency']:.4f} cov={s['catalog_unique_tracks']} "
              f"({s['catalog_coverage']:.4%}) repeat={s['mean_repeat_rate']:.3f} div={s['mean_diversity']:.3f} "
              f"jac={s['mean_pairwise_jaccard']:.3f} signal={s['mean_ranking_signal_of_served']:.3f} "
              f"div>=RE {cfg['vs_random_eligible']['requests_with_diversity_at_least_random_eligible']}/12 "
              f"lat={s['latency_ms']}")

    sampling_sweep = {}
    for pm in SAMPLING_POOL_MULTIPLIERS:
        for wname, wfn in WEIGHTINGS.items():
            cfg = run_config(catalog, z, lambda t, s, pm=pm, wfn=wfn:
                             coverage_top_variant(catalog, t, ev.LIST_SIZE, pm, wfn), pm, args.runs, args.seed)
            cfg["vs_random_eligible"] = compare_to_reference(cfg, reference)
            sampling_sweep[f"pool_multiplier={pm}, weighting={wname}"] = cfg
            s = cfg["summary"]
            print(f"pm={pm:>3} {wname:>8}: cov={s['catalog_unique_tracks']} repeat={s['mean_repeat_rate']:.3f} "
                  f"div={s['mean_diversity']:.3f} jac={s['mean_pairwise_jaccard']:.3f} "
                  f"signal={s['mean_ranking_signal_of_served']:.3f} "
                  f"div>=RE {cfg['vs_random_eligible']['requests_with_diversity_at_least_random_eligible']}/12")

    all_configs = {**{f"A: {k}": v for k, v in pool_sweep.items()},
                   **{f"B: {k}": v for k, v in sampling_sweep.items() if not k.endswith("weighting=current")}}
    report = {
        "analysis_only": True,
        "evaluation_date": datetime.date.today().isoformat(),
        "production_configuration": {"pool_multiplier": PRODUCTION_POOL_MULTIPLIER, "weighting": "current"},
        "catalog": {"path": os.path.relpath(MUSIC_CATALOG_278K_PATH, os.path.dirname(MODELS_DIR)),
                    "sha256": catalog_hash, "rows": int(len(catalog))},
        "runs_per_request": args.runs, "seed": args.seed, "list_size": ev.LIST_SIZE,
        "weightings": {"current": "w = s - min(pool s) + 1e-6 (production _sample_best)",
                       "sqrt": "w = sqrt(s - min(pool s)) + 1e-6",
                       "uniform": "w = 1 for every pool candidate"},
        "replica_fidelity": "replica with current weighting == production for 12 requests x 3 seeds",
        "random_eligible_reference": reference,
        "pool_size_sweep": pool_sweep,
        "sampling_sweep": sampling_sweep,
        "pareto_front": pareto_front(all_configs),
        "notes": [
            "Precision/NDCG are 1.0 by construction (candidates are gated by the relevance definitions); "
            "they are reported but are not evidence of quality.",
            "mean_ranking_signal_of_served is the classifier probability (label targets) or derived score "
            "(angry/stressed) of served tracks: what a larger pool or flatter weighting trades away. It is the "
            "ranking's own signal, so it is not an independent quality measure either.",
            "effective_candidate_pool = sum over targets of min(|candidates|, share x pool_multiplier).",
        ],
        "runtime_s": None,
    }
    if mp.file_sha256(MUSIC_CATALOG_278K_PATH) != catalog_hash:
        raise SystemExit("The catalog changed during the analysis -- this must never happen.")
    report["runtime_s"] = round(time.perf_counter() - t0, 1)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Pareto front: {report['pareto_front']}")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
