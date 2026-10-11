"""
ANALYSIS / PROTOTYPE ONLY -- faster stratified within-pool selection for the
music recommender (Phase 2H.8). Nothing here changes production: the app
still calls src.recommend.recommend_music (pool_multiplier=20, production
weighting). Builds on evaluate_music_diversity_analysis.py (Phase 2H.7) and
reuses its replica, metrics, bootstrap and seeds.

Run from the project root:
    python evaluate_music_stratified_prototype.py [--runs N] [--output PATH]

Reads   data/music_catalog_278k.csv (read-only)
Writes  models/music_stratified_prototype.json (aggregate metrics only: no
        track IDs, titles, artists or catalog rows). No other artifact is touched.

Profiling (Phase 2H.8) showed the 2H.7 stratified selector's cost is almost all
scikit-learn KMeans per-call overhead (~14 ms warm, ~2.4 s on the first call in
a process) for a 600 x 9 problem that converges in ~10 Lloyd iterations. The
prototypes keep the 2H.7 contract (same pool, eligibility, exact shares,
interleaving; one production-weighted pick per stratum) and change only how the
pool is partitioned:
  D  stratified k-means   Phase 2H.7: sklearn KMeans(k = share, n_init=1)
  E  numpy k-means        k-means++ seeding + Lloyd iterations until labels stop
                          changing (at most LLOYD_MAX_ITER), numpy only, no cache
  F  PC1 quantile strata  sort the pool along its first principal component and
                          cut it into `share` equal-count strata (no iterations)
Selection (fixed before results): among the prototypes E and F, the simplest
(F, then E) that passes every gate on the exploratory seeds; its gates are then
re-checked on the held-out seeds, which are never used to choose anything.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_music_diversity_analysis as div
import evaluate_music_recommender_analysis as sweep
from src.config import MODELS_DIR, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp

OUTPUT_PATH = os.path.join(MODELS_DIR, "music_stratified_prototype.json")
LLOYD_MAX_ITER = 20
SPEECH_LIKE = 0.66          # speechiness above this is speech-like (the stressed guard's threshold)
CALM_REQUESTS = ("chill / match", "stressed / uplift")
COLD_START_PROCESSES = 5
SIMPLICITY_ORDER = ("F: PC1 quantile strata", "E: numpy k-means")

# Fixed before results (the Phase 2H.8 brief's gates; thresholds are not tuned).
GATES = {
    "target_consistency": 1.0,
    "exact_target_shares_all_lists": True,
    "all_same_seed_identical": True,
    "min_single_target_mean_diversity": 3.3,
    "min_sad_match_diversity": 2.0,
    "min_single_target_requests_at_least_random_eligible": 5,
    "paired_lower_bound_positive_for": ["happy / match", "sad / match", "energetic / match", "stressed / match"],
    "min_mean_ranking_signal": 0.99,
    "min_catalog_unique_tracks": 2000,
    "max_mean_repeat_rate": 0.75,
    "max_latency_p95_ms": 25.0,
    "max_calm_speech_like_share": 0.20,
}


# ---- prototypes ----

def _one_pick_per_stratum(idx, labels, n, signal):
    """One production-weighted pick per non-empty stratum; if fewer than n strata are
    non-empty, the rest are production-weighted picks from the unchosen pool."""
    picks = []
    for k in np.unique(labels):
        members = idx[labels == k]
        picks.append(np.random.choice(members, p=div._production_weights(signal[members])))
    if len(picks) < n:
        rest = np.setdiff1d(idx, picks)
        picks.extend(np.random.choice(rest, size=n - len(picks), replace=False,
                                      p=div._production_weights(signal[rest])).tolist())
    return div._best_first(np.array(picks, dtype=int), signal)


def kmeans_labels(f, k, max_iter=LLOYD_MAX_ITER):
    """k-means++ seeding (first centre uniform, then proportional to squared distance) and
    Lloyd iterations until the assignment stops changing, all from np.random."""
    centres = np.empty((k, f.shape[1]))
    centres[0] = f[np.random.randint(len(f))]
    d2 = ((f - centres[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        total = d2.sum()
        nxt = np.random.choice(len(f), p=d2 / total) if total > 0 else np.random.randint(len(f))
        centres[c] = f[nxt]
        d2 = np.minimum(d2, ((f - centres[c]) ** 2).sum(axis=1))
    f_sq = (f ** 2).sum(axis=1)[:, None]
    labels = None
    for _ in range(max_iter):
        # squared distances as |f|^2 + |c|^2 - 2 f.c (one small matrix product, no n x k x d temporary)
        new = (f_sq + (centres ** 2).sum(axis=1)[None, :] - 2 * f @ centres.T).argmin(axis=1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        onehot = np.zeros((len(f), k))
        onehot[np.arange(len(f)), labels] = 1
        counts = onehot.sum(axis=0)
        filled = counts > 0
        centres[filled] = (onehot.T @ f)[filled] / counts[filled, None]   # an empty cluster keeps its centre
    return labels


def numpy_kmeans_select(signal, available, n, pool_multiplier, z):
    idx = div.candidate_pool(signal, available, n, pool_multiplier)
    if len(idx) <= n:
        return div._best_first(idx, signal)
    return _one_pick_per_stratum(idx, kmeans_labels(z[idx], n), n, signal)


def pc1_strata_labels(f, k):
    """Equal-count strata along the first principal component (ties broken by position)."""
    centred = f - f.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    order = np.argsort(centred @ vt[0], kind="stable")
    labels = np.empty(len(f), dtype=int)
    labels[order] = np.arange(len(f)) * k // len(f)
    return labels


def pc1_strata_select(signal, available, n, pool_multiplier, z):
    idx = div.candidate_pool(signal, available, n, pool_multiplier)
    if len(idx) <= n:
        return div._best_first(idx, signal)
    return _one_pick_per_stratum(idx, pc1_strata_labels(z[idx], n), n, signal)


def strategies():
    return {"A: production": div.select_production,
            "D: stratified k-means (2H.7)": div.stratified_select,
            "E: numpy k-means": numpy_kmeans_select,
            "F: PC1 quantile strata": pc1_strata_select}


# ---- evaluation ----

def run(catalog, z, select, runs, seed):
    """div.run_strategy plus every call's latency and the served rows (kept in memory only)."""
    record, latencies = {}, []

    def recommender(targets, s):
        t0 = time.perf_counter()
        rows = div.coverage_top_selector(catalog, targets, ev.LIST_SIZE, div.POOL_MULTIPLIER, select, z)
        latencies.append((time.perf_counter() - t0) * 1000)
        record.setdefault((tuple(t for t, _ in targets), s), rows)
        return rows

    cfg = sweep.run_config(catalog, z, recommender, div.POOL_MULTIPLIER, runs, seed)
    lists, per_run = {}, {}
    for name in cfg["requests"]:
        mood, mode = name.split(" / ")
        key = tuple(t for t, _ in ev.request_targets(mood, mode))
        lists[name] = [record[(key, seed + r)] for r in range(runs)]
        per_run[name] = [ev.intra_list_diversity(z[rows]) for rows in lists[name]]
    return cfg, per_run, lists, latencies


def latency_summary(latencies):
    lat = np.asarray(latencies)
    return {"first_call_in_run_ms": round(float(lat[0]), 2),
            "steady_state": {"calls": int(len(lat) - 1), "median_ms": round(float(np.median(lat[1:])), 2),
                             "p95_ms": round(float(np.percentile(lat[1:], 95)), 2),
                             "max_ms": round(float(lat[1:].max()), 2)}}


def speech_like(catalog, lists):
    sp = catalog["speechiness"].to_numpy()
    out = {name: round(float((sp[np.concatenate(lists[name])] > SPEECH_LIKE).mean()), 4) for name in lists}
    return {"per_request": out, "calm_requests": round(float(np.mean([out[n] for n in CALM_REQUESTS])), 4)}


def evaluate(catalog, z, runs, seed, names, rng):
    single = list(div.single_target_requests())
    reference = sweep.random_eligible_reference(catalog, z, runs, seed)
    results, per_runs = {}, {}
    for name in names:
        cfg, per_runs[name], lists, lat = run(catalog, z, strategies()[name], runs, seed)
        res = div.summarize(cfg, reference, single)
        res["latency"] = latency_summary(lat)
        res["speech_like_share"] = speech_like(catalog, lists)
        results[name] = res
    for name in names:
        if name != "A: production":
            results[name]["paired_vs_production"] = div.paired(per_runs["A: production"], per_runs[name], single, rng)
    for name in names:
        results[name]["gates"] = gate_status(results[name])
    return results


def gate_status(res):
    s, g = res["summary"], GATES
    paired = res.get("paired_vs_production")
    status = {
        "target_consistency_100": s["target_consistency"] >= g["target_consistency"],
        "exact_target_shares": s["exact_target_shares_all_lists"] is True,
        "deterministic": s["all_same_seed_identical"] is True,
        "single_target_mean_diversity_>=3.3": s["single_target_mean_diversity"] >= g["min_single_target_mean_diversity"],
        "sad_match_diversity_>=2.0": res["requests"]["sad / match"]["diversity"] >= g["min_sad_match_diversity"],
        ">=5_of_7_single_target_at_least_random_eligible":
            s["single_target_requests_at_least_random_eligible"] >= g["min_single_target_requests_at_least_random_eligible"],
        "paired_lower_bounds_>0": (None if paired is None else
                                   all(paired[r]["bootstrap_95"][0] > 0 for r in g["paired_lower_bound_positive_for"])),
        "ranking_signal_>=0.99": s["mean_ranking_signal_of_served"] >= g["min_mean_ranking_signal"],
        "coverage_>=2000": s["catalog_unique_tracks"] >= g["min_catalog_unique_tracks"],
        "repeat_rate_<=0.75": s["mean_repeat_rate"] <= g["max_mean_repeat_rate"],
        "p95_latency_<=25ms": s["latency_ms"]["p95"] <= g["max_latency_p95_ms"],
        "calm_speech_like_<=0.20": res["speech_like_share"]["calm_requests"] <= g["max_calm_speech_like_share"],
    }
    return {k: ("NOT ESTABLISHED" if v is None else "PASS" if v else "FAIL") for k, v in status.items()}


def all_pass(gates):
    return all(v == "PASS" for v in gates.values())


# ---- cold start: fresh processes ----

def cold_start_probe(name):
    """Runs in a fresh interpreter: load the catalog (not timed), then time the strategy's
    first, second and tenth single-target call (sad / match)."""
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    targets = ev.request_targets("sad", "match")
    select = strategies()[name]
    times = []
    for i in range(10):
        np.random.seed(i)
        t0 = time.perf_counter()
        div.coverage_top_selector(catalog, targets, ev.LIST_SIZE, div.POOL_MULTIPLIER, select, z)
        times.append((time.perf_counter() - t0) * 1000)
    print(json.dumps({"first_ms": times[0], "second_ms": times[1], "tenth_ms": times[9]}))


def cold_start(names):
    out = {}
    for name in names:
        rows = []
        for _ in range(COLD_START_PROCESSES):
            res = subprocess.run([sys.executable, os.path.abspath(__file__), "--cold-start-probe", name],
                                 capture_output=True, text=True, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
            rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
        out[name] = {k: {"median_ms": round(float(np.median([r[k] for r in rows])), 2),
                         "max_ms": round(float(np.max([r[k] for r in rows])), 2)}
                     for k in ("first_ms", "second_ms", "tenth_ms")}
        out[name]["processes"] = COLD_START_PROCESSES
    return out


def main():
    parser = argparse.ArgumentParser(description="Stratified-selection prototype (analysis only)")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--cold-start-probe", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold_start_probe:
        cold_start_probe(args.cold_start_probe)
        return

    t0 = time.perf_counter()
    catalog_hash = mp.file_sha256(MUSIC_CATALOG_278K_PATH)
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    names = list(strategies())
    rng = np.random.RandomState(0)

    exploratory = evaluate(catalog, z, args.runs, div.EXPLORATORY_SEED, names, rng)
    for name, res in exploratory.items():
        s = res["summary"]
        print(f"[exploratory] {name:>30}: st-div {s['single_target_mean_diversity']:.3f} sad "
              f"{res['requests']['sad / match']['diversity']:.3f} cov {s['catalog_unique_tracks']} rep "
              f"{s['mean_repeat_rate']:.3f} sig {s['mean_ranking_signal_of_served']:.4f} p95 "
              f"{s['latency_ms']['p95']} calm-speech {res['speech_like_share']['calm_requests']:.3f} "
              f"fails {[k for k, v in res['gates'].items() if v != 'PASS']}")
    chosen = next((n for n in SIMPLICITY_ORDER if all_pass(exploratory[n]["gates"])), None)
    print(f"Chosen on exploratory seeds: {chosen}")

    held_out = evaluate(catalog, z, args.runs, div.HELD_OUT_SEED, names, rng)
    for name, res in held_out.items():
        s = res["summary"]
        print(f"[held-out]    {name:>30}: st-div {s['single_target_mean_diversity']:.3f} sad "
              f"{res['requests']['sad / match']['diversity']:.3f} rep {s['mean_repeat_rate']:.3f} p95 "
              f"{s['latency_ms']['p95']} fails {[k for k, v in res['gates'].items() if v != 'PASS']}")

    cold = cold_start(names)
    print(f"cold start: {json.dumps(cold)}")

    if mp.file_sha256(MUSIC_CATALOG_278K_PATH) != catalog_hash:
        raise SystemExit("The catalog changed during the analysis -- this must never happen.")
    report = {
        "analysis_only": True,
        "evaluation_date": datetime.date.today().isoformat(),
        "production_configuration": {"pool_multiplier": div.POOL_MULTIPLIER, "weighting": "current"},
        "catalog": {"path": os.path.relpath(MUSIC_CATALOG_278K_PATH, os.path.dirname(MODELS_DIR)),
                    "sha256": catalog_hash, "rows": int(len(catalog))},
        "runs_per_request": args.runs, "list_size": ev.LIST_SIZE,
        "seeds": {"exploratory": f"{div.EXPLORATORY_SEED}..{div.EXPLORATORY_SEED + args.runs - 1}",
                  "held_out": f"{div.HELD_OUT_SEED}..{div.HELD_OUT_SEED + args.runs - 1}"},
        "strategy_definitions": {
            "A: production": "recommend._sample_best",
            "D: stratified k-means (2H.7)": "sklearn KMeans(k = share, n_init=1, seeded) on the pool; one "
                                            "production-weighted pick per cluster",
            "E: numpy k-means": f"numpy k-means++ seeding + Lloyd until stable (max {LLOYD_MAX_ITER}); one "
                                "production-weighted pick per cluster; empty clusters filled by production-weighted "
                                "picks from the rest of the pool",
            "F: PC1 quantile strata": "equal-count strata along the pool's first principal component; one "
                                      "production-weighted pick per stratum",
            "shared": "same eligibility, exact shares, top pool_multiplier x share pool and interleaving as production",
        },
        "gates": GATES,
        "selection_rule": "simplest prototype (F, then E) passing every gate on the exploratory seeds; "
                          "held-out seeds only re-check the gates",
        "latency_definitions": {
            "harness p95 (gate)": "summary.latency_ms: every call in the 50-run harness, as in Phases 2H.6/2H.7",
            "first_call_in_run": "first call of the strategy in this already-warm process",
            "steady_state": "all harness calls except the first",
            "cold_start": f"fresh interpreter per measurement ({COLD_START_PROCESSES} processes), catalog loaded "
                          "first (untimed); first, second and tenth sad/match call",
        },
        "speech_like_definition": f"share of served tracks with speechiness > {SPEECH_LIKE}; calm_requests = mean "
                                  f"over {list(CALM_REQUESTS)} (both are calm-only requests)",
        "exploratory": exploratory,
        "chosen_on_exploratory": chosen,
        "held_out": held_out,
        "cold_start": cold,
        "notes": [
            "Diversity is feature-space diversity (mean pairwise distance in the 9 z-scored audio features), not "
            "perceived musical diversity.",
            "Precision/NDCG are omitted: relevance is the eligibility definition, so they are 1.0 for every strategy.",
            "Bootstrap intervals resample one request's runs; they describe sampling variability only.",
            "Latency is machine-dependent; strategies were run sequentially in one process.",
        ],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
