"""
ANALYSIS ONLY -- Phase 2H.8.1: re-validate the numpy k-means stratified
prototype on a fresh third seed set, and test an experimental speechiness
guard on calm-target candidates. Nothing here changes production: the app still
calls src.recommend.recommend_music, and production eligibility is unchanged.

Run from the project root:
    python evaluate_music_calm_guard_analysis.py [--runs N] [--output PATH]

Reads   data/music_catalog_278k.csv (read-only)
Writes  models/music_calm_guard_analysis.json (aggregate metrics only: no track
        IDs, titles, artists or catalog rows). No earlier report is touched.

Variants (same requests, pools, target shares, interleaving and metrics):
  A   production   recommend._sample_best, via the Phase 2H.7 replica
  E1  numpy k-means (Phase 2H.8 prototype E, imported unchanged)
  E2  E1 with calm-target candidates restricted to speechiness <= 0.66 (the
      stressed rule's spoken-word guard). Other targets are not guarded.
      Fallback: if a calm target has fewer guarded candidates than slots, the
      guard is NOT relaxed; the shortfall goes through production's existing
      rule (unfilled slots pass to the request's other targets, in order), so
      lists stay all-eligible and deterministic but that request's target
      shares change. Every shortfall slot is counted and reported.
No parameters or thresholds are chosen here; the Phase 2H.8 gates are reused.
"""
import argparse
import datetime
import itertools
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
import evaluate_music_stratified_prototype as proto
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend

OUTPUT_PATH = os.path.join(MODELS_DIR, "music_calm_guard_analysis.json")
THIRD_SEED = 12000
GUARDED_TARGET = "calm"
SPEECH_LIKE = proto.SPEECH_LIKE   # 0.66
# Seeds already used by earlier phases (smoke runs were subsets of the first two ranges;
# tests and cold-start probes used small seeds 0..29).
USED_SEED_RANGES = {"2H.5-2H.8 exploratory": (2026, 2075), "2H.7-2H.8 held-out": (7000, 7049),
                    "tests / cold-start probes": (0, 29)}


def third_seed_range(runs):
    return THIRD_SEED, THIRD_SEED + runs - 1


def seeds_disjoint(runs):
    lo, hi = third_seed_range(runs)
    return all(hi < a or lo > b for a, b in USED_SEED_RANGES.values())


# ---- E2: guarded calm candidates ----

def guarded_candidates(df, target):
    """recommend._target_candidates, with speech-like tracks removed from calm only."""
    member, signal = recommend._target_candidates(df, target)
    if target == GUARDED_TARGET:
        member = member & ~(df["speechiness"].to_numpy(dtype=float) > SPEECH_LIKE)
    return member, signal


def coverage_top_guarded(df, targets, top_n, pool_multiplier, select, z, shortfall=None):
    """div.coverage_top_selector with guarded_candidates. `shortfall`, if given, is a dict that
    receives target -> slots its guarded candidates could not fill. Returns row positions."""
    shares = recommend._allocate_shares([w for _, w in targets], top_n)
    taken = np.zeros(len(df), dtype=bool)
    picks = []
    for (target, _), share in zip(targets, shares):
        member, signal = guarded_candidates(df, target)
        chosen = select(signal, member & ~taken, share, pool_multiplier, z)
        taken[chosen] = True
        if shortfall is not None and len(chosen) < share:
            shortfall[target] = shortfall.get(target, 0) + share - len(chosen)
        picks.append([target, chosen, member, signal])
    missing = top_n - int(taken.sum())
    for pick in picks:
        if missing <= 0:
            break
        target, chosen, member, signal = pick
        extra = select(signal, member & ~taken, missing, pool_multiplier, z)
        taken[extra] = True
        missing -= len(extra)
        pick[1] = div._best_first(np.concatenate([chosen, extra]), signal)
    slots = []
    for t_i, (target, chosen, _, signal) in enumerate(picks):
        for k, row in enumerate(chosen):
            slots.append(((k + 0.5) / len(chosen), t_i, row))
    slots.sort(key=lambda slot: slot[:2])
    return np.array([slot[2] for slot in slots], dtype=int)


def variants(catalog, z, shortfall):
    pm = div.POOL_MULTIPLIER
    return {
        "A: production": lambda t: div.coverage_top_selector(catalog, t, ev.LIST_SIZE, pm, div.select_production, z),
        "E1: numpy k-means": lambda t: div.coverage_top_selector(catalog, t, ev.LIST_SIZE, pm,
                                                                 proto.numpy_kmeans_select, z),
        "E2: numpy k-means + calm speech guard": lambda t: coverage_top_guarded(
            catalog, t, ev.LIST_SIZE, pm, proto.numpy_kmeans_select, z, shortfall),
    }


def calm_requests():
    return [f"{mood} / {mode}" for mood in MOODS for mode in ev.MODES
            if any(t == GUARDED_TARGET for t, _ in ev.request_targets(mood, mode))]


# ---- evaluation ----

def run(catalog, z, recommend_fn, runs, seed):
    record, latencies = {}, []

    def recommender(targets, s):
        t0 = time.perf_counter()
        rows = recommend_fn(targets)
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


def guard_report(catalog, lists):
    """Per calm-containing request: speech-like share of all served tracks and of the calm slots."""
    sp = catalog["speechiness"].to_numpy(dtype=float)
    calm_member, _ = recommend._target_candidates(catalog, GUARDED_TARGET)
    out = {}
    for name in calm_requests():
        rows = np.concatenate(lists[name])
        calm_rows = rows[calm_member[rows]]
        out[name] = {"speech_like_share_all_served": round(float((sp[rows] > SPEECH_LIKE).mean()), 4),
                     "speech_like_share_of_calm_slots": round(float((sp[calm_rows] > SPEECH_LIKE).mean()), 4)}
    return out


def evaluate(catalog, z, runs, seed, rng):
    single = list(div.single_target_requests())
    reference = sweep.random_eligible_reference(catalog, z, runs, seed)
    shortfall = {}
    results, per_runs = {}, {}
    for name, fn in variants(catalog, z, shortfall).items():
        shortfall.clear()
        cfg, per_runs[name], lists, lat = run(catalog, z, fn, runs, seed)
        res = div.summarize(cfg, reference, single)
        res["latency"] = proto.latency_summary(lat)
        res["speech_like_share"] = proto.speech_like(catalog, lists)
        res["calm_containing_requests"] = guard_report(catalog, lists)
        res["fallback_shortfall_slots"] = dict(shortfall)
        results[name] = res
    for name in results:
        if name != "A: production":
            results[name]["paired_vs_production"] = div.paired(per_runs["A: production"], per_runs[name], single, rng)
            results[name]["paired_vs_production_calm_requests"] = div.paired(
                per_runs["A: production"], per_runs[name], calm_requests(), rng)
        results[name]["gates"] = proto.gate_status(results[name])
        prod_speech = results["A: production"]["speech_like_share"]["calm_requests"]
        results[name]["calm_speech_like_vs_same_seed_production"] = {
            "variant": results[name]["speech_like_share"]["calm_requests"], "production": prod_speech,
            "passes_absolute_0.20_but_worse_than_production":
                results[name]["speech_like_share"]["calm_requests"] <= proto.GATES["max_calm_speech_like_share"]
                and results[name]["speech_like_share"]["calm_requests"] > prod_speech}
    return results, reference


def eligibility_counts(catalog):
    sp = catalog["speechiness"].to_numpy(dtype=float)
    out = {}
    for target in sorted({t for m in MOODS for md in ev.MODES for t, _ in ev.request_targets(m, md)}):
        before, signal = recommend._target_candidates(catalog, target)
        after, _ = guarded_candidates(catalog, target)
        top = before & (signal == signal[before].max())
        out[target] = {"eligible_before_guard": int(before.sum()), "eligible_after_guard": int(after.sum()),
                       "removed_by_guard": int((before & ~after).sum()),
                       "speech_like_among_eligible": int((before & (sp > SPEECH_LIKE)).sum()),
                       "tracks_at_max_signal": int(top.sum()),
                       "speech_like_at_max_signal": int((top & (sp > SPEECH_LIKE)).sum())}
    return out


# ---- cold start ----

def cold_start_probe(name):
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    fn = variants(catalog, z, {})[name]
    targets = ev.request_targets("chill", "match")     # calm-only request: the guard's path
    times = []
    for i in range(10):
        np.random.seed(i)
        t0 = time.perf_counter()
        fn(targets)
        times.append((time.perf_counter() - t0) * 1000)
    print(json.dumps({"first_ms": times[0], "second_ms": times[1], "tenth_ms": times[9]}))


def cold_start(names):
    out = {}
    for name in names:
        rows = []
        for _ in range(proto.COLD_START_PROCESSES):
            res = subprocess.run([sys.executable, os.path.abspath(__file__), "--cold-start-probe", name],
                                 capture_output=True, text=True, check=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__)))
            rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
        out[name] = {k: {"median_ms": round(float(np.median([r[k] for r in rows])), 2),
                         "max_ms": round(float(np.max([r[k] for r in rows])), 2)}
                     for k in ("first_ms", "second_ms", "tenth_ms")}
        out[name]["processes"] = proto.COLD_START_PROCESSES
    return out


def main():
    parser = argparse.ArgumentParser(description="Calm speechiness guard + third-seed validation (analysis only)")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--cold-start-probe", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold_start_probe:
        cold_start_probe(args.cold_start_probe)
        return
    if not seeds_disjoint(args.runs):
        raise SystemExit("Third seed set overlaps seeds used by earlier phases")

    t0 = time.perf_counter()
    catalog_hash = mp.file_sha256(MUSIC_CATALOG_278K_PATH)
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)

    # the production replica must reproduce production on the third seeds
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(THIRD_SEED)
        a = recommend._coverage_top(catalog, targets, ev.LIST_SIZE).index.to_numpy()
        np.random.seed(THIRD_SEED)
        b = variants(catalog, z, {})["A: production"](targets)
        if not np.array_equal(a, b):
            raise SystemExit(f"Replica diverges from production for {mood}/{mode}")

    rng = np.random.RandomState(0)
    results, reference = evaluate(catalog, z, args.runs, THIRD_SEED, rng)
    for name, res in results.items():
        s = res["summary"]
        print(f"{name:>40}: st-div {s['single_target_mean_diversity']:.3f} sad "
              f"{res['requests']['sad / match']['diversity']:.3f} >=RE {s['single_target_requests_at_least_random_eligible']}/7 "
              f"cov {s['catalog_unique_tracks']} rep {s['mean_repeat_rate']:.4f} sig {s['mean_ranking_signal_of_served']:.4f} "
              f"p95 {s['latency_ms']['p95']} calm-speech {res['speech_like_share']['calm_requests']:.4f} "
              f"shortfall {res['fallback_shortfall_slots']} fails {[k for k, v in res['gates'].items() if v != 'PASS']}")
    cold = cold_start(list(results))
    print(f"cold start: {json.dumps(cold)}")

    if mp.file_sha256(MUSIC_CATALOG_278K_PATH) != catalog_hash:
        raise SystemExit("The catalog changed during the analysis -- this must never happen.")
    lo, hi = third_seed_range(args.runs)
    report = {
        "analysis_only": True,
        "evaluation_date": datetime.date.today().isoformat(),
        "production_configuration": {"pool_multiplier": div.POOL_MULTIPLIER, "weighting": "current"},
        "catalog": {"path": os.path.relpath(MUSIC_CATALOG_278K_PATH, os.path.dirname(MODELS_DIR)),
                    "sha256": catalog_hash, "rows": int(len(catalog))},
        "runs_per_request": args.runs, "list_size": ev.LIST_SIZE,
        "third_seed_set": {"range": f"{lo}..{hi}", "previously_used": {k: f"{a}..{b}" for k, (a, b) in
                                                                       USED_SEED_RANGES.items()},
                           "disjoint_from_previously_used": seeds_disjoint(args.runs)},
        "variants": {
            "A: production": "recommend._sample_best (production replica, verified identical)",
            "E1: numpy k-means": "Phase 2H.8 prototype E, unchanged (evaluate_music_stratified_prototype.py)",
            "E2: numpy k-means + calm speech guard": f"E1 with calm candidates restricted to speechiness <= "
                                                     f"{SPEECH_LIKE}; no other target guarded",
        },
        "guard": {"target": GUARDED_TARGET, "rule": f"exclude speechiness > {SPEECH_LIKE}",
                  "affected_requests": calm_requests(),
                  "fallback": "never relaxes the guard; a calm shortfall goes through production's existing "
                              "rule (other targets of the request, in order), changing that request's shares; "
                              "shortfall slots are counted per variant"},
        "eligibility_counts": eligibility_counts(catalog),
        "gates": proto.GATES,
        "random_eligible_reference": reference["summary"],
        "results": results,
        "cold_start": cold,
        "notes": [
            "Diversity is feature-space diversity (mean pairwise distance in the 9 z-scored audio features); it is "
            "not listener-perceived musical variety.",
            "Bootstrap intervals resample one request's runs on one catalog; they describe sampling variability "
            "only and establish nothing about users or listeners.",
            "The random-eligible baseline uses production (unguarded) eligibility for every variant.",
            "Precision/NDCG are omitted: relevance is the eligibility definition, so they cannot rank variants.",
            "Latency is machine-dependent; variants ran sequentially in one process.",
        ],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
