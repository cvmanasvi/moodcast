"""
ANALYSIS ONLY -- Phase 2H.9: pre-registered, calm-aware evaluation of the
stratified prototypes on a fourth, unseen seed set. Nothing here changes
production: the app still calls src.recommend.recommend_music.

Run from the project root:
    python evaluate_music_phase_2h9.py [--output PATH]

Reads   data/music_catalog_278k.csv (read-only)
Writes  models/music_phase_2h9_evaluation.json (aggregate metrics only: no
        track IDs, titles, artists or catalog rows). No earlier report is touched.

Variants (unchanged implementations, imported):
  A   production                          (Phase 2H.7 replica of recommend._coverage_top)
  E1  numpy k-means                       (Phase 2H.8, evaluate_music_stratified_prototype.py)
  E2  numpy k-means + calm speech guard   (Phase 2H.8.1, evaluate_music_calm_guard_analysis.py)

The scoring rules (PREREGISTRATION) and gates (GATES) below were fixed, and the
tests written, before this script was first run on seeds 15000-15049. The
script records its own SHA-256 in the report.
"""
import argparse
import datetime
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_music_calm_guard_analysis as guard
import evaluate_music_diversity_analysis as div
import evaluate_music_stratified_prototype as proto
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend

OUTPUT_PATH = os.path.join(MODELS_DIR, "music_phase_2h9_evaluation.json")
FOURTH_SEED = 15000
RUNS = 50
BOOTSTRAP_SEED = 0
# Every seed an earlier phase or test used (smoke runs were subsets of 2026.. and 7000..).
USED_SEEDS = {"tests and cold-start probes": (0, 49), "Phase 2H.6 API test": (2024, 2024),
              "exploratory (2H.5-2H.8)": (2026, 2075), "held-out (2H.7-2H.8)": (7000, 7049),
              "third set (2H.8.1)": (12000, 12049)}
VARIANTS = ("A: production", "E1: numpy k-means", "E2: numpy k-means + calm speech guard")
GUARDED_VARIANTS = ("E2: numpy k-means + calm speech guard",)

PREREGISTRATION = {
    "A_calm_eligibility": "E2 only: calm-target candidates exclude speechiness > 0.66; no other target is "
                          "guarded; the guard is never relaxed; exact shares whenever candidates suffice; any "
                          "shortfall is counted and reported (it passes slots to the request's other targets "
                          "by production's existing rule).",
    "B_random_eligible_baseline": "Per request, the recommender's per-target shares filled uniformly at random "
                                  "from each target's candidates (src.music_evaluation.random_eligible_list), "
                                  "RandomState(seed) with the same draw order as the earlier phases, seed = "
                                  "15000, 50 lists. A and E1 use production eligibility; E2 uses guarded calm "
                                  "eligibility (identical to production for every other target). The unguarded "
                                  "baseline is also reported for E2 as a reference only.",
    "C_unique_configurations": "Single-target requests are deduplicated by their underlying (target, share) "
                               "configuration; chill/match and stressed/uplift are both [(calm, 30)] and count "
                               "once. Aggregate single-target diversity and the baseline count use the 6 unique "
                               "configurations; per-request reporting keeps all 12 requests. The 5-of-7 gate "
                               "becomes 5 of 6.",
    "D_frozen_criteria": "All other Phase 2H.8 gates unchanged (see gates). Calm speech-like share is also "
                         "compared with production on identical seeds and flagged if worse.",
}

GATES = {
    "target_consistency": 1.0,
    "exact_target_shares_all_lists": True,
    "all_same_seed_identical": True,
    "min_unique_single_target_mean_diversity": 3.3,
    "min_sad_match_diversity": 2.0,
    "min_unique_single_target_configs_at_least_baseline": 5,      # of 6
    "paired_lower_bound_positive_for": ["happy / match", "sad / match", "energetic / match", "stressed / match"],
    "min_mean_ranking_signal": 0.99,
    "min_catalog_unique_tracks": 2000,
    "max_mean_repeat_rate": 0.75,
    "max_latency_p95_ms": 25.0,
    "max_calm_speech_like_share": 0.20,
}


# ---- seeds ----

def seed_range():
    return FOURTH_SEED, FOURTH_SEED + RUNS - 1


def seeds_disjoint():
    lo, hi = seed_range()
    return all(hi < a or lo > b for a, b in USED_SEEDS.values())


# ---- unique configurations ----

def request_configuration(mood, mode):
    targets = ev.request_targets(mood, mode)
    shares = recommend._allocate_shares([w for _, w in targets], ev.LIST_SIZE)
    return tuple((t, int(s)) for (t, _), s in zip(targets, shares))


def unique_single_target_configurations():
    """{configuration: [request names]} for single-target requests, in request order."""
    out = {}
    for mood in MOODS:
        for mode in ev.MODES:
            config = request_configuration(mood, mode)
            if len(config) == 1:
                out.setdefault(config, []).append(f"{mood} / {mode}")
    return out


# ---- baselines ----

def random_eligible_baseline(catalog, z, seed, runs, candidates):
    """Per-request random-eligible diversity with the given candidate function; same RNG scheme
    (and draw order) as evaluate_music_recommender_analysis.random_eligible_reference."""
    out = {}
    for mood in MOODS:
        for mode in ev.MODES:
            targets = ev.request_targets(mood, mode)
            members = [candidates(catalog, t)[0] for t, _ in targets]
            weights = [w for _, w in targets]
            rng = np.random.RandomState(seed)
            _ = [ev.random_catalog_list(len(catalog), ev.LIST_SIZE, rng) for _ in range(runs)]
            lists = [ev.random_eligible_list(members, weights, ev.LIST_SIZE, rng) for _ in range(runs)]
            out[f"{mood} / {mode}"] = round(float(np.mean([ev.intra_list_diversity(z[l]) for l in lists])), 4)
    return out


# ---- scoring ----

def score(res, baseline, configs, paired, prod_speech):
    """Aggregates over unique configurations and every gate as PASS / FAIL / NOT ESTABLISHED."""
    reps = [names[0] for names in configs.values()]
    per_config = {}
    for config, names in configs.items():
        divs = [res["requests"][n]["diversity"] for n in names]
        per_config[" + ".join(f"{t}:{s}" for t, s in config)] = {
            "requests": names, "diversity": divs[0], "aliases_identical": len(set(divs)) == 1,
            "baseline": baseline[names[0]], "at_least_baseline": divs[0] >= baseline[names[0]]}
    s, g = res["summary"], GATES
    mean_div = round(float(np.mean([res["requests"][n]["diversity"] for n in reps])), 4)
    at_least = sum(v["at_least_baseline"] for v in per_config.values())
    speech = res["speech_like_share"]["calm_requests"]
    checks = {
        "target_consistency_100": s["target_consistency"] >= g["target_consistency"],
        "exact_target_shares": s["exact_target_shares_all_lists"] is True,
        "deterministic": s["all_same_seed_identical"] is True,
        "unique_single_target_mean_diversity_>=3.3": mean_div >= g["min_unique_single_target_mean_diversity"],
        "sad_match_diversity_>=2.0": res["requests"]["sad / match"]["diversity"] >= g["min_sad_match_diversity"],
        ">=5_of_6_unique_configs_at_least_baseline": at_least >= g["min_unique_single_target_configs_at_least_baseline"],
        "paired_lower_bounds_>0": (None if paired is None else
                                   all(paired[r]["bootstrap_95"][0] > 0 for r in g["paired_lower_bound_positive_for"])),
        "ranking_signal_>=0.99": s["mean_ranking_signal_of_served"] >= g["min_mean_ranking_signal"],
        "coverage_>=2000": s["catalog_unique_tracks"] >= g["min_catalog_unique_tracks"],
        "repeat_rate_<=0.75": s["mean_repeat_rate"] <= g["max_mean_repeat_rate"],
        "p95_latency_<=25ms": s["latency_ms"]["p95"] <= g["max_latency_p95_ms"],
        "calm_speech_like_<=0.20": speech <= g["max_calm_speech_like_share"],
    }
    return {
        "unique_single_target_configurations": per_config,
        "unique_single_target_mean_diversity": mean_div,
        "unique_configs_at_least_baseline": f"{at_least}/{len(per_config)}",
        "calm_speech_like_vs_production": {"variant": speech, "production_same_seeds": prod_speech,
                                           "worse_than_production": speech > prod_speech},
        "gates": {k: ("NOT ESTABLISHED" if v is None else "PASS" if v else "FAIL") for k, v in checks.items()},
    }


def evaluate(catalog, z):
    configs = unique_single_target_configurations()
    single = [n for names in configs.values() for n in names]
    baselines = {"production_eligibility": random_eligible_baseline(catalog, z, FOURTH_SEED, RUNS,
                                                                    recommend._target_candidates),
                 "guarded_calm_eligibility": random_eligible_baseline(catalog, z, FOURTH_SEED, RUNS,
                                                                      guard.guarded_candidates)}
    reference = {"requests": {n: {"diversity": d} for n, d in baselines["production_eligibility"].items()}}
    shortfall = {}
    fns = guard.variants(catalog, z, shortfall)
    results, per_runs = {}, {}
    for name in VARIANTS:
        shortfall.clear()
        cfg, per_runs[name], lists, lat = guard.run(catalog, z, fns[name], RUNS, FOURTH_SEED)
        res = div.summarize(cfg, reference, single)
        res["latency"] = proto.latency_summary(lat)
        res["speech_like_share"] = proto.speech_like(catalog, lists)
        res["calm_containing_requests"] = guard.guard_report(catalog, lists)
        res["fallback_shortfall_slots"] = dict(shortfall)
        base_name = "guarded_calm_eligibility" if name in GUARDED_VARIANTS else "production_eligibility"
        res["applicable_baseline"] = base_name
        for n, req in res["requests"].items():
            req["applicable_random_eligible_diversity"] = baselines[base_name][n]
            req["diversity_minus_applicable_baseline"] = round(req["diversity"] - baselines[base_name][n], 4)
        results[name] = res
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    prod_speech = results["A: production"]["speech_like_share"]["calm_requests"]
    for name in VARIANTS:
        paired = None
        if name != "A: production":
            paired = div.paired(per_runs["A: production"], per_runs[name], list(results[name]["requests"]), rng)
            results[name]["paired_vs_production"] = paired
        base_name = results[name]["applicable_baseline"]
        results[name].update(score(results[name], baselines[base_name], configs, paired, prod_speech))
    return results, baselines, configs


def main():
    parser = argparse.ArgumentParser(description="Phase 2H.9 pre-registered evaluation (analysis only)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    if not seeds_disjoint():
        raise SystemExit("Fourth seed set overlaps seeds used earlier")

    t0 = time.perf_counter()
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    catalog_hash = mp.file_sha256(MUSIC_CATALOG_278K_PATH)
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)

    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(FOURTH_SEED)
        a = recommend._coverage_top(catalog, targets, ev.LIST_SIZE).index.to_numpy()
        np.random.seed(FOURTH_SEED)
        if not np.array_equal(a, guard.variants(catalog, z, {})["A: production"](targets)):
            raise SystemExit(f"Replica diverges from production for {mood}/{mode}")

    results, baselines, configs = evaluate(catalog, z)
    for name, res in results.items():
        s = res["summary"]
        print(f"{name:>40}: unique st-div {res['unique_single_target_mean_diversity']:.3f} "
              f">=base {res['unique_configs_at_least_baseline']} sad {res['requests']['sad / match']['diversity']:.3f} "
              f"cov {s['catalog_unique_tracks']} rep {s['mean_repeat_rate']:.4f} sig {s['mean_ranking_signal_of_served']:.4f} "
              f"p95 {s['latency_ms']['p95']} speech {res['speech_like_share']['calm_requests']:.4f} "
              f"shortfall {res['fallback_shortfall_slots']} fails {[k for k, v in res['gates'].items() if v != 'PASS']}")
    cold = guard.cold_start(list(VARIANTS))

    if mp.file_sha256(MUSIC_CATALOG_278K_PATH) != catalog_hash:
        raise SystemExit("The catalog changed during the analysis -- this must never happen.")
    lo, hi = seed_range()
    report = {
        "analysis_only": True,
        "evaluation_date": datetime.date.today().isoformat(),
        "script_sha256": script_sha,
        "production_configuration": {"pool_multiplier": div.POOL_MULTIPLIER, "weighting": "current"},
        "catalog": {"path": os.path.relpath(MUSIC_CATALOG_278K_PATH, os.path.dirname(MODELS_DIR)),
                    "sha256": catalog_hash, "rows": int(len(catalog))},
        "preregistration": PREREGISTRATION,
        "gates": GATES,
        "seeds": {"fourth_set": f"{lo}..{hi}", "runs_per_request": RUNS,
                  "previously_used": {k: f"{a}..{b}" for k, (a, b) in USED_SEEDS.items()},
                  "disjoint_from_previously_used": seeds_disjoint(), "bootstrap_rng": BOOTSTRAP_SEED},
        "unique_single_target_configurations": {" + ".join(f"{t}:{s}" for t, s in c): n for c, n in configs.items()},
        "eligibility_counts": guard.eligibility_counts(catalog),
        "random_eligible_baselines": baselines,
        "results": results,
        "cold_start": cold,
        "notes": [
            "Diversity is feature-space diversity (mean pairwise distance in the 9 z-scored audio features), not "
            "necessarily listener-perceived variety.",
            "Bootstrap intervals resample one request's 50 runs on one catalog: sampling variability only, no "
            "listener-level or cross-user significance.",
            "Precision/NDCG are omitted: relevance is the eligibility definition, so they cannot rank variants.",
            "Latency is machine-dependent; variants ran sequentially in one process; cold start uses fresh "
            "interpreters on the calm-only request.",
        ],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
