"""
ANALYSIS ONLY -- Phase 2I.1: pre-registered confirmation of confidence-threshold
candidate pools against a confidence-matched random baseline, on a fresh seed
set. Nothing here changes production: the app still calls
src.recommend.recommend_music.

Run from the project root:
    python evaluate_music_pool_confirmation.py [--output PATH]

Reads   data/music_catalog_278k.csv (read-only)
Writes  models/music_pool_confirmation.json (aggregate metrics only: no track
        IDs, titles, artists or catalog rows). No earlier report is touched.

Variants (implementations imported unchanged from evaluate_music_pool_redesign.py):
  A  production                         top 20 x share, no guard (reference)
  B  tau = 0.99 + calm guard            PRIMARY candidate
  C  tau = 0.95 + calm guard            secondary sensitivity analysis only
Everything in PREREGISTRATION and GATES was fixed before this script was first
run on seeds 21000-21049; the script records its own SHA-256 in the report.
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
import evaluate_music_phase_2h9 as h9
import evaluate_music_pool_redesign as pr
import evaluate_music_stratified_prototype as proto
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend

OUTPUT_PATH = os.path.join(MODELS_DIR, "music_pool_confirmation.json")
SEED = 21000
RUNS = 50
USED_SEEDS = {"tests and cold-start probes": (0, 49), "Phase 2H.6 API test": (2024, 2024),
              "exploratory (2H.5-2I)": (2026, 2075), "held-out (2H.7-2H.8)": (7000, 7049),
              "third set (2H.8.1)": (12000, 12049), "fourth set (2H.9)": (15000, 15049),
              "held-out (2I)": (18000, 18049)}
PRODUCTION = "A: production"
PRIMARY = "B: tau 0.99 + calm guard (primary)"
SENSITIVITY = "C: tau 0.95 + calm guard (sensitivity)"
VARIANTS = {
    PRODUCTION: (pr.STRATEGIES[pr.CONTROL], False),
    PRIMARY: (pr.STRATEGIES["C: threshold 0.99"], True),
    SENSITIVITY: (pr.STRATEGIES["C: threshold 0.95"], True),
}
PAIRED_REQUESTS = ("happy / match", "sad / match", "energetic / match", "stressed / match")
DERIVED = tuple(t for t in ("angry", "stressed"))

PREREGISTRATION = {
    "primary_strategy": "Phase 2I threshold pool, tau = 0.99, on base-label targets (sad, happy, energetic, calm) "
                        "only; selection, shares and interleaving unchanged; angry/stressed keep production logic; "
                        "calm candidates exclude speechiness > 0.66 (never relaxed); the existing never-shrink "
                        "fallback (production pool if fewer than 20 x share pass tau) and any shortfall are counted.",
    "sensitivity_strategy": "Same with tau = 0.95; reported against the same criteria; cannot replace the primary.",
    "confidence_matched_baseline": "Per request, the recommender's target shares filled uniformly at random "
                                   "(src.music_evaluation.random_eligible_list, RandomState(21000), same draw order as "
                                   "earlier phases, 50 lists) from each target's population: for base-label targets, "
                                   "exactly the strategy's candidate pool (guarded eligibility, signal >= tau, with the "
                                   "same never-shrink fallback); for angry/stressed, their full current eligibility "
                                   "(no probability floor is invented). It measures the diversity attainable among "
                                   "candidates meeting the same confidence requirement, not the maximum diversity of "
                                   "all eligible tracks. Unrestricted random-eligible (production and guarded "
                                   "eligibility) and production results are secondary references.",
    "calm_scoring": "Calm-only configuration counted once (chill/match = stressed/uplift); both aliases reported; "
                    "calm compared with unguarded production on identical seeds; speech-like spread is never "
                    "counted as an improvement.",
    "paired_gate": "Lower 95% bootstrap bound > 0 versus production for happy/sad/energetic/stressed match. A "
                   "component is NOT ESTABLISHED when the strategy leaves that target's selection identical to "
                   "production (stressed: derived logic unchanged by pre-registration), since the comparison is "
                   "then between identical algorithms. Gate = FAIL if any component fails, else NOT ESTABLISHED if "
                   "any is, else PASS.",
    "signal_gate": "Mean served classifier probability over every base-label-target slot in all 12 requests >= "
                   "0.99. The earlier all-target ranking-signal mean (which mixes in derived scores) is also "
                   "reported, with mean, 5th percentile and minimum per base-label target. A mean pass is not a "
                   "per-track floor.",
    "old_results": "Phase 2H.9 and 2I results and gate statuses stand as recorded; the baseline definition here is "
                   "new and is not applied to them.",
}

GATES = {
    "target_consistency": 1.0,
    "exact_target_shares_all_lists": True,
    "all_same_seed_identical": True,
    "min_unique_single_target_mean_diversity": 3.3,
    "min_sad_match_diversity": 2.0,
    "min_unique_configs_at_least_matched_baseline": 5,       # of 6
    "paired_lower_bound_positive_for": list(PAIRED_REQUESTS),
    "min_mean_served_classifier_probability": 0.99,
    "min_catalog_unique_tracks": 2000,
    "max_mean_repeat_rate": 0.75,
    "max_latency_p95_ms": 25.0,
    "max_calm_speech_like_share": 0.20,
    "calm_speech_like_not_worse_than_production": True,
}


def seed_range():
    return SEED, SEED + RUNS - 1


def seeds_disjoint():
    lo, hi = seed_range()
    return all(hi < a or lo > b for a, b in USED_SEEDS.values())


def fn(catalog, name, counters=None):
    spec, guarded = VARIANTS[name]
    return lambda targets: pr.coverage_top_pool(catalog, targets, ev.LIST_SIZE, spec, guarded, counters)


# ---- baselines ----

def baseline_population(catalog, target, share, spec, guarded):
    """Boolean mask of the population a baseline draws from for one target."""
    member, signal = (guard.guarded_candidates if guarded else recommend._target_candidates)(catalog, target)
    if target in DERIVED or spec["kind"] != "threshold":
        return member
    pool, _ = pr.pool_rows(target, signal, member, share, spec)
    mask = np.zeros(len(catalog), dtype=bool)
    mask[pool] = True
    return mask


def matched_baseline(catalog, z, seed, runs, spec, guarded):
    out = {}
    for mood in MOODS:
        for mode in ev.MODES:
            targets = ev.request_targets(mood, mode)
            weights = [w for _, w in targets]
            shares = recommend._allocate_shares(weights, ev.LIST_SIZE)
            members = [baseline_population(catalog, t, s, spec, guarded) for (t, _), s in zip(targets, shares)]
            rng = np.random.RandomState(seed)
            _ = [ev.random_catalog_list(len(catalog), ev.LIST_SIZE, rng) for _ in range(runs)]
            lists = [ev.random_eligible_list(members, weights, ev.LIST_SIZE, rng) for _ in range(runs)]
            out[f"{mood} / {mode}"] = round(float(np.mean([ev.intra_list_diversity(z[l]) for l in lists])), 4)
    return out


# ---- served confidence ----

def served_confidence(catalog, lists):
    """Per base-label target: classifier probability of every served slot of that target, all requests."""
    labels = catalog["mood"].to_numpy()
    per_target, pooled = {}, []
    for target in pr.LABEL_TARGETS:
        _, proba = recommend._target_candidates(catalog, target)
        values = []
        for name, runs in lists.items():
            mood, mode = name.split(" / ")
            if target in [t for t, _ in ev.request_targets(mood, mode)]:
                rows = np.concatenate(runs)
                values.append(proba[rows[labels[rows] == target]])
        v = np.concatenate(values)
        pooled.append(v)
        per_target[target] = {"slots": int(len(v)), "mean": round(float(v.mean()), 4),
                              "p5": round(float(np.percentile(v, 5)), 4), "min": round(float(v.min()), 4),
                              "share_below_0.99": round(float((v < 0.99).mean()), 4)}
    allv = np.concatenate(pooled)
    return {"by_target": per_target, "mean_all_base_label_slots": round(float(allv.mean()), 4),
            "share_of_slots_below_0.99": round(float((allv < 0.99).mean()), 4)}


# ---- scoring ----

def paired_component(name, request, paired):
    target = request.split(" / ")[0]
    if name != PRODUCTION and target in DERIVED:
        return "NOT ESTABLISHED", "selection for this derived target is identical to production by pre-registration"
    lb = paired[request]["bootstrap_95"][0]
    return ("PASS" if lb > 0 else "FAIL"), f"lower bound {lb}"


def combine(statuses):
    if "FAIL" in statuses:
        return "FAIL"
    if "NOT ESTABLISHED" in statuses:
        return "NOT ESTABLISHED"
    return "PASS"


def score(name, res, matched, configs, paired, prod_speech):
    g = GATES
    per_config = {}
    for config, names in configs.items():
        d = res["requests"][names[0]]["diversity"]
        per_config[" + ".join(f"{t}:{s}" for t, s in config)] = {
            "requests": names, "diversity": d,
            "aliases_identical": len({res["requests"][n]["diversity"] for n in names}) == 1,
            "confidence_matched_baseline": matched[names[0]], "at_least_baseline": d >= matched[names[0]]}
    mean_div = round(float(np.mean([v["diversity"] for v in per_config.values()])), 4)
    at_least = sum(v["at_least_baseline"] for v in per_config.values())
    if paired is None:
        paired_detail = {r: ("NOT ESTABLISHED", "production is the reference") for r in PAIRED_REQUESTS}
    else:
        paired_detail = {r: paired_component(name, r, paired) for r in PAIRED_REQUESTS}
    s = res["summary"]
    speech = res["speech_like_share"]["calm_requests"]
    status = {
        "target_consistency_100": "PASS" if s["target_consistency"] >= g["target_consistency"] else "FAIL",
        "exact_target_shares": "PASS" if s["exact_target_shares_all_lists"] else "FAIL",
        "deterministic": "PASS" if s["all_same_seed_identical"] else "FAIL",
        "unique_config_mean_diversity_>=3.3":
            "PASS" if mean_div >= g["min_unique_single_target_mean_diversity"] else "FAIL",
        "sad_match_diversity_>=2.0":
            "PASS" if res["requests"]["sad / match"]["diversity"] >= g["min_sad_match_diversity"] else "FAIL",
        ">=5_of_6_at_least_confidence_matched_baseline":
            "PASS" if at_least >= g["min_unique_configs_at_least_matched_baseline"] else "FAIL",
        "paired_lower_bounds_>0": combine([v[0] for v in paired_detail.values()]),
        "mean_served_classifier_probability_>=0.99":
            "PASS" if res["served_confidence"]["mean_all_base_label_slots"] >= g["min_mean_served_classifier_probability"]
            else "FAIL",
        "coverage_>=2000": "PASS" if s["catalog_unique_tracks"] >= g["min_catalog_unique_tracks"] else "FAIL",
        "repeat_rate_<=0.75": "PASS" if s["mean_repeat_rate"] <= g["max_mean_repeat_rate"] else "FAIL",
        "p95_latency_<=25ms": "PASS" if s["latency_ms"]["p95"] <= g["max_latency_p95_ms"] else "FAIL",
        "calm_speech_like_<=0.20_and_not_worse_than_production":
            "PASS" if speech <= g["max_calm_speech_like_share"] and speech <= prod_speech else "FAIL",
    }
    return {"unique_single_target_configurations": per_config,
            "unique_single_target_mean_diversity": mean_div,
            "unique_configs_at_least_confidence_matched_baseline": f"{at_least}/{len(per_config)}",
            "paired_gate_components": {r: {"status": v[0], "why": v[1]} for r, v in paired_detail.items()},
            "calm_speech_like_vs_production": {"variant": speech, "production_same_seeds": prod_speech},
            "gates": status}


def evaluate(catalog, z):
    configs = h9.unique_single_target_configurations()
    single = [n for ns in configs.values() for n in ns]
    unrestricted = {"production_eligibility": h9.random_eligible_baseline(catalog, z, SEED, RUNS,
                                                                          recommend._target_candidates),
                    "guarded_calm_eligibility": h9.random_eligible_baseline(catalog, z, SEED, RUNS,
                                                                            guard.guarded_candidates)}
    reference = {"requests": {n: {"diversity": d} for n, d in unrestricted["production_eligibility"].items()}}
    matched = {name: matched_baseline(catalog, z, SEED, RUNS, *VARIANTS[name]) for name in VARIANTS}
    results, per_runs = {}, {}
    for name in VARIANTS:
        counters = {}
        cfg, per_runs[name], lists, lat = guard.run(catalog, z, fn(catalog, name, counters), RUNS, SEED)
        res = div.summarize(cfg, reference, single)
        res["latency"] = proto.latency_summary(lat)
        res["speech_like_share"] = proto.speech_like(catalog, lists)
        res["calm_containing_requests"] = guard.guard_report(catalog, lists)
        res["served_confidence"] = served_confidence(catalog, lists)
        res["fallbacks"] = counters.get("fallbacks", {})
        res["shortfall_slots"] = counters.get("shortfall", {})
        res["pool_size_by_target"] = {t: {"min": int(min(v)), "median": int(np.median(v)), "max": int(max(v))}
                                      for t, v in counters.get("pool_sizes", {}).items()}
        for n, req in res["requests"].items():
            req["confidence_matched_baseline"] = matched[name][n]
            req["unrestricted_baseline_production_eligibility"] = unrestricted["production_eligibility"][n]
            req["unrestricted_baseline_guarded_calm"] = unrestricted["guarded_calm_eligibility"][n]
            req["production_diversity_same_seeds"] = None
        results[name] = res
    rng = np.random.RandomState(0)
    prod_speech = results[PRODUCTION]["speech_like_share"]["calm_requests"]
    for name in VARIANTS:
        paired = None
        for n, req in results[name]["requests"].items():
            req["production_diversity_same_seeds"] = results[PRODUCTION]["requests"][n]["diversity"]
        if name != PRODUCTION:
            paired = div.paired(per_runs[PRODUCTION], per_runs[name], list(results[name]["requests"]), rng)
            results[name]["paired_vs_production"] = paired
        results[name].update(score(name, results[name], matched[name], configs, paired, prod_speech))
    return results, matched, unrestricted, configs


def eligibility(catalog):
    sp = catalog["speechiness"].to_numpy(dtype=float)
    out = {}
    for target in list(pr.LABEL_TARGETS) + list(DERIVED):
        member, signal = recommend._target_candidates(catalog, target)
        gm, _ = guard.guarded_candidates(catalog, target)
        row = {"eligible": int(member.sum()), "eligible_after_calm_guard": int(gm.sum())}
        if target in pr.LABEL_TARGETS:
            for tau in (0.99, 0.95):
                row[f"guarded_eligible_with_signal_>={tau}"] = int((gm & (signal >= tau)).sum())
        if target == "calm":
            row["speech_like_removed"] = int((member & (sp > guard.SPEECH_LIKE)).sum())
        out[target] = row
    return out


def main():
    parser = argparse.ArgumentParser(description="Phase 2I.1 pre-registered confirmation (analysis only)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    if not seeds_disjoint():
        raise SystemExit("Seed set overlaps seeds used earlier")

    t0 = time.perf_counter()
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    catalog_hash = mp.file_sha256(MUSIC_CATALOG_278K_PATH)
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(SEED)
        a = recommend._coverage_top(catalog, targets, ev.LIST_SIZE).index.to_numpy()
        np.random.seed(SEED)
        if not np.array_equal(a, fn(catalog, PRODUCTION)(targets)):
            raise SystemExit(f"Production replica diverges for {mood}/{mode}")

    results, matched, unrestricted, configs = evaluate(catalog, z)
    for name, r in results.items():
        s = r["summary"]
        print(f"{name:>40}: uniq {r['unique_single_target_mean_diversity']:.3f} >=matched "
              f"{r['unique_configs_at_least_confidence_matched_baseline']} happy {r['requests']['happy / match']['diversity']:.3f} "
              f"sad {r['requests']['sad / match']['diversity']:.3f} cov {s['catalog_unique_tracks']} rep "
              f"{s['mean_repeat_rate']:.3f} conf {r['served_confidence']['mean_all_base_label_slots']} p95 "
              f"{s['latency_ms']['p95']} speech {r['speech_like_share']['calm_requests']:.3f} "
              f"not-PASS {[k for k, v in r['gates'].items() if v != 'PASS']}")
    cold = pr.cold_start([pr.CONTROL, "C: threshold 0.99 + calm guard", "C: threshold 0.95 + calm guard"])

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
        "seeds": {"set": f"{lo}..{hi}", "runs_per_request": RUNS,
                  "previously_used": {k: f"{a}..{b}" for k, (a, b) in USED_SEEDS.items()},
                  "disjoint_from_previously_used": seeds_disjoint(), "bootstrap_rng": 0},
        "variants": {k: {"pool": v[0], "calm_guard": v[1]} for k, v in VARIANTS.items()},
        "unique_single_target_configurations": {" + ".join(f"{t}:{s}" for t, s in c): n for c, n in configs.items()},
        "eligibility": eligibility(catalog),
        "baselines": {"confidence_matched": matched, "unrestricted": unrestricted},
        "results": results,
        "cold_start": {"A: production": cold[pr.CONTROL], PRIMARY: cold["C: threshold 0.99 + calm guard"],
                       SENSITIVITY: cold["C: threshold 0.95 + calm guard"]},
        "notes": [
            "Diversity is feature-space diversity (mean pairwise distance in the 9 z-scored audio features), not "
            "listener-perceived variety.",
            "Bootstrap intervals resample one request's 50 runs on one catalog: sampling variability only, no "
            "listener-level or cross-user significance.",
            "The confidence-matched baseline is new in this phase; Phase 2H.9 and 2I gate statuses are unchanged.",
            "Precision/NDCG are omitted: relevance is the eligibility definition.",
            "Latency is machine-dependent; cold start uses fresh interpreters on sad/match.",
        ],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
