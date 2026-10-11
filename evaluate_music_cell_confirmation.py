"""
ANALYSIS ONLY -- Phase 2J.1: confirm the base-label-only feature-cell strategy.
Nothing here changes production: the app still calls
src.recommend.recommend_music (pool_multiplier=20).

Run from the project root:
    python evaluate_music_cell_confirmation.py [--output PATH]

Reads   data/music_catalog_278k.csv (read-only)
Writes  models/music_cell_confirmation.json (aggregate metrics only: no track
        IDs, titles, artists, names or URIs). No earlier report is touched.

Hypothesis: applying Phase 2J's fixed feature-cell sampling only to base-label
single-target requests (happy, sad, calm, energetic) improves within-mood
diversity while avoiding Phase 2J's angry/stressed regression and uplift
whole-list diversity loss.

Strategies (implementations imported unchanged from evaluate_music_balanced_quality.py):
  P  production
  G  production + calm speech guard (reference)
  T  PRIMARY: Phase 2J strategy S for single-target requests whose target is happy,
     sad, calm or energetic; G, unchanged, for angry and stressed single-target
     requests and for every multi-target uplift request.
No other cell counts, thresholds, pool sizes or variants are tested.
PREREGISTRATION and GATES were fixed, and the tests written, before the first run;
the script records its own SHA-256 and the source-artifact hashes before running.
"""
import argparse
import datetime
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_music_balanced_quality as bq
import evaluate_music_diversity_analysis as div
import evaluate_music_recommender_analysis as sweep
import evaluate_music_stratified_prototype as proto
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend

OUTPUT_PATH = os.path.join(MODELS_DIR, "music_cell_confirmation.json")
RUNS = 50
PRIMARY_SEED = 30000
REPLICATION_SEED = 33000
USED_SEEDS = {**bq.USED_SEEDS, "primary (2J)": (24000, 24049), "replication (2J)": (27000, 27049)}
CELL_TARGETS = ("happy", "sad", "calm", "energetic")
P, G, T = "P: production", "G: production + calm guard", "T: cells for base-label single-target, else G"
VARIANTS = (P, G, T)
REFERENCE = G
GATES = bq.GATES        # Phase 2J gates, unchanged (asserted equal to the brief in tests)
COLD_START_PROCESSES = 5

PREREGISTRATION = {
    "routing": "T uses Phase 2J strategy S (30 fixed PCA feature cells per target, one production-weighted pick "
               "among each chosen cell's 20 most confident available tracks, calm guard) only when the request "
               "has exactly one target and it is happy, sad, calm or energetic; every other request is served "
               "by G unchanged.",
    "reference": "G (production + calm guard). Unguarded production is reported alongside.",
    "gates_resolved": [
        "Single-target gates use within-target diversity (equal to list diversity for one target).",
        "Calm speech-like share must be <= 0.20 and <= unguarded production's share on the same seeds.",
        "Configurations where T routes to G (angry, stressed, all five multi-target mixes) have paired "
        "differences of exactly 0: they count as 'lower bound not > 0' for the 4-of-6 count and pass the "
        "no-regression and non-inferiority gates.",
        "The legacy mean classifier probability >= 0.99 is reported, not applied.",
        "Within-target diversity of multi-target requests is reported as a diagnostic only.",
    ],
    "seeds": "Gates are decided on 30000-30049. 33000-33049 is a replication set: it is reported, and any gate it "
             "fails is flagged as not replicated. Both are disjoint from every earlier seed set.",
    "uncertainty": "Paired per-seed differences against G; 2,000-sample percentile bootstrap over 50 runs "
                   "(family means bootstrap the per-run family mean). Sampling variability on one catalog only.",
}


def seeds_disjoint():
    sets = [(PRIMARY_SEED, PRIMARY_SEED + RUNS - 1), (REPLICATION_SEED, REPLICATION_SEED + RUNS - 1)]
    return all(hi < a or lo > b for lo, hi in sets for a, b in USED_SEEDS.values()) and sets[0][1] < sets[1][0]


def uses_cells(targets):
    return len(targets) == 1 and targets[0][0] in CELL_TARGETS


def recommend_rows(catalog, targets, variant, cells, counters=None):
    if variant == T:
        return bq.recommend_rows(catalog, targets, bq.S if uses_cells(targets) else bq.G, cells, counters)
    return bq.recommend_rows(catalog, targets, bq.P if variant == P else bq.G, cells, counters)


def run_variant(catalog, z, variant, cells, seed):
    record, latencies, counters = {}, [], {}

    def recommender(targets, s):
        t0 = time.perf_counter()
        rows = recommend_rows(catalog, targets, variant, cells, counters)
        latencies.append((time.perf_counter() - t0) * 1000)
        record.setdefault((tuple(t for t, _ in targets), s), rows)
        return rows

    cfg = sweep.run_config(catalog, z, recommender, div.POOL_MULTIPLIER, RUNS, seed)
    lists = {}
    for name in cfg["requests"]:
        mood, mode = name.split(" / ")
        key = tuple(t for t, _ in ev.request_targets(mood, mode))
        lists[name] = [record[(key, seed + r)] for r in range(RUNS)]
    return cfg, lists, latencies, counters


def evaluate(catalog, z, seed, cells):
    single, multi = bq.families()
    reference = sweep.random_eligible_reference(catalog, z, RUNS, seed)
    results, metrics = {}, {}
    for variant in VARIANTS:
        cfg, lists, lat, shortfall = run_variant(catalog, z, variant, cells, seed)
        summ = div.summarize(cfg, reference, list(cfg["requests"]))["summary"]
        metrics[variant] = bq.per_run_metrics(catalog, z, lists)
        res = {"summary": summ, "latency": proto.latency_summary(lat),
               "speech_like_share": proto.speech_like(catalog, lists),
               "relevance": bq.relevance(catalog, lists), "shortfall_slots": shortfall, "configurations": {}}
        for fam_name, fam in (("single", single), ("multi", multi)):
            for config, names in fam.items():
                n = names[0]
                req, m = cfg["requests"][n], metrics[variant][n]
                res["configurations"][bq.config_name(config)] = {
                    "family": fam_name, "requests": names,
                    "routed_to_cells": variant == T and uses_cells(list(config)),
                    "aliases_identical": all(np.array_equal(a, b) for a, b in zip(lists[names[0]], lists[names[-1]])),
                    "list_diversity": round(float(np.mean(m["list"])), 4),
                    "within_target_diversity": round(float(np.mean(m["within"])), 4),
                    "between_target_separation": (round(float(np.mean(m["between"])), 4) if m["between"] else None),
                    "random_eligible_list_diversity": reference["requests"][n]["diversity"],
                    "typicality_by_target": {t: bq.typicality(catalog, z, lists[n], t) for t, _ in config},
                    "unique_tracks_over_runs": req["unique_tracks_over_runs"], "repeat_rate": req["repeat_rate"],
                    "target_consistency": req["target_consistency"],
                    "exact_target_shares_fraction": req["exact_target_shares_fraction"],
                }
        results[variant] = res
    rng = np.random.RandomState(0)
    for variant in VARIANTS:
        if variant == REFERENCE:
            continue
        paired = {}
        for config, names in list(single.items()) + list(multi.items()):
            a, b = metrics[REFERENCE][names[0]], metrics[variant][names[0]]
            paired[bq.config_name(config)] = {
                "list_diversity": bq.boot(np.subtract(b["list"], a["list"]), rng),
                "within_target_diversity": bq.boot(np.subtract(b["within"], a["within"]), rng)}
        fam_mean = {}
        for fam_name, fam, key in (("single", single, "within"), ("multi", multi, "list"),
                                   ("multi_within_diagnostic", multi, "within")):
            per_run = np.mean([np.subtract(metrics[variant][ns[0]][key], metrics[REFERENCE][ns[0]][key])
                               for ns in fam.values()], axis=0)
            fam_mean[fam_name] = bq.boot(per_run, rng)
        results[variant]["paired_vs_reference"] = paired
        results[variant]["family_mean_change_vs_reference"] = fam_mean
    prod_speech = results[P]["speech_like_share"]["calm_requests"]
    for variant in VARIANTS:
        results[variant]["gates"] = (bq.gate_status(results[variant], prod_speech, single, multi)
                                     if variant != REFERENCE else None)
    return results


# ---- cold start ----

def cold_start_probe(variant):
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    t0 = time.perf_counter()
    cells = bq.build_cells(catalog, z, guarded=True) if variant == T else None
    build_ms = (time.perf_counter() - t0) * 1000
    targets = ev.request_targets("sad", "match")
    times = []
    for i in range(10):
        np.random.seed(i)
        t1 = time.perf_counter()
        recommend_rows(catalog, targets, variant, cells)
        times.append((time.perf_counter() - t1) * 1000)
    print(json.dumps({"cell_build_ms": build_ms, "first_ms": times[0], "second_ms": times[1], "tenth_ms": times[9]}))


def cold_start():
    out = {}
    for variant in VARIANTS:
        rows = []
        for _ in range(COLD_START_PROCESSES):
            res = subprocess.run([sys.executable, os.path.abspath(__file__), "--cold-start-probe", variant],
                                 capture_output=True, text=True, check=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__)))
            rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
        out[variant] = {k: round(float(np.median([r[k] for r in rows])), 2)
                        for k in ("cell_build_ms", "first_ms", "second_ms", "tenth_ms")}
    return out


def main():
    parser = argparse.ArgumentParser(description="Phase 2J.1 base-label cell confirmation (analysis only)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--cold-start-probe", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold_start_probe:
        cold_start_probe(args.cold_start_probe)
        return
    if not seeds_disjoint():
        raise SystemExit("Seed sets overlap seeds used earlier")

    t0 = time.perf_counter()
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    hashes_before = {"catalog": mp.file_sha256(MUSIC_CATALOG_278K_PATH),
                     "classifier": mp.file_sha256(bq.CLASSIFIER_PATH)}
    print(f"script sha256 {script_sha}; source artifacts {hashes_before}")
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(PRIMARY_SEED)
        a = recommend._coverage_top(catalog, targets, ev.LIST_SIZE).index.to_numpy()
        np.random.seed(PRIMARY_SEED)
        if not np.array_equal(a, recommend_rows(catalog, targets, P, None)):
            raise SystemExit(f"Production replica diverges for {mood}/{mode}")
    cells = bq.build_cells(catalog, z, guarded=True)

    primary = evaluate(catalog, z, PRIMARY_SEED, cells)
    replication = evaluate(catalog, z, REPLICATION_SEED, cells)
    replicated = {k: (v == replication[T]["gates"][k]) for k, v in primary[T]["gates"].items()}
    for tag, res in (("primary", primary), ("replication", replication)):
        for v, r in res.items():
            fm = r.get("family_mean_change_vs_reference") or {}
            print(f"[{tag}] {v:>50}: single {fm.get('single')} multi {fm.get('multi')} "
                  f"not-PASS {[k for k, x in (r['gates'] or {}).items() if x != 'PASS']}")
    cold = cold_start()
    hashes_after = {"catalog": mp.file_sha256(MUSIC_CATALOG_278K_PATH),
                    "classifier": mp.file_sha256(bq.CLASSIFIER_PATH)}
    if hashes_after != hashes_before:
        raise SystemExit("A source artifact changed during the analysis -- this must never happen.")

    single, multi = bq.families()
    report = {
        "analysis_only": True,
        "evaluation_date": datetime.date.today().isoformat(),
        "script_sha256": script_sha,
        "phase_2j_script_sha256": mp.file_sha256(bq.__file__),
        "source_artifact_sha256": hashes_before,
        "production_configuration": {"pool_multiplier": div.POOL_MULTIPLIER, "weighting": "current"},
        "hypothesis": __doc__.split("Hypothesis: ")[1].split("\n\n")[0].replace("\n", " "),
        "preregistration": PREREGISTRATION,
        "gates": GATES,
        "seeds": {"primary": f"{PRIMARY_SEED}..{PRIMARY_SEED + RUNS - 1}",
                  "replication": f"{REPLICATION_SEED}..{REPLICATION_SEED + RUNS - 1}",
                  "previously_used": {k: f"{a}..{b}" for k, (a, b) in USED_SEEDS.items()},
                  "disjoint": seeds_disjoint(), "runs_per_request": RUNS, "bootstrap_rng": 0},
        "configurations": {"single_target": {bq.config_name(c): n for c, n in single.items()},
                           "multi_target": {bq.config_name(c): n for c, n in multi.items()}},
        "routing": {bq.config_name(c): ("cells" if uses_cells(list(c)) else "G") for c in list(single) + list(multi)},
        "measurement_validity": bq.MEASUREMENT_VALIDITY,
        "primary": primary,
        "replication": replication,
        "gate_decision_replicated": replicated,
        "cold_start_median_of_5_processes": cold,
        "notes": [
            "Diversity is feature-space diversity, not listener-perceived variety.",
            "Bootstrap intervals describe sampling variability over 50 seeded runs on one catalog only.",
            "Latency is machine-dependent; T's cells are built once per catalog (cell_build_ms), not per request.",
        ],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
