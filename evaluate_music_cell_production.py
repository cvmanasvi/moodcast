"""
ANALYSIS ONLY -- Phase 2K post-implementation check of the production
feature-cell routing in src/recommend.py against the previous production
ranking. It calls the production code; it changes nothing.

Run from the project root:
    python evaluate_music_cell_production.py [--output PATH]

Reads   data/music_catalog_278k.csv (read-only)
Writes  models/music_cell_production_check.json (aggregate metrics only: no
        track IDs, titles, artists, names or URIs). No earlier report is touched.

Variants:
  B  previous production    recommend._coverage_top for every request (what
                            recommend_music returned before Phase 2K) -- the reference
  N  new production         recommend.recommend_music as the API calls it: feature
                            cells for single-target happy/sad/calm/energetic, B otherwise
  V  validated T (diagnostic only) -- Phase 2J.1's T: guarded calm cells and the
                            calm-guarded production ranking elsewhere. Production has no
                            calm speech guard, so N and V differ on calm (cells built over
                            unguarded calm eligibility) and on the three uplift mixes that
                            include calm (unguarded ranking). V is not gated.
PREREGISTRATION and GATES were fixed before the first run on these seeds.
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
import evaluate_music_cell_confirmation as cc
import evaluate_music_diversity_analysis as div
import evaluate_music_recommender_analysis as sweep
import evaluate_music_stratified_prototype as proto
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend

OUTPUT_PATH = os.path.join(MODELS_DIR, "music_cell_production_check.json")
RUNS = 50
PRIMARY_SEED = 36000
REPLICATION_SEED = 39000
USED_SEEDS = {**cc.USED_SEEDS, "primary (2J.1)": (30000, 30049), "replication (2J.1)": (33000, 33049)}
B, N, V = ("B: previous production", "N: new production (recommend_music)",
           "V: validated T, guarded (diagnostic only)")
VARIANTS = (B, N, V)
REFERENCE = B
GATES = bq.GATES
COLD_START_PROCESSES = 5

PREREGISTRATION = {
    "reference": "B, the unchanged previous ranking (production has no calm speech guard), on identical seeds.",
    "gates": "The Phase 2J.1 gates, unchanged, applied to N against B. Calm: B's calm lists contain speech-like "
             "tracks whose spread inflates their diversity (Phase 2H.8.1); no adjustment is made for that, so the "
             "calm configuration is compared as-is. Calm speech-like share must be <= 0.20 and <= B's.",
    "resolved": [
        "Single-target gates use within-target diversity (equal to list diversity for one target).",
        "Configurations N serves exactly as B (angry, stressed, all five uplift mixes) have zero paired "
        "differences: not a gain for the 4-of-6 count, and they pass no-regression and non-inferiority.",
        "Legacy mean classifier probability >= 0.99 is reported, not applied.",
        "V is a diagnostic: reported against B but never gated.",
    ],
    "seeds": "Gates on 36000-36049; replication 39000-39049 reported, and a gate it fails is flagged as not "
             "replicated. Both disjoint from every earlier seed set.",
}


def seeds_disjoint():
    sets = [(PRIMARY_SEED, PRIMARY_SEED + RUNS - 1), (REPLICATION_SEED, REPLICATION_SEED + RUNS - 1)]
    return all(hi < a or lo > b for lo, hi in sets for a, b in USED_SEEDS.values()) and sets[0][1] < sets[1][0]


def request_for(targets):
    """An app (mood, mode) whose targets are `targets` (aliases share targets, so either works)."""
    for mood, mode in itertools.product(MOODS, ev.MODES):
        if ev.request_targets(mood, mode) == list(targets):
            return mood, ev.MODES[mode]
    raise KeyError(targets)


def recommender_for(variant, catalog, guarded_cells):
    if variant == B:
        return lambda targets: recommend._coverage_top(catalog, targets, ev.LIST_SIZE).index.to_numpy()
    if variant == N:
        return lambda targets: recommend.recommend_music(catalog, *request_for(targets),
                                                         top_n=ev.LIST_SIZE).index.to_numpy()
    return lambda targets: cc.recommend_rows(catalog, targets, cc.T, guarded_cells)


def evaluate(catalog, z, seed, guarded_cells):
    single, multi = bq.families()
    reference = sweep.random_eligible_reference(catalog, z, RUNS, seed)
    results, metrics = {}, {}
    for variant in VARIANTS:
        fn = recommender_for(variant, catalog, guarded_cells)
        record, latencies = {}, []

        def rec(targets, s, fn=fn, record=record, latencies=latencies):
            t0 = time.perf_counter()
            rows = fn(targets)
            latencies.append((time.perf_counter() - t0) * 1000)
            record.setdefault((tuple(t for t, _ in targets), s), rows)
            return rows

        cfg = sweep.run_config(catalog, z, rec, div.POOL_MULTIPLIER, RUNS, seed)
        lists = {}
        for name in cfg["requests"]:
            mood, mode = name.split(" / ")
            key = tuple(t for t, _ in ev.request_targets(mood, mode))
            lists[name] = [record[(key, seed + r)] for r in range(RUNS)]
        summ = div.summarize(cfg, reference, list(cfg["requests"]))["summary"]
        metrics[variant] = bq.per_run_metrics(catalog, z, lists)
        sp = catalog["speechiness"].to_numpy(dtype=float)
        res = {"summary": summ, "latency": proto.latency_summary(latencies),
               "speech_like_share": proto.speech_like(catalog, lists),
               "calm_slot_speech_like_share": {
                   n: round(float((sp[np.concatenate(lists[n])][recommend._target_candidates(catalog, "calm")[0][
                       np.concatenate(lists[n])]] > 0.66).mean()), 4)
                   for n in lists if "calm" in [t for t, _ in ev.request_targets(*n.split(" / "))]},
               "relevance": bq.relevance(catalog, lists), "configurations": {}}
        for fam_name, fam in (("single", single), ("multi", multi)):
            for config, names in fam.items():
                n = names[0]
                req, m = cfg["requests"][n], metrics[variant][n]
                res["configurations"][bq.config_name(config)] = {
                    "family": fam_name, "requests": names,
                    "served_by_feature_cells": variant != B and cc.uses_cells(list(config)),
                    "list_diversity": round(float(np.mean(m["list"])), 4),
                    "within_target_diversity": round(float(np.mean(m["within"])), 4),
                    "between_target_separation": (round(float(np.mean(m["between"])), 4) if m["between"] else None),
                    "random_eligible_list_diversity": reference["requests"][n]["diversity"],
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
        fam = {}
        for fam_name, f, key in (("single", single, "within"), ("multi", multi, "list"),
                                 ("multi_within_diagnostic", multi, "within")):
            per_run = np.mean([np.subtract(metrics[variant][ns[0]][key], metrics[REFERENCE][ns[0]][key])
                               for ns in f.values()], axis=0)
            fam[fam_name] = bq.boot(per_run, rng)
        results[variant]["paired_vs_reference"] = paired
        results[variant]["family_mean_change_vs_reference"] = fam
    ref_speech = results[REFERENCE]["speech_like_share"]["calm_requests"]
    results[N]["gates"] = bq.gate_status(results[N], ref_speech, single, multi)
    results[V]["diagnostic_gate_view"] = bq.gate_status(results[V], ref_speech, single, multi)
    return results


def cold_start_probe(which):
    t0 = time.perf_counter()
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str})
    load_ms = (time.perf_counter() - t0) * 1000
    times = []
    for i in range(10):
        np.random.seed(i)
        t1 = time.perf_counter()
        if which == B:
            recommend._coverage_top(catalog, ev.request_targets("sad", "match"), ev.LIST_SIZE)
        else:
            recommend.recommend_music(catalog, "sad", "Match my mood", top_n=ev.LIST_SIZE)
        times.append((time.perf_counter() - t1) * 1000)
    print(json.dumps({"catalog_load_ms": load_ms, "first_ms": times[0], "second_ms": times[1], "tenth_ms": times[9]}))


def cold_start():
    out = {}
    for which in (B, N):
        rows = []
        for _ in range(COLD_START_PROCESSES):
            res = subprocess.run([sys.executable, os.path.abspath(__file__), "--cold-start-probe", which],
                                 capture_output=True, text=True, check=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__)))
            rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
        out[which] = {k: round(float(np.median([r[k] for r in rows])), 2)
                      for k in ("catalog_load_ms", "first_ms", "second_ms", "tenth_ms")}
    return out


def main():
    parser = argparse.ArgumentParser(description="Phase 2K post-implementation check (analysis only)")
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
                     "classifier": mp.file_sha256(bq.CLASSIFIER_PATH),
                     "src_recommend": mp.file_sha256(recommend.__file__)}
    print(f"script sha256 {script_sha}; hashes {hashes_before}")
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    guarded_cells = bq.build_cells(catalog, z, guarded=True)

    primary = evaluate(catalog, z, PRIMARY_SEED, guarded_cells)
    replication = evaluate(catalog, z, REPLICATION_SEED, guarded_cells)
    replicated = {k: v == replication[N]["gates"][k] for k, v in primary[N]["gates"].items()}
    for tag, res in (("primary", primary), ("replication", replication)):
        for v, r in res.items():
            fm = r.get("family_mean_change_vs_reference") or {}
            print(f"[{tag}] {v:>44}: single {fm.get('single')} multi {fm.get('multi')} "
                  f"not-PASS {[k for k, x in (r.get('gates') or r.get('diagnostic_gate_view') or {}).items() if x != 'PASS']}")
    cold = cold_start()
    hashes_after = {"catalog": mp.file_sha256(MUSIC_CATALOG_278K_PATH),
                    "classifier": mp.file_sha256(bq.CLASSIFIER_PATH),
                    "src_recommend": mp.file_sha256(recommend.__file__)}
    if hashes_after != hashes_before:
        raise SystemExit("A source artifact changed during the analysis -- this must never happen.")

    single, multi = bq.families()
    report = {
        "analysis_only": True,
        "evaluation_date": datetime.date.today().isoformat(),
        "script_sha256": script_sha,
        "source_artifact_sha256": hashes_before,
        "production_configuration": {"pool_multiplier": div.POOL_MULTIPLIER,
                                     "feature_cell_targets": list(recommend.FEATURE_CELL_TARGETS),
                                     "calm_speech_guard": False},
        "preregistration": PREREGISTRATION,
        "gates": GATES,
        "seeds": {"primary": f"{PRIMARY_SEED}..{PRIMARY_SEED + RUNS - 1}",
                  "replication": f"{REPLICATION_SEED}..{REPLICATION_SEED + RUNS - 1}",
                  "previously_used": {k: f"{a}..{b}" for k, (a, b) in USED_SEEDS.items()},
                  "disjoint": seeds_disjoint(), "runs_per_request": RUNS, "bootstrap_rng": 0},
        "configurations": {"single_target": {bq.config_name(c): n for c, n in single.items()},
                           "multi_target": {bq.config_name(c): n for c, n in multi.items()}},
        "primary": primary,
        "replication": replication,
        "gate_decision_replicated": replicated,
        "cold_start_median_of_5_processes": cold,
        "notes": [
            "Diversity is feature-space diversity, not listener-perceived variety.",
            "Bootstrap intervals describe sampling variability over 50 seeded runs on one catalog only.",
            "Harness latency for B and N includes building the result DataFrame; V returns row positions only, so "
            "its latency is not comparable.",
            "N's first routed call in a process builds the feature cells; the cold-start figures include it.",
        ],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
