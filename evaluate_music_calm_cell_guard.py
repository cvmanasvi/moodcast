"""
ANALYSIS ONLY -- Phase 2K.1 check of the calm speechiness guard on the
production calm feature-cell route. It calls the production code; it changes
nothing.

Run from the project root:
    python evaluate_music_calm_cell_guard.py [--output PATH]

Reads   data/music_catalog_278k.csv (read-only)
Writes  models/music_calm_cell_guard_check.json (aggregate metrics only: no
        track IDs, titles, artists, names or URIs). No earlier report is touched.

Variants:
  K1  current production (Phase 2K.1): recommend.recommend_music, calm cells exclude
      speechiness > 0.66
  K   Phase 2K as implemented before the guard: identical to K1 except that the two
      calm-only requests are served from cells over unguarded calm eligibility
      (reproduced with the same unguarded cell selection the Phase 2K tests verified)
  B   previous production for context: recommend._coverage_top for every request
Only chill/match and stressed/uplift can differ between K and K1.
PREREGISTRATION was fixed before the first run on these seeds.
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
import evaluate_music_cell_production as cp
import evaluate_music_diversity_analysis as div
import evaluate_music_recommender_analysis as sweep
import evaluate_music_stratified_prototype as proto
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend

OUTPUT_PATH = os.path.join(MODELS_DIR, "music_calm_cell_guard_check.json")
RUNS = 50
PRIMARY_SEED = 42000
REPLICATION_SEED = 45000
USED_SEEDS = {**cp.USED_SEEDS, "primary (2K)": (36000, 36049), "replication (2K)": (39000, 39049)}
K1, K, B = ("K1: production with calm cell guard (2K.1)", "K: production before the guard (2K)",
            "B: previous production, no cells (context)")
VARIANTS = (K1, K, B)
CALM_ONLY = ("chill / match", "stressed / uplift")
COLD_START_PROCESSES = 5

PREREGISTRATION = {
    "primary_comparison": "K1 against K, paired on identical seeds: calm-only configuration diversity and "
                          "speech-like share, plus every other configuration (expected identical).",
    "prior_gates": "The Phase 2J.1 gates are reported unchanged for K1 against B, the frame used for the Phase 2K "
                   "check. No gate is added, removed or changed.",
    "interpretation_fixed_in_advance": "Calm diversity of the unguarded routes (B ~4.06, K ~4.71) is inflated by "
                                       "speech-like tracks (Phase 2H.8.1, 2K). A lower guarded calm diversity is "
                                       "therefore reported as the removal of that inflation, not as a regression "
                                       "in variety, and guarded calm diversity is not claimed as a gain over B.",
    "scope_of_claims": "K1 matches the Phase 2J.1 guarded selection on the calm-only route only; the three uplift "
                       "mixes containing calm stay on the unguarded previous ranking, so K1 does not reproduce the "
                       "Phase 2J.1 guarded-baseline experiment as a whole.",
    "seeds": "42000-42049 decide the reported gate statuses; 45000-45049 replicate. Both disjoint from every "
             "earlier seed set.",
}


def seeds_disjoint():
    sets = [(PRIMARY_SEED, PRIMARY_SEED + RUNS - 1), (REPLICATION_SEED, REPLICATION_SEED + RUNS - 1)]
    return all(hi < a or lo > b for lo, hi in sets for a, b in USED_SEEDS.values()) and sets[0][1] < sets[1][0]


def recommender_for(variant, catalog, unguarded_calm_cells):
    if variant == B:
        return lambda targets: recommend._coverage_top(catalog, targets, ev.LIST_SIZE).index.to_numpy()
    k1 = lambda targets: recommend.recommend_music(catalog, *cp.request_for(targets),
                                                   top_n=ev.LIST_SIZE).index.to_numpy()
    if variant == K1:
        return k1
    member, signal = recommend._target_candidates(catalog, "calm")

    def k(targets):
        if list(targets) == [("calm", 1.0)]:
            return bq.cell_select("calm", signal, member, ev.LIST_SIZE, unguarded_calm_cells)
        return k1(targets)
    return k


def evaluate(catalog, z, seed, unguarded_calm_cells):
    single, multi = bq.families()
    reference = sweep.random_eligible_reference(catalog, z, RUNS, seed)
    sp = catalog["speechiness"].to_numpy(dtype=float)
    calm_member = recommend._target_candidates(catalog, "calm")[0]
    results, metrics = {}, {}
    for variant in VARIANTS:
        fn = recommender_for(variant, catalog, unguarded_calm_cells)
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
        metrics[variant] = bq.per_run_metrics(catalog, z, lists)
        calm_slot = {}
        for n in lists:
            if "calm" in [t for t, _ in ev.request_targets(*n.split(" / "))]:
                rows = np.concatenate(lists[n])
                calm_slot[n] = round(float((sp[rows[calm_member[rows]]] > 0.66).mean()), 4)
        res = {"summary": div.summarize(cfg, reference, list(cfg["requests"]))["summary"],
               "latency": proto.latency_summary(latencies),
               "speech_like_share": proto.speech_like(catalog, lists),
               "calm_slot_speech_like_share": calm_slot,
               "relevance": bq.relevance(catalog, lists), "configurations": {}}
        for fam_name, fam in (("single", single), ("multi", multi)):
            for config, names in fam.items():
                n = names[0]
                req, m = cfg["requests"][n], metrics[variant][n]
                res["configurations"][bq.config_name(config)] = {
                    "family": fam_name, "requests": names,
                    "list_diversity": round(float(np.mean(m["list"])), 4),
                    "within_target_diversity": round(float(np.mean(m["within"])), 4),
                    "random_eligible_list_diversity": reference["requests"][n]["diversity"],
                    "unique_tracks_over_runs": req["unique_tracks_over_runs"], "repeat_rate": req["repeat_rate"],
                    "target_consistency": req["target_consistency"],
                    "exact_target_shares_fraction": req["exact_target_shares_fraction"],
                }
        results[variant] = res
    rng = np.random.RandomState(0)

    def paired(a_name, b_name):
        out = {}
        for config, names in list(single.items()) + list(multi.items()):
            a, b = metrics[a_name][names[0]], metrics[b_name][names[0]]
            out[bq.config_name(config)] = {
                "list_diversity": bq.boot(np.subtract(b["list"], a["list"]), rng),
                "within_target_diversity": bq.boot(np.subtract(b["within"], a["within"]), rng),
                "identical_lists": all(np.array_equal(x, y) for x, y in zip(a["list"], b["list"]))}
        return out

    def family(a_name, b_name):
        fam = {}
        for fam_name, f, key in (("single", single, "within"), ("multi", multi, "list")):
            per_run = np.mean([np.subtract(metrics[b_name][ns[0]][key], metrics[a_name][ns[0]][key])
                               for ns in f.values()], axis=0)
            fam[fam_name] = bq.boot(per_run, rng)
        return fam

    results[K1]["paired_vs_K"] = paired(K, K1)
    results[K1]["paired_vs_B"] = paired(B, K1)
    results[K1]["paired_vs_reference"] = results[K1]["paired_vs_B"]
    results[K1]["family_mean_change_vs_reference"] = family(B, K1)
    results[K]["paired_vs_reference"] = paired(B, K)
    results[K]["family_mean_change_vs_reference"] = family(B, K)
    b_speech = results[B]["speech_like_share"]["calm_requests"]
    results[K1]["prior_gates_vs_B"] = bq.gate_status(results[K1], b_speech, single, multi)
    results[K]["prior_gates_vs_B"] = bq.gate_status(results[K], b_speech, single, multi)
    return results


def cold_start_probe(_):
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str})
    times = []
    for i in range(10):
        np.random.seed(i)
        t0 = time.perf_counter()
        recommend.recommend_music(catalog, "chill", "Match my mood", top_n=ev.LIST_SIZE)
        times.append((time.perf_counter() - t0) * 1000)
    print(json.dumps({"first_ms": times[0], "second_ms": times[1], "tenth_ms": times[9]}))


def cold_start():
    rows = []
    for _ in range(COLD_START_PROCESSES):
        res = subprocess.run([sys.executable, os.path.abspath(__file__), "--cold-start-probe", K1],
                             capture_output=True, text=True, check=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
    return {K1: {k: round(float(np.median([r[k] for r in rows])), 2) for k in ("first_ms", "second_ms", "tenth_ms")}}


def main():
    parser = argparse.ArgumentParser(description="Phase 2K.1 calm cell guard check (analysis only)")
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
    unguarded_calm_cells = {"calm": bq.build_cells(catalog, z, guarded=False)["calm"]}

    primary = evaluate(catalog, z, PRIMARY_SEED, unguarded_calm_cells)
    replication = evaluate(catalog, z, REPLICATION_SEED, unguarded_calm_cells)
    for tag, res in (("primary", primary), ("replication", replication)):
        r = res[K1]
        calm = r["paired_vs_K"]["calm:30"]
        others_identical = all(v["identical_lists"] for c, v in r["paired_vs_K"].items() if c != "calm:30")
        print(f"[{tag}] calm K {res[K]['configurations']['calm:30']['list_diversity']} -> K1 "
              f"{r['configurations']['calm:30']['list_diversity']} (B {res[B]['configurations']['calm:30']['list_diversity']}); "
              f"K1-K {calm['list_diversity']}; speech K {res[K]['speech_like_share']['calm_requests']} K1 "
              f"{r['speech_like_share']['calm_requests']} B {res[B]['speech_like_share']['calm_requests']}; "
              f"other configs identical: {others_identical}; prior gates not PASS (K1 vs B): "
              f"{[k for k, v in r['prior_gates_vs_B'].items() if v != 'PASS']}")
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
                                     "calm_cell_speechiness_max": recommend.CALM_CELL_SPEECHINESS_MAX},
        "preregistration": PREREGISTRATION,
        "prior_gates": bq.GATES,
        "seeds": {"primary": f"{PRIMARY_SEED}..{PRIMARY_SEED + RUNS - 1}",
                  "replication": f"{REPLICATION_SEED}..{REPLICATION_SEED + RUNS - 1}",
                  "previously_used": {k: f"{a}..{b}" for k, (a, b) in USED_SEEDS.items()},
                  "disjoint": seeds_disjoint(), "runs_per_request": RUNS, "bootstrap_rng": 0},
        "configurations": {"single_target": {bq.config_name(c): n for c, n in single.items()},
                           "multi_target": {bq.config_name(c): n for c, n in multi.items()}},
        "primary": primary,
        "replication": replication,
        "cold_start_median_of_5_processes": cold,
        "notes": [
            "Diversity is feature-space diversity, not listener-perceived variety; on calm it is inflated by "
            "speech-like tracks, so the guarded calm figure is not comparable as a gain or loss of variety.",
            "Bootstrap intervals describe sampling variability over 50 seeded runs on one catalog only.",
            "Cold start covers a calm-only request in a fresh process, including the one-time cell build.",
        ],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
