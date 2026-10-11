"""
Offline evaluation of the music recommender on the 278k catalog.

Run from the project root:  python evaluate_music_recommender.py [--runs N]

Reads   data/music_catalog_278k.csv (never modified)
Writes  models/music_recommender_eval.json  (aggregate metrics only: no track
        IDs, titles or other catalog rows)

This evaluates the recommendation lists, not the classifier; see
src/music_evaluation.py for the metric and relevance definitions and the
limitations, and README "Music recommendation evaluation".
"""
import argparse
import datetime
import json
import os
import sys
import time
import tracemalloc

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import MODELS_DIR, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp

REPORT_PATH = os.path.join(MODELS_DIR, "music_recommender_eval.json")


def _row(label, metrics, keys):
    return f"| {label} | " + " | ".join(f"{metrics[k]:.4f}" for k in keys) + " |"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=50, help="recommendation lists per (mood, mode) request")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    t0 = time.perf_counter()
    catalog_hash_before = mp.file_sha256(MUSIC_CATALOG_278K_PATH)
    tracemalloc.start()
    load_start = time.perf_counter()
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str})
    load_s = time.perf_counter() - load_start
    _, load_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    checks = ev.catalog_checks(catalog)
    result = ev.evaluate(catalog, runs=args.runs, seed=args.seed)
    if mp.file_sha256(MUSIC_CATALOG_278K_PATH) != catalog_hash_before:
        raise SystemExit("The catalog changed during the evaluation -- this must never happen.")

    report = {
        "evaluation_date": datetime.date.today().isoformat(),
        "what_this_measures": "recommendation lists served by src.recommend.recommend_music on the 278k catalog; "
                              "NOT classifier accuracy (see music_mood_classifier_278k_eval.json for that)",
        "catalog": {"path": os.path.relpath(MUSIC_CATALOG_278K_PATH, os.path.dirname(MODELS_DIR)),
                    "sha256": catalog_hash_before, **checks},
        "definitions": ev.DEFINITIONS,
        **result,
        "performance": {
            "catalog_csv_load_s": round(load_s, 2),
            "catalog_load_peak_traced_memory_mb": round(load_peak / 1e6, 1),
            "catalog_in_memory_mb": round(catalog.memory_usage(deep=True).sum() / 1e6, 1),
            "recommend_music_latency_ms": result["latency_ms"],
            "evaluation_runtime_s": round(time.perf_counter() - t0, 1),
        },
        "limitations": ev.LIMITATIONS,
    }
    report.pop("latency_ms")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    keys_p = [f"precision@{k}" for k in ev.KS]
    keys_n = [f"ndcg@{k}" for k in ev.KS]
    print(f"Catalog: {checks['rows']} tracks, {checks['unique_track_ids']} unique IDs; {args.runs} runs per request")
    for title, keys in [("Precision", keys_p), ("NDCG", keys_n)]:
        print(f"\n| Request | " + " | ".join(keys) + " |")
        for name, req in report["requests"].items():
            print(_row(name, req["moodcast"], keys))
    overall = report["overall_mean_over_requests"]
    print("\n| Metric | MoodCast | Random (catalog) | Random (eligible) |")
    for key in ["precision@10", "recall@10", "ndcg@10", "diversity"]:
        print(f"| {key} | {overall['moodcast'][key]:.6f} | {overall['random_catalog_baseline'][key]:.6f} | "
              f"{overall['random_eligible_baseline'][key]:.6f} |")
    cov = report["catalog_coverage"]
    print(f"\nCoverage: {cov['unique_tracks_served']} / {cov['catalog_tracks']} = {cov['coverage']:.4%}")
    print(f"Performance: {json.dumps(report['performance'])}")
    print(f"\nWrote {os.path.relpath(REPORT_PATH)}")


if __name__ == "__main__":
    main()
