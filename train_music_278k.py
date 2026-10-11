"""
Offline music pipeline: build the 278k Spotify-track-ID-keyed catalog,
retrain the 4-label audio mood classifier on it, add the derived moods
(angry, experimental stressed), and measure target coverage of the
recommender.

Run from the project root:  python train_music_278k.py

Reads   data/music_dataset2/278k_labelled_uri.csv   (never modified; its
        SHA-256 is checked before and after the run)
Writes  data/music_catalog_278k.csv
        models/music_mood_classifier_278k.pkl
        models/music_mood_classifier_278k_eval.json

Unlike train_all_models.py this touches nothing else. (The comparisons with
the legacy music_final.csv pipeline and the old audio_mood_classifier.pkl were
removed when those were retired in Phase 2H.4; a re-run's eval JSON no longer
has the `comparison_with_legacy` / `legacy_audio_classifier` sections.)

The classifier is evaluated on a stratified 15% held-out split (same split
seed as train_all_models.py's audio classifier, so the two are comparable);
the shipped model is then refit on all 277,938 tracks. Recommendation
behavior is measured as target coverage (how many tracks each requested
target gets), not as a quality claim.
"""
import json
import os
import pickle
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (DERIVED_MUSIC_MOODS, LABEL_TO_MOOD, MODELS_DIR, MOODS,
                        MUSIC_278K_SOURCE_PATH, MUSIC_CATALOG_278K_PATH)
from src import music_pipeline as mp
from src import recommend
from src.ml.metrics import accuracy_score, classification_report, confusion_matrix
from src.ml.models import GaussianNaiveBayes, SoftmaxRegression
from src.ml.preprocessing import k_fold_indices, train_test_split

MODEL_PATH = os.path.join(MODELS_DIR, "music_mood_classifier_278k.pkl")
EVAL_PATH = os.path.join(MODELS_DIR, "music_mood_classifier_278k_eval.json")

MODES = ["Match my mood", "Cheer me up"]
TOP_K = 30
COVERAGE_KS = (30, 50)
SERVED_REPEATS = 20
EXAMPLE_REQUESTS = [("Cheer me up", "sad"), ("Match my mood", "angry"),
                    ("Match my mood", "stressed"), ("Cheer me up", "chill")]
EXAMPLE_COLUMNS = ["track_id", "target", "mood", "score", "valence", "energy", "loudness", "danceability"]


def _label_mix(moods: pd.Series) -> dict:
    return {m: int(n) for m, n in moods.value_counts().items()}


def evaluate_classifier(df, idx_tr, idx_te):
    X = df[mp.AUDIO_FEATURES].values.astype(float)
    y = df["label"].values
    Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]

    scaler, model = mp.fit_classifier(Xtr, ytr)
    preds = model.predict(scaler.transform(Xte))
    nb_preds = GaussianNaiveBayes().fit(scaler.transform(Xtr), ytr).predict(scaler.transform(Xte))
    names = [LABEL_TO_MOOD[l] for l in mp.LABELS]
    report_txt, report = classification_report(yte, preds, labels=mp.LABELS, target_names=names)
    cm, _ = confusion_matrix(yte, preds, labels=mp.LABELS)

    cv = []
    Xtr_s = scaler.transform(Xtr)
    for fold_tr, fold_val in k_fold_indices(len(Xtr_s), k=5, random_state=42, stratify=ytr):
        m = SoftmaxRegression(**{**mp.CLASSIFIER_PARAMS, "epochs": 60}).fit(Xtr_s[fold_tr], ytr[fold_tr])
        cv.append(accuracy_score(ytr[fold_val], m.predict(Xtr_s[fold_val])))

    print(f"  NaiveBayes acc={accuracy_score(yte, nb_preds):.4f} | Softmax acc={accuracy_score(yte, preds):.4f} | "
          f"5-fold CV={np.mean(cv):.4f} +/- {np.std(cv):.4f}")
    print(report_txt)
    return scaler, model, {
        "train_rows": int(len(idx_tr)), "test_rows": int(len(idx_te)),
        "naive_bayes_test_accuracy": accuracy_score(yte, nb_preds),
        "softmax_test_accuracy": accuracy_score(yte, preds),
        "softmax_cv_mean_accuracy": float(np.mean(cv)), "softmax_cv_std_accuracy": float(np.std(cv)),
        "classification_report": report, "confusion_matrix": cm.tolist(),
        "confusion_matrix_label_order": names,
    }


def _off_target(recs) -> int:
    """Tracks that don't actually belong to the target they were picked for."""
    bad = 0
    for target, rows in recs.groupby("target"):
        member, _ = recommend._target_candidates(rows, target)
        bad += int((~member).sum())
    return bad


def _weighted_sum_top(catalog, targets, k):
    """Phase 1 baseline: rank-weighted sum of classifier probabilities (real labels only)."""
    if any(t in DERIVED_MUSIC_MOODS for t, _ in targets):
        return None
    score = sum(w * catalog[f"proba_{t}"].to_numpy() for t, w in targets)
    top = catalog.iloc[np.argsort(-score, kind="stable")[:k]]
    return {t: int((top["mood"] == t).sum()) for t, _ in targets}


STRESSED_DIAG_FEATURES = ["valence", "energy", "danceability", "loudness", "instrumentalness",
                          "speechiness", "duration_ms", "tempo"]


def _outlier_counts(rows, catalog) -> dict:
    return {
        "zero_valence_danceability_or_tempo": int(((rows.valence == 0) | (rows.danceability == 0)
                                                  | (rows.tempo == 0)).sum()),
        "speechiness_gt_0.66": int((rows.speechiness > 0.66).sum()),
        "loudness_below_catalog_p5": int((rows.loudness < catalog.loudness.quantile(0.05)).sum()),
        "instrumentalness_gt_0.9": int((rows.instrumentalness > 0.9).sum()),
    }


def evaluate_stressed(catalog):
    """
    Outlier diagnostics for the experimental stressed ranking: its
    deterministic top-30 and top-90 (the served sampling pool) versus the
    previous extreme-score ranking (mean of inverted valence/danceability
    percentiles over the unguarded rule matches), kept here as a baseline.
    """
    def summarize(order):
        top30, top90 = catalog.iloc[order[:30]], catalog.iloc[order[:90]]
        return {
            "top30_feature_min_median_max": {
                f: [round(float(top30[f].min()), 3), round(float(top30[f].median()), 3), round(float(top30[f].max()), 3)]
                for f in STRESSED_DIAG_FEATURES},
            "top30_true_label_mix": _label_mix(top30["mood"]),
            "top30_outliers": _outlier_counts(top30, catalog),
            "top90_outliers": _outlier_counts(top90, catalog),
        }

    recs = recommend._coverage_top(catalog, [("stressed", 1.0)], 90, pool_multiplier=1)
    current = catalog.index.get_indexer(recs.index)

    rule_only = np.flatnonzero(_rule_matches(catalog, "stressed"))
    extreme = (2 - catalog["valence"].rank(pct=True) - catalog["danceability"].rank(pct=True)).to_numpy() / 2
    baseline = rule_only[np.argsort(-extreme[rule_only], kind="stable")]
    return {"profile_ranking": summarize(current), "previous_extreme_ranking_baseline": summarize(baseline)}


def _rule_matches(catalog, mood):
    """Tracks meeting a derived mood's percentile rule, before guards."""
    mask = np.ones(len(catalog), dtype=bool)
    for feature, op, percentile in DERIVED_MUSIC_MOODS[mood]["rule"]:
        mask &= mp._RULE_OPS[op](catalog[feature].to_numpy(dtype=float), catalog[feature].quantile(percentile))
    return mask


def evaluate_target_coverage(catalog):
    """
    For every app mood/mode on the full catalog: requested shares, per-target
    counts of the deterministic ranking (pool_multiplier=1) at top-30/50,
    what the served (sampled) path returns over repeated calls, duplicates,
    off-target picks, and the Phase 1 weighted-sum baseline. This measures
    representation of the requested targets, not recommendation quality.
    """
    out, examples = {}, {}
    for mode in MODES:
        for mood in MOODS:
            targets = recommend._target_music_weighted(mood, mode)
            entry = {"targets": {t: round(w, 3) for t, w in targets}}
            for k in COVERAGE_KS:
                recs = recommend._coverage_top(catalog, targets, k, pool_multiplier=1)
                entry[f"top{k}_requested_shares"] = dict(zip(
                    [t for t, _ in targets], recommend._allocate_shares([w for _, w in targets], k)))
                entry[f"top{k}_target_counts"] = _label_mix(recs["target"])
                entry[f"top{k}_true_label_mix"] = _label_mix(recs["mood"])
                entry[f"top{k}_duplicate_ids"] = int(recs["track_id"].duplicated().sum())
                entry[f"top{k}_off_target"] = _off_target(recs)
                entry[f"top{k}_weighted_sum_baseline"] = _weighted_sum_top(catalog, targets, k)
                if k == TOP_K and (mode, mood) in EXAMPLE_REQUESTS:
                    examples[f"{mode} / {mood}"] = recs[EXAMPLE_COLUMNS].head(8).round(3).to_dict(orient="records")

            served_ids, counts, dups, off = set(), [], 0, 0
            for _ in range(SERVED_REPEATS):
                recs = recommend.recommend_music(catalog, mood, mode, top_n=TOP_K)
                served_ids.update(recs["track_id"])
                counts.append(recs["target"].value_counts())
                dups += int(recs["track_id"].duplicated().sum())
                off += _off_target(recs)
            counts = pd.DataFrame(counts).fillna(0)
            entry[f"served_{SERVED_REPEATS}x_top{TOP_K}"] = {
                "target_counts_min_max": {t: [int(counts[t].min()), int(counts[t].max())] for t in counts.columns},
                "unique_tracks": len(served_ids),
                "duplicate_ids_within_a_response": dups,
                "off_target": off,
            }
            out[f"{mode} / {mood}"] = entry
    return out, examples


def main():
    t0 = time.time()
    tracemalloc.start()
    source_hash_before = mp.file_sha256(MUSIC_278K_SOURCE_PATH)

    print("=== [1/4] Load + validate 278k source ===")
    t = time.time()
    df = mp.build_training_frame(mp.load_source(MUSIC_278K_SOURCE_PATH))
    report = mp.dataset_report(df)
    load_s = time.time() - t
    print(f"  {report['rows']} rows, {report['unique_track_ids']} unique track IDs, "
          f"{report['duplicate_track_ids']} duplicate IDs ({load_s:.1f}s)")
    print(f"  class distribution: {report['class_distribution']}")

    print("\n=== [2/4] Train/evaluate classifier on held-out split ===")
    t = time.time()
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.15, random_state=42,
                                      stratify=df["label"].values)
    _, _, clf_eval = evaluate_classifier(df, idx_tr, idx_te)
    eval_s = time.time() - t

    print("=== [3/4] Refit on all tracks + build catalog + derived moods ===")
    t = time.time()
    X_all = df[mp.AUDIO_FEATURES].values.astype(float)
    final_scaler, final_model = mp.fit_classifier(X_all, df["label"].values)
    catalog = mp.build_catalog(df, mp.predict_label_proba(df, final_scaler, final_model))
    fit_s = time.time() - t
    t = time.time()
    catalog, derived = mp.derive_moods(catalog)
    derive_s = time.time() - t
    for mood, meta in derived.items():
        flag = " (EXPERIMENTAL)" if meta["experimental"] else ""
        print(f"  derived {mood}{flag}: {meta['rule_match_count']} rule matches -> {meta['candidate_count']} "
              f"candidates, true labels {meta['candidate_true_label_mix']}, cutoffs "
              f"{[(r['feature'], r['op'], round(r['cutoff'], 3)) for r in meta['rule']]}")
        for name, g in meta["guards"].items():
            print(f"    guard {name}: excludes {g['rule_matches_excluded']} rule matches "
                  f"({g['tracks_in_catalog']} tracks catalog-wide)")
        if "profile_percentiles" in meta["score"]:
            print(f"    profile percentiles: { {f: round(v, 3) for f, v in meta['score']['profile_percentiles'].items()} }")
    # LF on every platform, so the bytes match the SHA-256 recorded in models/*.json
    catalog.to_csv(MUSIC_CATALOG_278K_PATH, index=False, lineterminator="\n")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "scaler": final_scaler, "features": mp.AUDIO_FEATURES,
                     "label_to_mood": LABEL_TO_MOOD}, f)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t = time.time()
    reloaded = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str})
    reload_s = time.time() - t
    in_sample_acc = float((catalog["predicted_mood"] == catalog["mood"]).mean())

    print("\n=== [4/4] Target coverage of the recommender ===")
    t = time.time()
    coverage, examples = evaluate_target_coverage(reloaded)
    coverage_s = time.time() - t
    for key, c in coverage.items():
        print(f"  {key:28s} top30={c['top30_target_counts']} top50={c['top50_target_counts']} "
              f"dup={c['top30_duplicate_ids']} off={c['top30_off_target']} "
              f"phase1_baseline={c['top30_weighted_sum_baseline']}")

    stressed_diag = evaluate_stressed(reloaded)
    for name, d in stressed_diag.items():
        print(f"  stressed {name}: top30 outliers={d['top30_outliers']} top90 outliers={d['top90_outliers']} "
              f"labels={d['top30_true_label_mix']}")

    source_hash_after = mp.file_sha256(MUSIC_278K_SOURCE_PATH)
    if source_hash_after != source_hash_before:
        raise SystemExit("278k source file changed during the run -- this must never happen.")

    result = {
        "source_file": os.path.relpath(MUSIC_278K_SOURCE_PATH, os.path.dirname(MODELS_DIR)),
        "source_sha256": source_hash_before,
        "label_to_mood": LABEL_TO_MOOD,
        "features": mp.AUDIO_FEATURES,
        "classifier_params": mp.CLASSIFIER_PARAMS,
        "dataset": report,
        "held_out_evaluation": clf_eval,
        "derived_moods": derived,
        "target_coverage": coverage,
        "stressed_outlier_diagnostics": stressed_diag,
        "recommendation_examples_top30_head": examples,
        "final_model": {"trained_rows": int(len(df)), "in_sample_accuracy": in_sample_acc},
        "catalog": {"path": os.path.relpath(MUSIC_CATALOG_278K_PATH, os.path.dirname(MODELS_DIR)),
                    "rows": int(len(reloaded)), "columns": reloaded.columns.tolist(),
                    "unique_track_ids": int(reloaded["track_id"].nunique()),
                    "missing_values": int(reloaded.isna().sum().sum())},
        "runtime": {
            "load_validate_s": round(load_s, 1), "held_out_train_eval_s": round(eval_s, 1),
            "final_fit_s": round(fit_s, 1), "derive_moods_s": round(derive_s, 2),
            "coverage_eval_s": round(coverage_s, 1),
            "total_s": round(time.time() - t0, 1),
            "catalog_csv_mb": round(os.path.getsize(MUSIC_CATALOG_278K_PATH) / 1e6, 1),
            "catalog_load_s": round(reload_s, 2),
            "catalog_in_memory_mb": round(reloaded.memory_usage(deep=True).sum() / 1e6, 1),
            "pipeline_peak_traced_memory_mb": round(peak / 1e6, 1),
        },
    }
    with open(EVAL_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nDone in {result['runtime']['total_s']}s. Catalog: {MUSIC_CATALOG_278K_PATH}\n"
          f"Model: {MODEL_PATH}\nEvaluation: {EVAL_PATH}")


if __name__ == "__main__":
    main()
