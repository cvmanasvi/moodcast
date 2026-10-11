"""
Training pipeline for the text and movie mood models.

Run this from the project root (`python train_all_models.py`) any time you
want to retrain them from scratch (`python train_all_models.py --text-only`
retrains only the text mood model and leaves the movie model and
data/movies_final.csv untouched): it downloads nothing and needs no
external ML library (scikit-learn/xgboost) -- every model here is implemented
directly on numpy/pandas in src/ml/ (softmax regression, TF-IDF), so
`pip install -r requirements.txt` is all you need.

Stages:
  1-2. Text mood dataset + classifier -- TF-IDF + softmax regression, trained
     on a hand-curated ~1300-example dataset spanning the 6 moods, used to
     replace the old keyword-matching mood detector.
  3. Movie mood classifier -- TF-IDF over each movie's plot + genres,
     predicting the mood category from natural language instead of a fixed
     genre lookup table.
  apply_ml_labels -- re-scores data/movies_final.csv with the movie model
     (adds mood_ml / mood_confidence / proba_* columns, and makes the
     ML-predicted mood the operational `mood_tag` column the app uses). The
     original rule-based labels are preserved as
     data/movies_final_rulebased_backup.csv and as a `mood_tag_rulebased` column.

The music models live elsewhere: the 278k catalog and its classifier are
built by train_music_278k.py. (The unused research audio classifier and
global K-Means this script used to train were retired in Phase 2H.4.)

All metrics printed along the way (test accuracy, 5-fold CV, confusion
matrices, held-out generalization checks) are also written as JSON next to
each model under models/, so you can inspect exactly how the ML was
validated instead of taking it on faith.
"""
import os
import sys
import csv
import json
import time
import pickle
import random
import itertools
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import BASE_DIR, MODELS_DIR
from src.ml.preprocessing import TfidfVectorizer, train_test_split, k_fold_indices
from src.ml.models import SoftmaxRegression
from src.ml.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)

def save_json(obj, name):
    with open(os.path.join(MODELS_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ============================================================================
# 3. TEXT MOOD DATASET + CLASSIFIER  (for the "describe your mood" box)
# ============================================================================
def _norm(text):
    """Case/punctuation-insensitive form used to keep evaluation sentences out of training."""
    return " ".join("".join(c for c in str(text).casefold() if c.isalnum() or c.isspace()).split())


def _evaluation_texts():
    from text_mood_probe import BLIND, PROBE
    return [t for t, _ in _TEXT_HOLDOUT] + [t for t, _, _ in PROBE] + [t for t, _, _ in BLIND]


def build_text_mood_dataset():
    print("\n=== [1/3] Building text mood training set ===")
    random.seed(42)
    VOCAB = _TEXT_VOCAB
    CONNECTORS = ["feeling {feeling} {context}", "{feeling} {context}", "feeling {feeling}",
                  "{feeling}, {extra}", "{extra}", "{subject} feel {feeling} {context}",
                  "{subject} {feeling} {context}", "not going to lie, {feeling} {context}"]
    rows = []
    for mood, v in VOCAB.items():
        combos = list(itertools.product(v["subjects"], v["feelings"], v["contexts"] + _TEXT_SHARED_CONTEXTS))
        random.shuffle(combos)
        used = set()
        for subject, feeling, context in combos[:260]:
            template = random.choice(CONNECTORS)
            extra = random.choice(v["extra"])
            text = " ".join(template.format(subject=subject, feeling=feeling, context=context, extra=extra).split())
            if text and text not in used:
                used.add(text)
                rows.append((text, mood))
        for extra in v["extra"] + _TEXT_EVERYDAY.get(mood, []):
            if extra not in used:
                used.add(extra)
                rows.append((extra, mood))
    # evaluation sentences (held-out set and the audit probes) must never be trained on
    held_out = {_norm(t) for t in _evaluation_texts()}
    before = len(rows)
    rows = [(t, m) for t, m in rows if _norm(t) not in held_out]
    random.shuffle(rows)
    print(f"Generated {len(rows)} examples ({before - len(rows)} evaluation sentences left out): "
          f"{Counter(m for _, m in rows)}")
    out_path = os.path.join(DATA_DIR, "text_mood_dataset.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "mood"])
        writer.writerows(rows)
    return out_path


def train_text_classifier(dataset_path):
    print("\n=== [2/3] Text mood classifier ===")
    df = pd.read_csv(dataset_path)
    moods = sorted(df["mood"].unique())
    mood_to_idx = {m: i for i, m in enumerate(moods)}
    y = df["mood"].map(mood_to_idx).values
    texts = df["text"].tolist()

    texts_tr, texts_te, ytr, yte = train_test_split(np.array(texts, dtype=object), y,
                                                      test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
    Xtr, Xte = vec.fit_transform(texts_tr.tolist()), vec.transform(texts_te.tolist())
    clf = SoftmaxRegression(n_classes=len(moods), lr=0.8, epochs=400, l2=5e-4, batch_size=256, random_state=42)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    acc = accuracy_score(yte, preds)
    report_txt, report_dict = classification_report(yte, preds, labels=list(range(len(moods))), target_names=moods)
    cm, cm_labels = confusion_matrix(yte, preds, labels=list(range(len(moods))))

    accs = []
    texts_arr = np.array(texts, dtype=object)
    for train_idx, val_idx in k_fold_indices(len(texts_arr), k=5, random_state=42, stratify=y):
        v2 = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
        Xtr2, Xval2 = v2.fit_transform(texts_arr[train_idx].tolist()), v2.transform(texts_arr[val_idx].tolist())
        m2 = SoftmaxRegression(n_classes=len(moods), lr=0.8, epochs=200, l2=5e-4, batch_size=256, random_state=42)
        m2.fit(Xtr2, y[train_idx])
        accs.append(accuracy_score(y[val_idx], m2.predict(Xval2)))

    print(f"Test acc={acc:.4f} | 5-fold CV={np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(report_txt)

    # final model on ALL data, evaluated against a genuinely unseen holdout set
    vec_final = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
    X_all = vec_final.fit_transform(texts)
    clf_final = SoftmaxRegression(n_classes=len(moods), lr=0.8, epochs=400, l2=5e-4, batch_size=256, random_state=42)
    clf_final.fit(X_all, y)

    holdout = pd.DataFrame(_TEXT_HOLDOUT, columns=["text", "mood"])
    Xh = vec_final.transform(holdout["text"].tolist())
    proba_h = clf_final.predict_proba(Xh)
    preds_h = proba_h.argmax(axis=1)
    yh = holdout["mood"].map(mood_to_idx).values
    holdout_acc = accuracy_score(yh, preds_h)
    print(f"Held-out hand-written sentence generalization accuracy: {holdout_acc:.4f} ({len(holdout)} examples)")

    with open(os.path.join(MODELS_DIR, "text_mood_classifier.pkl"), "wb") as f:
        pickle.dump({"model": clf_final, "vectorizer": vec_final, "moods": moods}, f)
    save_json({
        "n_examples": len(df), "moods": moods, "in_distribution_test_accuracy": acc,
        "cv_mean_accuracy": float(np.mean(accs)), "cv_std_accuracy": float(np.std(accs)),
        "classification_report": report_dict, "confusion_matrix": cm.tolist(),
        "confusion_matrix_label_order": [moods[l] for l in cm_labels],
        "holdout_generalization_accuracy": holdout_acc,
        "holdout_examples": [
            {"text": r["text"], "expected": r["mood"], "predicted": moods[preds_h[i]],
             "confidence": float(proba_h[i, preds_h[i]])}
            for i, r in holdout.iterrows()
        ],
    }, "text_mood_classifier_eval.json")


# ============================================================================
# 5. MOVIE MOOD CLASSIFIER  (TF-IDF over overview+genres -> mood)
# ============================================================================
def train_movie_classifier():
    print("\n=== [3/3] Movie mood classifier ===")
    backup_path = os.path.join(DATA_DIR, "movies_final_rulebased_backup.csv")
    source_path = backup_path if os.path.exists(backup_path) else os.path.join(DATA_DIR, "movies_final.csv")

    df = pd.read_csv(source_path).dropna(subset=["overview", "genres", "mood_tag"]).reset_index(drop=True)
    df["text"] = df["overview"].astype(str) + " " + df["genres"].astype(str)
    moods = sorted(df["mood_tag"].unique())
    mood_to_idx = {m: i for i, m in enumerate(moods)}
    y = df["mood_tag"].map(mood_to_idx).values
    texts = df["text"].tolist()

    texts_tr, texts_te, ytr, yte = train_test_split(np.array(texts, dtype=object), y,
                                                      test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=4)
    Xtr, Xte = vec.fit_transform(texts_tr.tolist()), vec.transform(texts_te.tolist())
    clf = SoftmaxRegression(n_classes=len(moods), lr=0.7, epochs=100, l2=2e-4, batch_size=4096,
                             random_state=42, class_weight="balanced")
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    acc = accuracy_score(yte, preds)
    report_txt, report_dict = classification_report(yte, preds, labels=list(range(len(moods))), target_names=moods)
    cm, cm_labels = confusion_matrix(yte, preds, labels=list(range(len(moods))))

    accs = []
    texts_arr = np.array(texts, dtype=object)
    for train_idx, val_idx in k_fold_indices(len(texts_arr), k=2, random_state=42, stratify=y):
        v2 = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=4)
        Xtr2, Xval2 = v2.fit_transform(texts_arr[train_idx].tolist()), v2.transform(texts_arr[val_idx].tolist())
        m2 = SoftmaxRegression(n_classes=len(moods), lr=0.7, epochs=60, l2=2e-4, batch_size=4096,
                                random_state=42, class_weight="balanced")
        m2.fit(Xtr2, y[train_idx])
        accs.append(accuracy_score(y[val_idx], m2.predict(Xval2)))

    print(f"Test acc={acc:.4f} | CV={np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(report_txt)

    vec_final = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=4)
    X_all = vec_final.fit_transform(texts)
    clf_final = SoftmaxRegression(n_classes=len(moods), lr=0.7, epochs=100, l2=2e-4, batch_size=4096,
                                   random_state=42, class_weight="balanced")
    clf_final.fit(X_all, y)

    with open(os.path.join(MODELS_DIR, "movie_mood_classifier.pkl"), "wb") as f:
        pickle.dump({"model": clf_final, "vectorizer": vec_final, "moods": moods}, f)
    save_json({
        "n_movies": len(df), "moods": moods, "test_accuracy": acc,
        "cv_mean_accuracy": float(np.mean(accs)), "cv_std_accuracy": float(np.std(accs)),
        "classification_report": report_dict, "confusion_matrix": cm.tolist(),
        "confusion_matrix_label_order": [moods[l] for l in cm_labels],
    }, "movie_mood_classifier_eval.json")


def _runtime_lookup():
    """Movie runtime (minutes) isn't in movies_final.csv -- it got dropped when
    build_datasets.py originally trimmed columns. Recover it from the raw
    source file (data/movie_dataset2/Movies (1970-2023).csv) so the frontend
    can show real runtimes instead of always displaying 'N/A'."""
    import re
    raw_path = os.path.join(DATA_DIR, "movie_dataset2", "Movies (1970-2023).csv")
    raw = pd.read_csv(raw_path)

    def parse_runtime(x):
        if pd.isna(x):
            return None
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None

    raw["runtime_min"] = raw["runtime"].apply(parse_runtime)
    lookup = raw[["movie title", "year", "runtime_min"]].rename(
        columns={"movie title": "title", "year": "release_year"})
    return lookup.drop_duplicates(subset=["title", "release_year"])


# ============================================================================
# 6. APPLY TRAINED MODELS to the app's live datasets
# ============================================================================
def apply_ml_labels():
    # Movies only: the legacy music_final.csv re-scoring (and its local
    # K-Means, music_local_kmeans.pkl) was retired in Phase 2H.4. The music
    # catalog is built by train_music_278k.py.
    print("\n=== Applying the trained movie model to data/movies_final.csv ===")
    with open(os.path.join(MODELS_DIR, "movie_mood_classifier.pkl"), "rb") as f:
        movie_art = pickle.load(f)

    # ---- MOVIES ----
    movies_backup_path = os.path.join(DATA_DIR, "movies_final_rulebased_backup.csv")
    movies_source = movies_backup_path if os.path.exists(movies_backup_path) else os.path.join(DATA_DIR, "movies_final.csv")
    movies = pd.read_csv(movies_source)
    if "runtime_min" not in movies.columns:
        movies = movies.merge(_runtime_lookup(), on=["title", "release_year"], how="left")
    movies_backup = movies.copy()
    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].fillna("")
    text = (movies["overview"].astype(str) + " " + movies["genres"].astype(str)).tolist()

    mvec, mclf, mmoods = movie_art["vectorizer"], movie_art["model"], movie_art["moods"]
    Xm = mvec.transform(text)
    proba_m = mclf.predict_proba(Xm)
    pred_idx = proba_m.argmax(axis=1)
    movies["mood_ml"] = [mmoods[i] for i in pred_idx]
    movies["mood_confidence"] = np.round(proba_m[np.arange(len(movies)), pred_idx], 3)
    for i, mood_name in enumerate(mmoods):
        movies[f"proba_{mood_name}"] = np.round(proba_m[:, i], 3)
    movies["mood_tag_rulebased"] = movies["mood_tag"]
    movies["mood_tag"] = movies["mood_ml"]

    agreement_m = (movies["mood_tag_rulebased"] == movies["mood_ml"]).mean()
    print(f"Movies: {len(movies)} movies | ML-vs-rule-based agreement: {agreement_m:.1%}")
    print(movies["mood_ml"].value_counts())

    movies_backup.to_csv(movies_backup_path, index=False)
    movies.to_csv(os.path.join(DATA_DIR, "movies_final.csv"), index=False)
    print("\nDone. data/movies_final.csv now carries ML-predicted moods; the original")
    print("rule-based labels are preserved in movies_final_rulebased_backup.csv and the")
    print("mood_tag_rulebased column for comparison.")


_TEXT_VOCAB = None
_TEXT_HOLDOUT = None
_TEXT_EVERYDAY = {}
_TEXT_SHARED_CONTEXTS = []

if __name__ == "__main__":
    from text_vocab import (TEXT_VOCAB as _TEXT_VOCAB, TEXT_HOLDOUT as _TEXT_HOLDOUT,
                            TEXT_EVERYDAY as _TEXT_EVERYDAY, TEXT_SHARED_CONTEXTS as _TEXT_SHARED_CONTEXTS)
    t0 = time.time()
    dataset_path = build_text_mood_dataset()
    train_text_classifier(dataset_path)
    if "--text-only" not in sys.argv[1:]:     # --text-only leaves the movie model and data untouched
        train_movie_classifier()
        apply_ml_labels()
    print(f"\nAll done in {time.time()-t0:.1f}s. Trained models are in models/, "
          f"evaluation reports are the *_eval.json files next to them.")
