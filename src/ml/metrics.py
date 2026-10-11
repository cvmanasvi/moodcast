"""Evaluation metrics implemented on numpy (mirrors sklearn.metrics API/output)."""
from __future__ import annotations
import numpy as np


def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels


def precision_recall_f1(y_true, y_pred, labels=None):
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    out = {}
    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = cm[i, :].sum()
        out[lbl] = dict(precision=precision, recall=recall, f1=f1, support=int(support))
    return out


def classification_report(y_true, y_pred, labels=None, target_names=None):
    scores = precision_recall_f1(y_true, y_pred, labels)
    keys = list(scores.keys())
    names = target_names if target_names else [str(k) for k in keys]
    lines = [f"{'class':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]
    for k, name in zip(keys, names):
        s = scores[k]
        lines.append(f"{name:<14}{s['precision']:>10.3f}{s['recall']:>10.3f}{s['f1']:>10.3f}{s['support']:>10d}")
    acc = accuracy_score(y_true, y_pred)
    lines.append(f"\naccuracy: {acc:.4f}  ({len(y_true)} samples)")
    macro_p = np.mean([s["precision"] for s in scores.values()])
    macro_r = np.mean([s["recall"] for s in scores.values()])
    macro_f1 = np.mean([s["f1"] for s in scores.values()])
    lines.append(f"macro avg:{macro_p:>10.3f}{macro_r:>10.3f}{macro_f1:>10.3f}")
    return "\n".join(lines), dict(accuracy=acc, macro_precision=macro_p, macro_recall=macro_r, macro_f1=macro_f1, per_class=scores)


def silhouette_score_sampled(X, labels, sample_size=3000, random_state=42):
    """Approximate silhouette score computed on a random subsample (full O(n^2) is too slow at scale)."""
    rng = np.random.RandomState(random_state)
    n = len(X)
    if n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X, labels = X[idx], labels[idx]
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    scores = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        own[i] = False
        if not own.any():
            scores[i] = 0
            continue
        a = dists[i, own].mean()
        b = np.inf
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            mask = labels == lbl
            if mask.any():
                b = min(b, dists[i, mask].mean())
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
    return float(scores.mean())


def cross_val_accuracy(model_fn, X, y, k_fold_indices_fn, k=5):
    accs = []
    for train_idx, val_idx in k_fold_indices_fn:
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], preds))
    return float(np.mean(accs)), float(np.std(accs)), accs
