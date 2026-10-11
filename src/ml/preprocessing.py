"""
Lightweight, dependency-free implementations of the preprocessing utilities
you'd normally reach for in scikit-learn. Implemented directly on top of
numpy/pandas so the project has zero fragile external ML dependencies.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd


class StandardScaler:
    """Zero-mean, unit-variance scaling. API mirrors sklearn.preprocessing.StandardScaler."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self.scale_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split(*arrays, test_size=0.2, random_state=42, stratify=None):
    n = len(arrays[0])
    rng = np.random.RandomState(random_state)

    if stratify is not None:
        stratify = np.asarray(stratify)
        idx_train, idx_test = [], []
        for cls in np.unique(stratify):
            cls_idx = np.where(stratify == cls)[0]
            rng.shuffle(cls_idx)
            n_test = max(1, int(round(len(cls_idx) * test_size)))
            idx_test.extend(cls_idx[:n_test])
            idx_train.extend(cls_idx[n_test:])
        idx_train = np.array(idx_train)
        idx_test = np.array(idx_test)
        rng.shuffle(idx_train)
        rng.shuffle(idx_test)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = int(round(n * test_size))
        idx_test = idx[:n_test]
        idx_train = idx[n_test:]

    out = []
    for a in arrays:
        if isinstance(a, (pd.DataFrame, pd.Series)):
            out.append(a.iloc[idx_train])
            out.append(a.iloc[idx_test])
        else:
            a = np.asarray(a)
            out.append(a[idx_train])
            out.append(a[idx_test])
    return out


def k_fold_indices(n, k=5, random_state=42, stratify=None):
    rng = np.random.RandomState(random_state)
    if stratify is not None:
        stratify = np.asarray(stratify)
        folds = [[] for _ in range(k)]
        for cls in np.unique(stratify):
            cls_idx = np.where(stratify == cls)[0]
            rng.shuffle(cls_idx)
            splits = np.array_split(cls_idx, k)
            for i, s in enumerate(splits):
                folds[i].extend(s.tolist())
        folds = [np.array(f) for f in folds]
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, k)

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


_TOKEN_RE = re.compile(r"[a-zA-Z']+")

_STOPWORDS = set("""
a an the and or but if while of at by for with about against between into
through during before after above below to from up down in out on off over
under again further then once here there all any both each few more most
other some such no nor not only own same so than too very s t can will just
don should now is are was were be been being i me my we our you your it its
this that these those do does did having have has had am
""".split())


def simple_tokenize(text: str):
    if not isinstance(text, str):
        return []
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


class TfidfVectorizer:
    """Minimal TF-IDF vectorizer (unigrams + bigrams), API mirrors sklearn's."""

    def __init__(self, max_features=4000, ngram_range=(1, 2), min_df=2):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vocabulary_ = {}
        self.idf_ = None

    def _ngrams(self, tokens):
        grams = []
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(tokens) - n + 1):
                grams.append(" ".join(tokens[i:i + n]))
        return grams

    def _analyze(self, text):
        return self._ngrams(simple_tokenize(text))

    def fit(self, docs):
        df_counts = {}
        for doc in docs:
            grams = set(self._analyze(doc))
            for g in grams:
                df_counts[g] = df_counts.get(g, 0) + 1

        n_docs = len(docs)
        candidates = {g: c for g, c in df_counts.items() if c >= self.min_df}
        ranked = sorted(candidates.items(), key=lambda kv: -kv[1])
        top = ranked[: self.max_features]

        self.vocabulary_ = {g: i for i, (g, _) in enumerate(top)}
        idf = np.zeros(len(self.vocabulary_))
        for g, i in self.vocabulary_.items():
            idf[i] = np.log((1 + n_docs) / (1 + df_counts[g])) + 1
        self.idf_ = idf
        return self

    def transform(self, docs) -> np.ndarray:
        n = len(docs)
        X = np.zeros((n, len(self.vocabulary_)), dtype=np.float32)
        for row, doc in enumerate(docs):
            grams = self._analyze(doc)
            if not grams:
                continue
            counts = {}
            for g in grams:
                if g in self.vocabulary_:
                    counts[g] = counts.get(g, 0) + 1
            if not counts:
                continue
            max_count = max(counts.values())
            for g, c in counts.items():
                j = self.vocabulary_[g]
                tf = 0.5 + 0.5 * (c / max_count)
                X[row, j] = tf * self.idf_[j]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, docs) -> np.ndarray:
        return self.fit(docs).transform(docs)
