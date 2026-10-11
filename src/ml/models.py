"""
Core supervised + unsupervised ML models implemented from scratch on numpy.
No external ML dependency (scikit-learn/xgboost) required to train or run
these models — everything here is plain linear algebra / gradient descent.
"""
from __future__ import annotations
import numpy as np


class SoftmaxRegression:
    """Multinomial logistic regression trained with mini-batch gradient descent + L2 reg."""

    def __init__(self, n_classes, lr=0.5, epochs=300, l2=1e-4, batch_size=512, random_state=42,
                 verbose=False, class_weight=None):
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight  # None or "balanced"
        self.W = None
        self.b = None
        self.loss_history_ = []

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / expZ.sum(axis=1, keepdims=True)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        rng = np.random.RandomState(self.random_state)
        self.W = rng.normal(0, 0.01, size=(d, self.n_classes))
        self.b = np.zeros(self.n_classes)

        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y] = 1

        if sample_weight is None and self.class_weight == "balanced":
            counts = np.bincount(y, minlength=self.n_classes).astype(float)
            counts[counts == 0] = 1
            class_w = n / (self.n_classes * counts)
            sample_weight = class_w[y]
        if sample_weight is None:
            sample_weight = np.ones(n)
        sample_weight = np.asarray(sample_weight, dtype=np.float64)

        idx_all = np.arange(n)
        for epoch in range(self.epochs):
            rng.shuffle(idx_all)
            for start in range(0, n, self.batch_size):
                batch = idx_all[start:start + self.batch_size]
                Xb, Yb, wb = X[batch], Y[batch], sample_weight[batch]
                Z = Xb @ self.W + self.b
                P = self._softmax(Z)
                err = (P - Yb) * wb[:, None]
                grad_W = Xb.T @ err / len(batch) + self.l2 * self.W
                grad_b = err.mean(axis=0)
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

            if self.verbose and epoch % 50 == 0:
                P_full = self._softmax(X @ self.W + self.b)
                loss = -np.mean(np.sum(Y * np.log(P_full + 1e-12), axis=1))
                self.loss_history_.append(loss)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier, used as a fast interpretable baseline."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.mean_ = {}
        self.var_ = {}
        self.priors_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.mean_[c] = Xc.mean(axis=0)
            self.var_[c] = Xc.var(axis=0) + 1e-6
            self.priors_[c] = len(Xc) / len(X)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_probs = np.zeros((len(X), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            mean, var = self.mean_[c], self.var_[c]
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(
                ((X - mean) ** 2) / var, axis=1
            )
            log_probs[:, i] = log_likelihood + np.log(self.priors_[c])
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return self.classes_[idx]


class KMeans:
    """Lloyd's algorithm K-Means with k-means++ initialization."""

    def __init__(self, n_clusters=6, n_init=8, max_iter=200, random_state=42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.inertia_ = None

    def _kmeans_plus_plus(self, X, rng):
        n = X.shape[0]
        centers = [X[rng.randint(n)]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centers], axis=0
            )
            probs = dist_sq / dist_sq.sum()
            next_idx = rng.choice(n, p=probs)
            centers.append(X[next_idx])
        return np.array(centers)

    def _fit_once(self, X, rng):
        centers = self._kmeans_plus_plus(X, rng)
        labels = np.zeros(len(X), dtype=int)
        for _ in range(self.max_iter):
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                labels = new_labels
                break
            labels = new_labels
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    centers[k] = X[mask].mean(axis=0)
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        inertia = np.sum(dists[np.arange(len(X)), labels] ** 2)
        return centers, labels, inertia

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        best = None
        for i in range(self.n_init):
            centers, labels, inertia = self._fit_once(X, np.random.RandomState(self.random_state + i))
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia)
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return dists.argmin(axis=1)


class NearestNeighbors:
    """Brute-force cosine-similarity nearest neighbor search (fine at this dataset scale)."""

    def __init__(self, metric="cosine"):
        self.metric = metric
        self.X_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        self.X_ = X
        return self

    def kneighbors(self, query, n_neighbors=10):
        query = np.asarray(query, dtype=float).reshape(1, -1)
        if self.metric == "cosine":
            n = np.linalg.norm(query, axis=1, keepdims=True)
            n[n == 0] = 1.0
            query = query / n
            sims = self.X_ @ query.T  # higher = more similar
            order = np.argsort(-sims.ravel())[:n_neighbors]
            return order, sims.ravel()[order]
        else:
            dists = np.linalg.norm(self.X_ - query, axis=1)
            order = np.argsort(dists)[:n_neighbors]
            return order, dists[order]
