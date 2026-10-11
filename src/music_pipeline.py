"""
Offline music pipeline for the 278k labeled Spotify dataset.

Turns data/music_dataset2/278k_labelled_uri.csv (read-only) into a catalog
keyed by Spotify track ID, and trains the 4-label audio-feature mood
classifier (0=sad, 1=happy, 2=energetic, 3=calm) on it. The catalog carries
the dataset's ground-truth label plus the classifier's per-label
probabilities (`proba_<label>` columns), which src/recommend.py ranks on,
and the derived-mood columns (`derived_<mood>`, `derived_<mood>_score`) for
app moods the dataset has no label for (see DERIVED_MUSIC_MOODS). Derived
moods are feature rules layered on top; they never replace the labels.

Every structural assumption about the source (URI format, unique IDs, no
missing features, labels 0-3) is validated and fails loudly -- the pipeline
never silently drops or dedupes rows. Driven by train_music_278k.py.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from .config import DERIVED_MUSIC_MOODS, LABEL_TO_MOOD
from .ml.models import SoftmaxRegression
from .ml.preprocessing import StandardScaler

AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo']
LABELS = sorted(LABEL_TO_MOOD)  # [0, 1, 2, 3]
PROBA_COLUMNS = [f"proba_{LABEL_TO_MOOD[label]}" for label in LABELS]

TRACK_URI_PREFIX = "spotify:track:"
_TRACK_ID_RE = re.compile(r"^[0-9A-Za-z]{22}$")

_SOURCE_COLUMNS = ["uri", "labels", "duration (ms)"] + AUDIO_FEATURES

# Same hyperparameters the existing audio classifier in train_all_models.py
# uses, so the retrained model is directly comparable to it.
CLASSIFIER_PARAMS = dict(n_classes=len(LABELS), lr=0.7, epochs=120, l2=1e-4, batch_size=2048, random_state=42)


class PipelineValidationError(ValueError):
    """The 278k source data violates an assumption the pipeline relies on."""


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(path: str) -> pd.DataFrame:
    """Read only the columns the pipeline needs from the 278k source CSV."""
    return pd.read_csv(path, usecols=_SOURCE_COLUMNS)


def build_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the raw 278k rows and return one row per Spotify track with
    columns: track_id, label, mood, duration_ms, <AUDIO_FEATURES>.
    Row order matches the source. Raises PipelineValidationError instead of
    dropping anything.
    """
    missing_cols = [c for c in _SOURCE_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise PipelineValidationError(f"Source is missing columns: {missing_cols}")

    null_counts = raw[_SOURCE_COLUMNS].isna().sum()
    if null_counts.any():
        raise PipelineValidationError(f"Source has missing values: {null_counts[null_counts > 0].to_dict()}")

    uri = raw["uri"].astype(str)
    bad_prefix = ~uri.str.startswith(TRACK_URI_PREFIX)
    if bad_prefix.any():
        raise PipelineValidationError(f"{int(bad_prefix.sum())} URIs are not '{TRACK_URI_PREFIX}<id>', "
                                      f"e.g. {uri[bad_prefix].iloc[0]!r}")
    track_id = uri.str.slice(len(TRACK_URI_PREFIX))
    bad_id = ~track_id.str.match(_TRACK_ID_RE)
    if bad_id.any():
        raise PipelineValidationError(f"{int(bad_id.sum())} malformed Spotify track IDs, "
                                      f"e.g. {track_id[bad_id].iloc[0]!r}")
    dup = track_id.duplicated()
    if dup.any():
        raise PipelineValidationError(f"{int(dup.sum())} duplicate Spotify track IDs, "
                                      f"e.g. {track_id[dup].iloc[0]!r}")

    labels = raw["labels"]
    bad_label = ~labels.isin(LABELS)
    if bad_label.any():
        raise PipelineValidationError(f"Unexpected labels {sorted(labels[bad_label].unique().tolist())}; "
                                      f"expected {LABELS}")

    df = pd.DataFrame({
        "track_id": track_id.values,
        "label": labels.astype(int).values,
        "mood": labels.map(LABEL_TO_MOOD).values,
        "duration_ms": raw["duration (ms)"].astype(int).values,
    })
    for feature in AUDIO_FEATURES:
        df[feature] = raw[feature].astype(float).values
    return df


def dataset_report(df: pd.DataFrame) -> dict:
    """Data-quality measurements for a frame from build_training_frame."""
    class_counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    return {
        "rows": int(len(df)),
        "unique_track_ids": int(df["track_id"].nunique()),
        "duplicate_track_ids": int(df["track_id"].duplicated().sum()),
        "missing_feature_values": {f: int(df[f].isna().sum()) for f in AUDIO_FEATURES},
        "class_distribution": {
            f"{label}={LABEL_TO_MOOD[label]}": {"count": int(n), "share": round(float(n) / len(df), 4)}
            for label, n in class_counts.items()
        },
        # Legitimate Spotify values, but worth knowing about: tempo 0 means no
        # beat was detected; rows sharing a feature vector are usually the same
        # recording released under several track IDs.
        "tracks_with_tempo_zero": int((df["tempo"] == 0).sum()),
        "tracks_with_positive_loudness": int((df["loudness"] > 0).sum()),
        "tracks_sharing_identical_feature_vector": int(df.duplicated(subset=AUDIO_FEATURES, keep=False).sum()),
    }


def fit_classifier(X: np.ndarray, y: np.ndarray, **overrides):
    """Standardize features and fit the softmax-regression mood classifier."""
    scaler = StandardScaler().fit(X)
    model = SoftmaxRegression(**{**CLASSIFIER_PARAMS, **overrides}).fit(scaler.transform(X), y)
    return scaler, model


def predict_label_proba(df: pd.DataFrame, scaler, model) -> np.ndarray:
    """(n_tracks, 4) label probabilities, columns in LABELS order."""
    return model.predict_proba(scaler.transform(df[AUDIO_FEATURES].values.astype(float)))


def build_catalog(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """
    Attach classifier output to the training frame. `mood` stays the dataset's
    ground-truth label; `predicted_mood` is the classifier's argmax and
    `mood_confidence` is the probability the classifier gives the true label.
    """
    catalog = df.copy()
    for i, col in enumerate(PROBA_COLUMNS):
        catalog[col] = np.round(proba[:, i], 4)
    catalog["predicted_mood"] = [LABEL_TO_MOOD[LABELS[i]] for i in proba.argmax(axis=1)]
    catalog["mood_confidence"] = np.round(proba[np.arange(len(df)), df["label"].map(LABELS.index).values], 4)
    return catalog


_RULE_OPS = {"<=": np.less_equal, ">=": np.greater_equal, "<": np.less, ">": np.greater, "==": np.equal}


def derive_moods(catalog: pd.DataFrame):
    """
    Add `derived_<mood>` (bool: track meets the rule and trips no guard) and
    `derived_<mood>_score` (0-1, higher = rank first) for each
    DERIVED_MUSIC_MOODS entry. Rule percentiles are resolved to concrete
    cutoffs over this catalog (pandas' default linear-interpolation quantile);
    percentile ranks are pandas' rank(pct=True) over the full catalog.
    Returns (catalog, metadata); metadata records the cutoffs, guard counts
    and (for profile ranking) the profile, so everything is reproducible.
    """
    out = catalog.copy()
    pct_rank = {}

    def pct(feature):
        if feature not in pct_rank:
            pct_rank[feature] = out[feature].rank(pct=True).to_numpy()
        return pct_rank[feature]

    metadata = {}
    for mood, spec in DERIVED_MUSIC_MOODS.items():
        rule_mask = np.ones(len(out), dtype=bool)
        rule = []
        for feature, op, percentile in spec["rule"]:
            cutoff = float(out[feature].quantile(percentile))
            rule_mask &= _RULE_OPS[op](out[feature].to_numpy(dtype=float), cutoff)
            rule.append({"feature": feature, "op": op, "percentile": percentile, "cutoff": cutoff})

        mask = rule_mask.copy()
        guards = {}
        for name, checks in spec.get("guards", {}).items():
            tripped = np.zeros(len(out), dtype=bool)
            for feature, op, value in checks:
                tripped |= _RULE_OPS[op](out[feature].to_numpy(dtype=float), value)
            mask &= ~tripped
            guards[name] = {"checks": [{"feature": f, "op": o, "value": v} for f, o, v in checks],
                            "tracks_in_catalog": int(tripped.sum()),
                            "rule_matches_excluded": int((tripped & rule_mask).sum())}

        if "profile" in spec:
            profile = {f: float(np.median(pct(f)[mask])) for f in spec["profile"]}
            score = 1 - np.mean([np.abs(pct(f) - center) for f, center in profile.items()], axis=0)
            score_meta = {"method": "profile closeness", "profile_percentiles": profile,
                          "definition": "1 - mean absolute distance between the track's percentile ranks and "
                                        "the median percentile ranks of the guarded candidates"}
        else:
            score = np.mean([pct(f) if d > 0 else 1 - pct(f) for f, d in spec["score"].items()], axis=0)
            score_meta = {"method": "extreme", "features": spec["score"],
                          "definition": "mean of percentile ranks, inverted where direction is -1"}

        out[f"derived_{mood}"] = mask
        out[f"derived_{mood}_score"] = np.round(score, 6)
        metadata[mood] = {
            "experimental": spec["experimental"],
            "note": spec["note"],
            "rule": rule,
            "quantile_method": "pandas.Series.quantile (linear interpolation) over the full catalog",
            "guards": guards,
            "score": score_meta,
            "rule_match_count": int(rule_mask.sum()),
            "candidate_count": int(mask.sum()),
            "candidate_true_label_mix": {m: int(n) for m, n in out.loc[mask, "mood"].value_counts().items()},
        }
    return out, metadata
