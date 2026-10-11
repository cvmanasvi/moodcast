"""
Prepare the local dataset behind the Popular Artists music mode.

Source: "Spotify Tracks Dataset (Updated every week)" by Gautham Vijayaraj,
Kaggle, version 2 (2024-12-09), file spotify_tracks.csv, licensed by its
author under Apache 2.0. The author does not document how the data was
collected; its fields are Spotify track data, which may be governed by
Spotify's terms separately from that licence (see DATA.md). The raw and
prepared files are LOCAL ONLY (data/external/ is git-ignored) and are never
committed or redistributed.

Run from the project root, with ONE of:
    python prepare_popular_artists.py                     # file already in place
    python prepare_popular_artists.py --source PATH       # a manually downloaded spotify_tracks.csv or .zip
    python prepare_popular_artists.py --download          # fetch the pinned version from Kaggle (opt-in)

Every input must match EXPECTED_SHA256 exactly or nothing is written.

Reads   data/external/popular_artists_2024/spotify_tracks.csv (raw, verified)
        config/popular_artists.json (curated artists)
        models/music_mood_classifier_278k.pkl (existing mood classifier)
        models/music_mood_classifier_278k_eval.json (recorded angry/stressed cutoffs)
        data/music_catalog_278k.csv (reference distribution for derived-mood scores)
Writes  data/external/popular_artists_2024/popular_tracks.csv   LOCAL ONLY
        data/external/popular_artists_2024/manifest.json        aggregates only, no track ids

Steps: strict schema check (exact columns, numeric fields); rows are dropped
for malformed ids, a track_url that is not exactly
https://open.spotify.com/track/<track_id>, placeholder or out-of-range
values (the source marks missing values -1), tracks under a minute
(intros, interludes, snippets), spoken-word or failed audio
analysis (the guards the 278k stressed rule uses), duplicate ids (exact
copies kept once; conflicting copies dropped entirely), and artists not in
the curated list (whole comma-separated entries only, never a partial
name). Versions of one recording (same artist string, same title, same
duration to the second -- e.g. the same song on many compilation albums)
are reduced to the most popular. The existing 278k classifier then scores
happy/sad/energetic/calm, and the existing angry/stressed rules are applied
with the cutoffs recorded for the 278k catalog (never recomputed here).
Mood assignments are model-derived, not human-validated.
"""
import argparse
import datetime
import json
import os
import pickle
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

from src import music_pipeline as mp
from src import popular_artists as pa
from src.config import BASE_DIR, DERIVED_MUSIC_MOODS, LABEL_TO_MOOD, MODELS_DIR

SOURCE = {
    "name": "Spotify Tracks Dataset (Updated every week)",
    "creator": "Gautham Vijayaraj",
    "url": "https://www.kaggle.com/datasets/gauthamvijayaraj/spotify-tracks-dataset-updated-every-week",
    "version": 2,
    "version_date": "2024-12-09",
    "file": "spotify_tracks.csv",
    "license": "Apache-2.0 (as stated by the dataset author on Kaggle)",
    "collection_method": "not documented by the author",
}
DOWNLOAD_URL = ("https://www.kaggle.com/api/v1/datasets/download/gauthamvijayaraj/"
                "spotify-tracks-dataset-updated-every-week?datasetVersionNumber=2")
EXPECTED_SHA256 = "2bdeea651ed626f89c034a1bb0c343ad47350ad852950c575f941f38da8d5f9d"
RAW_PATH = os.path.join(pa.DATA_DIR, SOURCE["file"])
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "music_mood_classifier_278k.pkl")
EVAL_PATH = os.path.join(MODELS_DIR, "music_mood_classifier_278k_eval.json")

EXPECTED_COLUMNS = [
    "track_id", "track_name", "artist_name", "year", "popularity", "artwork_url", "album_name",
    "acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
    "loudness", "mode", "speechiness", "tempo", "time_signature", "valence", "track_url", "language",
]
TEXT_COLUMNS = ["track_id", "track_name", "artist_name", "artwork_url", "album_name", "track_url", "language"]
NUMERIC_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in TEXT_COLUMNS]
UNIT_FEATURES = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "speechiness", "valence"]
# (min, max) a real value can take; anything else (including the -1 and -100000 placeholders) is rejected
RANGES = {**{f: (0.0, 1.0) for f in UNIT_FEATURES},
          # under a minute is an intro/interlude/snippet, not a song for this mode
          "duration_ms": (60_000.0, 3_600_000.0), "loudness": (-60.0, 5.0), "tempo": (0.0, 260.0),
          "popularity": (0, 100), "year": (1900, 2025), "key": (0, 11), "mode": (0, 1), "time_signature": (1, 7)}
# not used: artwork_url (artwork comes only from the app's approved oEmbed lookup), album_name, language


class PreparationError(Exception):
    """Input that must not be used; nothing is written."""


# ---------------------------------------------------------------- acquiring the verified raw file

def verify_sha256(path):
    digest = mp.file_sha256(path)
    if digest != EXPECTED_SHA256:
        raise PreparationError(f"{os.path.basename(path)} has SHA-256 {digest}, expected {EXPECTED_SHA256}. "
                               "It is not the pinned dataset version; nothing was written.")
    return digest


def _install(candidate):
    """Verify a candidate CSV, then move it into place (the existing file is replaced only on success)."""
    verify_sha256(candidate)
    os.makedirs(pa.DATA_DIR, exist_ok=True)
    os.replace(candidate, RAW_PATH)


def _extract_csv(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path) as zf:
        if SOURCE["file"] not in zf.namelist():
            raise PreparationError(f"{os.path.basename(zip_path)} does not contain {SOURCE['file']}")
        return zf.extract(SOURCE["file"], dest_dir)


def import_source(path):
    """Install a manually downloaded spotify_tracks.csv or Kaggle .zip after verifying its hash."""
    os.makedirs(pa.DATA_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=pa.DATA_DIR) as tmp:
        if zipfile.is_zipfile(path):
            candidate = _extract_csv(path, tmp)
        else:
            candidate = shutil.copyfile(path, os.path.join(tmp, SOURCE["file"]))
        _install(candidate)


def download():
    """Fetch the pinned version from Kaggle (opt-in) and install it after verifying its hash."""
    import requests
    os.makedirs(pa.DATA_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=pa.DATA_DIR) as tmp:
        archive = os.path.join(tmp, "download.zip")
        with requests.get(DOWNLOAD_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(archive, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
        if not zipfile.is_zipfile(archive):
            raise PreparationError("The download is not a zip archive (Kaggle may require signing in); "
                                   "download it manually and use --source.")
        _install(_extract_csv(archive, tmp))


# ---------------------------------------------------------------- validation

def load_raw(path):
    """The raw CSV with the exact expected schema; text kept verbatim, numeric columns strictly numeric."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(raw.columns) != EXPECTED_COLUMNS:
        raise PreparationError(f"Unexpected columns {list(raw.columns)}; expected {EXPECTED_COLUMNS}")
    for col in NUMERIC_COLUMNS:
        values = pd.to_numeric(raw[col], errors="coerce")
        bad = values.isna()
        if bad.any():
            raise PreparationError(f"Column {col!r} has {int(bad.sum())} non-numeric values")
        raw[col] = values
    return raw


def validate(raw, counts):
    """Rows that pass every check, with the number dropped per reason recorded in `counts`."""
    df = raw
    checks = [
        ("empty_text", (df["track_name"].str.strip() == "") | (df["artist_name"].str.strip() == "")),
        ("malformed_track_id", ~df["track_id"].str.fullmatch(r"[0-9A-Za-z]{22}")),
        ("track_url_mismatch", df["track_url"] != pa.SPOTIFY_TRACK_URL + df["track_id"]),
    ]
    out_of_range = np.zeros(len(df), dtype=bool)
    for col, (lo, hi) in RANGES.items():
        out_of_range |= ~df[col].between(lo, hi).to_numpy()
    checks.append(("placeholder_or_out_of_range", out_of_range))
    guards = DERIVED_MUSIC_MOODS["stressed"]["guards"]
    for name in ("failed_audio_analysis", "spoken_word"):
        tripped = np.zeros(len(df), dtype=bool)
        for feature, op, value in guards[name]:
            tripped |= mp._RULE_OPS[op](df[feature].to_numpy(dtype=float), value)
        checks.append((name, tripped))
    keep = np.ones(len(df), dtype=bool)
    for reason, bad in checks:
        bad = np.asarray(bad, dtype=bool) & keep
        counts[f"dropped_{reason}"] = int(bad.sum())
        keep &= ~bad
    df = df[keep]

    # duplicate ids: exact copies are kept once; copies that disagree are all dropped
    dup = df["track_id"].duplicated(keep=False)
    conflicting = set(df[dup].groupby("track_id").filter(lambda g: len(g.drop_duplicates()) > 1)["track_id"])
    counts["dropped_conflicting_duplicate_ids"] = int(df["track_id"].isin(conflicting).sum())
    df = df[~df["track_id"].isin(conflicting)]
    before = len(df)
    df = df.drop_duplicates()
    counts["dropped_exact_duplicate_rows"] = before - len(df)
    assert df["track_id"].is_unique
    return df


def match_curated(df, config, counts):
    """Rows by at least one curated artist, with the matched names and the first one's category."""
    categories = pa.artist_categories(config)
    names = list(categories)
    matched = df["artist_name"].map(lambda s: pa.match_artists(s, names))
    df = df.assign(matched_artists=matched)
    keep = matched.map(len) > 0
    counts["dropped_not_curated_artist"] = int((~keep).sum())
    df = df[keep].copy()
    df["category"] = df["matched_artists"].map(lambda m: categories[m[0]])
    return df


def _title_key(title):
    return " ".join(str(title).casefold().split())


def dedupe_versions(df, counts):
    """One row per (artist string, title, duration to the second): the most popular, then the lowest id."""
    key = (df["artist_name"].map(_title_key) + "\x1f" + df["track_name"].map(_title_key) + "\x1f"
           + (df["duration_ms"] / 1000).round().astype(int).astype(str))
    ordered = df.assign(_key=key).sort_values(["_key", "popularity", "track_id"], ascending=[True, False, True])
    out = ordered.drop_duplicates("_key").drop(columns="_key")
    counts["dropped_duplicate_versions"] = len(df) - len(out)
    return out


# ---------------------------------------------------------------- mood scoring (existing model and rules)

def recorded_derived_rules(catalog):
    """
    The angry/stressed cutoffs recorded for the 278k catalog. Checked against
    config.DERIVED_MUSIC_MOODS and the catalog itself so a stale record can't
    be used; never recomputed on the new dataset.
    """
    with open(EVAL_PATH, encoding="utf-8") as f:
        recorded = json.load(f)["derived_moods"]
    rules = {}
    for mood, spec in DERIVED_MUSIC_MOODS.items():
        rec = recorded[mood]["rule"]
        if [(r["feature"], r["op"], r["percentile"]) for r in rec] != [tuple(x) for x in spec["rule"]]:
            raise PreparationError(f"Recorded {mood} rule does not match config.DERIVED_MUSIC_MOODS")
        for r in rec:
            if abs(float(catalog[r["feature"]].quantile(r["percentile"])) - r["cutoff"]) > 5e-4:
                raise PreparationError(f"Recorded {mood} cutoff for {r['feature']} does not match the 278k catalog")
        rules[mood] = {"rule": [(r["feature"], r["op"], float(r["cutoff"])) for r in rec],
                       "guards": spec.get("guards", {}), "score": recorded[mood]["score"]}
    return rules


def _reference_pct(reference, values):
    """Percentile rank of each value within the 278k catalog's distribution (average rank for ties)."""
    ref = np.sort(reference)
    lo, hi = np.searchsorted(ref, values, "left"), np.searchsorted(ref, values, "right")
    return (lo + hi) / 2 / len(ref)


def score_moods(df, catalog):
    """Classifier probabilities plus derived angry/stressed membership and scores."""
    with open(CLASSIFIER_PATH, "rb") as f:
        bundle = pickle.load(f)
    if list(bundle["features"]) != mp.AUDIO_FEATURES:
        raise PreparationError("Classifier features do not match music_pipeline.AUDIO_FEATURES")
    proba = mp.predict_label_proba(df, bundle["scaler"], bundle["model"])
    out = df.copy()
    for i, col in enumerate(mp.PROBA_COLUMNS):
        out[col] = np.round(proba[:, i], 6)
    rules = recorded_derived_rules(catalog)
    pct = {}
    for mood, spec in rules.items():
        mask = np.ones(len(out), dtype=bool)
        for feature, op, cutoff in spec["rule"]:
            mask &= mp._RULE_OPS[op](out[feature].to_numpy(dtype=float), cutoff)
        for checks in spec["guards"].values():
            for feature, op, value in checks:
                mask &= ~mp._RULE_OPS[op](out[feature].to_numpy(dtype=float), value)
        for f in set(spec["score"].get("features", {})) | set(spec["score"].get("profile_percentiles", {})):
            pct.setdefault(f, _reference_pct(catalog[f].to_numpy(dtype=float), out[f].to_numpy(dtype=float)))
        if "profile_percentiles" in spec["score"]:
            prof = spec["score"]["profile_percentiles"]
            score = 1 - np.mean([np.abs(pct[f] - c) for f, c in prof.items()], axis=0)
        else:
            score = np.mean([pct[f] if d > 0 else 1 - pct[f] for f, d in spec["score"]["features"].items()], axis=0)
        out[f"derived_{mood}"] = mask
        out[f"derived_{mood}_score"] = np.round(score, 6)
    return out, rules


# ---------------------------------------------------------------- build

def coverage(df):
    """Eligible tracks per app mood ("Match my mood") and category, as the runtime defines eligibility."""
    out = {}
    for mood, targets in (("happy", "happy"), ("sad", "sad"), ("energetic", "energetic"),
                          ("chill", "calm"), ("angry", "angry"), ("stressed", "stressed")):
        member, _ = pa.target_candidates(df, targets)
        sub = df[member]
        out[mood] = {"tracks": int(len(sub)), "artists": int(sub["matched_artists"].explode().nunique()),
                     "by_category": {k: int(v) for k, v in sub["category"].value_counts().sort_index().items()}}
    return out


def build(raw_path=RAW_PATH, config_path=None, catalog=None):
    """(prepared frame, manifest dict) from a verified raw file; raises PreparationError on bad input."""
    raw_sha = verify_sha256(raw_path)
    config = pa.load_config(config_path)
    counts = {}
    raw = load_raw(raw_path)
    counts["raw_rows"] = len(raw)
    df = validate(raw, counts)
    df = match_curated(df, config, counts)
    df = dedupe_versions(df, counts)
    if catalog is None:
        from src.data_loader import load_music_catalog
        catalog = load_music_catalog()
    df, rules = score_moods(df, catalog)
    df = df.sort_values("track_id").reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["popularity"] = df["popularity"].astype(int)
    df["duration_ms"] = df["duration_ms"].astype(int)
    counts["prepared_rows"] = len(df)
    prepared = df.assign(matched_artists=df["matched_artists"].map(json.dumps))[pa.PREPARED_COLUMNS]

    found = df["matched_artists"].explode().value_counts()
    artists = {cat: {name: int(found.get(name, 0)) for name in names} for cat, names in config["categories"].items()}
    manifest = {
        "generated_on": datetime.date.today().isoformat(),
        "local_only": True,
        "contains_track_ids": False,
        "source": SOURCE,
        "raw_sha256": raw_sha,
        "inputs_sha256": {
            "config/popular_artists.json": mp.file_sha256(config_path or pa.CONFIG_PATH),
            "models/music_mood_classifier_278k.pkl": mp.file_sha256(CLASSIFIER_PATH),
            "models/music_mood_classifier_278k_eval.json": mp.file_sha256(EVAL_PATH),
        },
        "counts": counts,
        "curated_artists_tracks": artists,
        "curated_artists_missing": [n for names in artists.values() for n, c in names.items() if c == 0],
        "mood_method": {
            "base_moods": f"existing 278k classifier probability >= {pa.CONFIDENCE_MIN} for happy, sad, energetic, "
                          f"calm (chill); calm also excludes speechiness > {pa.recommend.CALM_CELL_SPEECHINESS_MAX}",
            "derived_moods": {m: {"rule": [{"feature": f, "op": o, "cutoff": c} for f, o, c in r["rule"]],
                                  "guards": sorted(r["guards"])} for m, r in rules.items()},
            "cutoffs_source": "models/music_mood_classifier_278k_eval.json (278k catalog; not recomputed)",
            "validation": "model-derived, not human-validated",
        },
        "coverage_match_my_mood": coverage(df),
    }
    return prepared, manifest


def write_outputs(prepared, manifest):
    os.makedirs(pa.DATA_DIR, exist_ok=True)
    tmp_csv, tmp_json = pa.PREPARED_PATH + ".tmp", pa.MANIFEST_PATH + ".tmp"
    prepared.to_csv(tmp_csv, index=False)
    manifest = {**manifest, "prepared_sha256": mp.file_sha256(tmp_csv)}
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_csv, pa.PREPARED_PATH)
    os.replace(tmp_json, pa.MANIFEST_PATH)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true", help="fetch the pinned version from Kaggle (network)")
    group.add_argument("--source", metavar="PATH", help="a manually downloaded spotify_tracks.csv or Kaggle .zip")
    args = parser.parse_args(argv)
    try:
        if args.download:
            download()
        elif args.source:
            import_source(args.source)
        elif not os.path.exists(RAW_PATH):
            raise PreparationError(f"{os.path.relpath(RAW_PATH, BASE_DIR)} not found. Download version "
                                   f"{SOURCE['version']} from {SOURCE['url']} and run with --source PATH, "
                                   "or run with --download.")
        prepared, manifest = build()
        manifest = write_outputs(prepared, manifest)
    except PreparationError as exc:
        print(f"Preparation stopped: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {os.path.relpath(pa.PREPARED_PATH, BASE_DIR)} ({manifest['counts']['prepared_rows']} tracks) "
          f"and {os.path.relpath(pa.MANIFEST_PATH, BASE_DIR)}")
    for mood, c in manifest["coverage_match_my_mood"].items():
        print(f"  {mood:9s} {c['tracks']:5d} tracks, {c['artists']:2d} artists, {c['by_category']}")
    if manifest["curated_artists_missing"]:
        print("  curated artists with no tracks:", ", ".join(manifest["curated_artists_missing"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
