"""Tests for the 278k music pipeline: source validation, the 6 app moods ->
4 real labels + derived moods mapping, target-coverage ranking, and the built
catalog/model artifacts.

The artifact tests need `python train_music_278k.py` to have been run; they
are skipped (not failed) if its outputs are missing."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import flask_app
from src import music_pipeline as mp
from src import recommend
from src.config import (DERIVED_MUSIC_MOODS, LABEL_TO_MOOD, MODELS_DIR, MOODS, MOOD_TO_MUSIC_TARGETS,
                        MUSIC_278K_SOURCE_PATH, MUSIC_CATALOG_278K_PATH)

MODES = ["Match my mood", "Cheer me up"]
EVAL_PATH = os.path.join(MODELS_DIR, "music_mood_classifier_278k_eval.json")
SOURCE_ROWS = 277_938
SOURCE_CLASS_COUNTS = {0: 82_058, 1: 106_429, 2: 47_065, 3: 42_386}
REAL_LABELS = set(LABEL_TO_MOOD.values())


def _track_id(i):
    return f"T{i:021d}"  # 22 alphanumeric chars, like a real Spotify ID


def _raw_rows(n=8, seed=0):
    rng = np.random.RandomState(seed)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:{_track_id(i)}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    return raw


def _synthetic_catalog(proba_rows, popularity=None):
    """Catalog with explicit per-track label probabilities; true labels cycle 0,1,2,3."""
    df = mp.build_training_frame(_raw_rows(len(proba_rows)))
    catalog = mp.build_catalog(df, np.asarray(proba_rows, dtype=float))
    if popularity is not None:
        catalog["popularity"] = popularity
    return catalog


def _random_catalog(n=800, seed=1):
    """Full-shape synthetic catalog: random features, Dirichlet probabilities, derived moods."""
    proba = np.random.RandomState(seed).dirichlet(np.ones(4), size=n)
    catalog, _ = mp.derive_moods(_synthetic_catalog(proba))
    return catalog


# ---- source validation ----

def test_build_training_frame_derives_track_ids_and_keeps_labels():
    df = mp.build_training_frame(_raw_rows())
    assert df["track_id"].tolist() == [_track_id(i) for i in range(8)]
    assert df["label"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert df["mood"].tolist() == ["sad", "happy", "energetic", "calm"] * 2
    assert list(df.columns) == ["track_id", "label", "mood", "duration_ms"] + mp.AUDIO_FEATURES


def _corrupt(kind):
    raw = _raw_rows()
    if kind == "duplicate_id":
        raw.loc[1, "uri"] = raw.loc[0, "uri"]
    elif kind == "bad_prefix":
        raw.loc[0, "uri"] = "spotify:album:" + _track_id(0)
    elif kind == "bad_id":
        raw.loc[0, "uri"] = "spotify:track:short"
    elif kind == "missing_feature":
        raw.loc[0, "energy"] = np.nan
    elif kind == "unknown_label":
        raw.loc[0, "labels"] = 4
    elif kind == "missing_column":
        raw = raw.drop(columns=["valence"])
    return raw


@pytest.mark.parametrize("kind", ["duplicate_id", "bad_prefix", "bad_id", "missing_feature",
                                  "unknown_label", "missing_column"])
def test_build_training_frame_fails_loudly(kind):
    with pytest.raises(mp.PipelineValidationError):
        mp.build_training_frame(_corrupt(kind))


def test_build_catalog_confidence_is_probability_of_true_label():
    catalog = _synthetic_catalog([[0.7, 0.1, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1],
                                  [0.1, 0.1, 0.1, 0.7], [0.1, 0.1, 0.1, 0.7]])
    # true labels are 0,1,2,3
    assert catalog["mood_confidence"].tolist() == [0.7, 0.2, 0.1, 0.7]
    assert catalog["predicted_mood"].tolist() == ["sad", "sad", "calm", "calm"]


# ---- derived moods ----

def test_derive_moods_records_cutoffs_and_never_touches_labels():
    base = _synthetic_catalog(np.full((400, 4), 0.25))
    catalog, meta = mp.derive_moods(base)
    pd.testing.assert_frame_equal(catalog[base.columns], base)  # labels/features unchanged
    assert set(meta) == set(DERIVED_MUSIC_MOODS)
    for mood, info in meta.items():
        expected = _mask_from_metadata(base, info)
        for rule in info["rule"]:
            assert rule["cutoff"] == pytest.approx(base[rule["feature"]].quantile(rule["percentile"]))
        assert (catalog[f"derived_{mood}"].to_numpy() == expected).all()
        assert info["candidate_count"] == int(expected.sum())
        assert catalog[f"derived_{mood}_score"].between(0, 1).all()


_OPS = {"<=": np.less_equal, ">=": np.greater_equal, "<": np.less, ">": np.greater, "==": np.equal}


def _mask_from_metadata(catalog, info):
    """Recompute a derived mood's candidates from its recorded cutoffs and guards only."""
    mask = np.ones(len(catalog), dtype=bool)
    for rule in info["rule"]:
        mask &= _OPS[rule["op"]](catalog[rule["feature"]].to_numpy(dtype=float), rule["cutoff"])
    for guard in info["guards"].values():
        for check in guard["checks"]:
            mask &= ~_OPS[check["op"]](catalog[check["feature"]].to_numpy(dtype=float), check["value"])
    return mask


def test_derived_rules_match_the_approved_definitions():
    angry = DERIVED_MUSIC_MOODS["angry"]
    assert angry["rule"] == [("valence", "<=", 0.25), ("energy", ">=", 0.75), ("loudness", ">=", 0.75)]
    assert angry["score"] == {"valence": -1, "energy": +1, "loudness": +1}
    assert angry["experimental"] is False
    assert "guards" not in angry and "profile" not in angry
    stressed = DERIVED_MUSIC_MOODS["stressed"]
    assert stressed["experimental"] is True
    assert stressed["rule"] == [("valence", "<=", 0.40), ("energy", ">=", 0.50), ("energy", "<", 0.75),
                                ("danceability", "<=", 0.50)]
    assert stressed["guards"] == {
        "failed_audio_analysis": [("danceability", "==", 0.0), ("tempo", "==", 0.0), ("valence", "==", 0.0)],
        "spoken_word": [("speechiness", ">", 0.66)],
    }
    assert stressed["profile"] == ["valence", "energy", "danceability", "loudness"]
    assert "score" not in stressed  # profile closeness replaced the extreme score
    # tempo is never a rule gate or a ranking feature; it only marks failed analysis
    for spec in DERIVED_MUSIC_MOODS.values():
        assert all(f != "tempo" for f, _, _ in spec["rule"])
        assert "tempo" not in spec.get("score", {}) and "tempo" not in spec.get("profile", [])


def test_guarded_tracks_cannot_become_stressed_candidates():
    catalog = _random_catalog(n=4000)
    rows = catalog.index[catalog["derived_stressed"]][:4]
    assert len(rows) == 4
    base = catalog.drop(columns=[c for c in catalog.columns if c.startswith("derived_")])
    base.loc[rows[0], "danceability"] = 0.0
    base.loc[rows[1], "tempo"] = 0.0
    base.loc[rows[2], "valence"] = 0.0
    base.loc[rows[3], "speechiness"] = 0.9
    rederived, meta = mp.derive_moods(base)
    assert not rederived.loc[rows, "derived_stressed"].any()
    guards = meta["stressed"]["guards"]
    assert guards["failed_audio_analysis"]["rule_matches_excluded"] >= 3
    assert guards["spoken_word"]["rule_matches_excluded"] >= 1
    assert meta["stressed"]["candidate_count"] == meta["stressed"]["rule_match_count"] - sum(
        g["rule_matches_excluded"] for g in guards.values())


def test_angry_and_stressed_are_disjoint():
    catalog = _random_catalog(n=2000)
    assert not (catalog["derived_angry"] & catalog["derived_stressed"]).any()


# ---- 6 app moods -> music targets ----

def test_every_app_mood_maps_to_real_labels_or_derived_moods():
    assert set(MOOD_TO_MUSIC_TARGETS) == set(MOODS)
    assert not set(DERIVED_MUSIC_MOODS) & REAL_LABELS  # derived moods are never ML labels
    for mood in MOODS:
        for mode in MODES:
            targets = recommend._target_music_weighted(mood, mode)
            assert targets, (mood, mode)
            assert {t for t, _ in targets} <= REAL_LABELS | set(DERIVED_MUSIC_MOODS)
            assert sum(w for _, w in targets) == pytest.approx(1.0)


@pytest.mark.parametrize("mood,target", [("happy", "happy"), ("sad", "sad"), ("energetic", "energetic"),
                                         ("chill", "calm"), ("angry", "angry"), ("stressed", "stressed")])
def test_match_mode_targets(mood, target):
    assert recommend._target_music_weighted(mood, "Match my mood") == [(target, 1.0)]


# ---- target-coverage ranking ----

@pytest.mark.parametrize("weights", [[1.0], [0.625, 0.375], [0.5, 0.3, 0.2], [1, 1, 1]])
@pytest.mark.parametrize("total", [1, 7, 10, 30, 50, 100])
def test_shares_always_sum_to_total(weights, total):
    shares = recommend._allocate_shares(weights, total)
    assert sum(shares) == total and all(s >= 0 for s in shares)


def test_shares_are_weight_proportional():
    assert recommend._allocate_shares([0.625, 0.375], 30) == [19, 11]
    assert recommend._allocate_shares([0.625, 0.375], 50) == [31, 19]


@pytest.mark.parametrize("top_n", [10, 30, 50])
def test_multi_target_request_respects_shares_without_duplicates(top_n):
    catalog = _random_catalog()
    targets = recommend._target_music_weighted("sad", "Cheer me up")  # happy + calm
    assert [t for t, _ in targets] == ["happy", "calm"]
    out = recommend.recommend_music(catalog, "sad", "Cheer me up", top_n=top_n)
    assert len(out) == top_n
    assert out["track_id"].is_unique
    counts = out["target"].value_counts().to_dict()
    assert counts == dict(zip(["happy", "calm"], recommend._allocate_shares([w for _, w in targets], top_n)))
    # the old weighted sum collapsed to the first target; both must appear, and early
    assert set(out["target"].head(3)) == {"happy", "calm"}


def test_candidates_come_only_from_the_true_label():
    catalog = _random_catalog()
    # give a sad-labelled track the highest happy probability of all
    sad_row = catalog.index[catalog["label"] == 0][0]
    catalog.loc[sad_row, "proba_happy"] = 1.0
    out = recommend._coverage_top(catalog, [("happy", 0.625), ("calm", 0.375)], 30, pool_multiplier=1)
    assert catalog.loc[sad_row, "track_id"] not in set(out["track_id"])
    assert (out.loc[out["target"] == "happy", "mood"] == "happy").all()
    assert (out.loc[out["target"] == "calm", "mood"] == "calm").all()


def test_classifier_probability_ranks_within_each_target():
    catalog = _random_catalog()
    out = recommend._coverage_top(catalog, [("happy", 0.625), ("calm", 0.375)], 30, pool_multiplier=1)
    for target, n in [("happy", 19), ("calm", 11)]:
        eligible = catalog[catalog["mood"] == target]
        expected = set(eligible.nlargest(n, f"proba_{target}")["track_id"])
        picked = out[out["target"] == target]
        assert set(picked["track_id"]) == expected
        assert picked["score"].is_monotonic_decreasing
        np.testing.assert_allclose(picked["score"], picked[f"proba_{target}"])


def test_derived_target_ranks_by_derived_score():
    catalog = _random_catalog(n=4000)
    out = recommend._coverage_top(catalog, [("angry", 1.0)], 10, pool_multiplier=1)
    expected = catalog[catalog["derived_angry"]].nlargest(10, "derived_angry_score")
    assert set(out["track_id"]) == set(expected["track_id"])
    assert out["derived_angry"].all()


def test_deterministic_mode_is_reproducible_and_sampling_is_seedable():
    catalog = _random_catalog()
    targets = [("happy", 0.625), ("calm", 0.375)]
    a = recommend._coverage_top(catalog, targets, 30, pool_multiplier=1)
    b = recommend._coverage_top(catalog, targets, 30, pool_multiplier=1)
    assert a["track_id"].tolist() == b["track_id"].tolist()
    np.random.seed(7)
    c = recommend.recommend_music(catalog, "sad", "Cheer me up", top_n=30)
    np.random.seed(7)
    d = recommend.recommend_music(catalog, "sad", "Cheer me up", top_n=30)
    assert c["track_id"].tolist() == d["track_id"].tolist()


def test_shortfall_is_filled_by_other_targets():
    catalog = _random_catalog(n=40)  # only 10 calm-labelled tracks
    out = recommend._coverage_top(catalog, [("happy", 0.5), ("calm", 0.5)], 16, pool_multiplier=1)
    assert len(out) == 16 and out["track_id"].is_unique
    assert out["target"].value_counts().to_dict() == {"happy": 8, "calm": 8}
    out = recommend._coverage_top(catalog, [("calm", 0.5), ("happy", 0.5)], 30, pool_multiplier=1)
    assert len(out) == 20  # all 10 calm + all 10 happy; nothing else is eligible
    assert out["track_id"].is_unique


def test_ranking_ignores_popularity():
    catalog = _random_catalog()
    popular = catalog.assign(popularity=np.random.RandomState(3).randint(0, 100, len(catalog)))
    targets = [("happy", 0.625), ("calm", 0.375)]
    a = recommend._coverage_top(catalog, targets, 30, pool_multiplier=1)
    b = recommend._coverage_top(popular, targets, 30, pool_multiplier=1)
    assert a["track_id"].tolist() == b["track_id"].tolist()


def test_recommend_music_only_accepts_the_278k_catalog_shape():
    catalog = _random_catalog()
    out = recommend.recommend_music(catalog, "sad", "Cheer me up", top_n=10)
    assert "target" in out.columns and len(out) == 10
    # no track IDs (the retired legacy shape) or missing target columns -> a clear error, no fallback ranking
    for broken in [catalog.drop(columns=["track_id"]), catalog.drop(columns=["proba_happy"])]:
        with pytest.raises(ValueError, match="278k catalog"):
            recommend.recommend_music(broken, "sad", "Cheer me up", top_n=10)


# ---- built artifacts (need train_music_278k.py to have run) ----

requires_artifacts = pytest.mark.skipif(
    not (os.path.exists(MUSIC_CATALOG_278K_PATH) and os.path.exists(EVAL_PATH)),
    reason="run `python train_music_278k.py` to build the 278k catalog")


@pytest.fixture(scope="module")
def catalog():
    return pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str})


@pytest.fixture(scope="module")
def eval_report():
    with open(EVAL_PATH) as f:
        return json.load(f)


@requires_artifacts
def test_source_dataset_is_untouched(eval_report):
    assert mp.file_sha256(MUSIC_278K_SOURCE_PATH) == eval_report["source_sha256"]


@requires_artifacts
def test_catalog_covers_every_source_track_once(catalog):
    source_ids = pd.read_csv(MUSIC_278K_SOURCE_PATH, usecols=["uri"])["uri"].str.slice(len(mp.TRACK_URI_PREFIX))
    assert len(catalog) == SOURCE_ROWS
    assert catalog["track_id"].is_unique
    assert set(catalog["track_id"]) == set(source_ids)
    assert catalog["track_id"].str.fullmatch(r"[0-9A-Za-z]{22}").all()


@requires_artifacts
def test_catalog_preserves_the_four_labels(catalog):
    assert catalog["label"].value_counts().to_dict() == SOURCE_CLASS_COUNTS
    assert (catalog["label"].map(LABEL_TO_MOOD) == catalog["mood"]).all()
    assert set(catalog["mood"]) == REAL_LABELS


@requires_artifacts
def test_catalog_has_no_missing_values_and_valid_probabilities(catalog):
    assert not catalog.isna().any().any()
    assert "popularity" not in catalog.columns
    proba = catalog[mp.PROBA_COLUMNS].to_numpy()
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=5e-4)


@requires_artifacts
def test_recorded_cutoffs_reproduce_the_derived_flags(catalog, eval_report):
    for mood, info in eval_report["derived_moods"].items():
        assert info["experimental"] == DERIVED_MUSIC_MOODS[mood]["experimental"]
        mask = _mask_from_metadata(catalog, info)
        assert (catalog[f"derived_{mood}"].to_numpy() == mask).all()
        assert mask.sum() == info["candidate_count"] > 0


@requires_artifacts
def test_no_stressed_candidate_trips_a_guard(catalog):
    cand = catalog[catalog["derived_stressed"]]
    assert not ((cand.danceability == 0) | (cand.tempo == 0) | (cand.valence == 0)).any()
    assert not (cand.speechiness > 0.66).any()


@requires_artifacts
def test_recorded_stressed_profile_reproduces_stored_scores(catalog, eval_report):
    profile = eval_report["derived_moods"]["stressed"]["score"]["profile_percentiles"]
    assert list(profile) == DERIVED_MUSIC_MOODS["stressed"]["profile"]
    expected = 1 - np.mean([np.abs(catalog[f].rank(pct=True).to_numpy() - v) for f, v in profile.items()], axis=0)
    np.testing.assert_allclose(catalog["derived_stressed_score"], expected, atol=1e-6)
    # the profile is the median of the guarded candidates' percentile ranks
    cand = catalog["derived_stressed"].to_numpy()
    for f, v in profile.items():
        assert v == pytest.approx(np.median(catalog[f].rank(pct=True).to_numpy()[cand]))


@requires_artifacts
@pytest.mark.parametrize("pool_multiplier", [1, 3])
def test_stressed_top30_has_no_guarded_or_zero_valued_features(catalog, pool_multiplier):
    np.random.seed(0)
    out = recommend._coverage_top(catalog, [("stressed", 1.0)], 30, pool_multiplier=pool_multiplier)
    assert len(out) == 30 and out["track_id"].is_unique
    for feature in ["valence", "danceability", "tempo", "speechiness", "energy"]:
        assert (out[feature] > 0).all(), feature
    assert (out["speechiness"] <= 0.66).all()


@requires_artifacts
def test_angry_behavior_is_unchanged(catalog, eval_report):
    info = eval_report["derived_moods"]["angry"]
    assert info["guards"] == {} and info["score"]["method"] == "extreme"
    assert info["candidate_count"] == info["rule_match_count"] == 5706
    pct = {f: catalog[f].rank(pct=True).to_numpy() for f in ["valence", "energy", "loudness"]}
    expected = ((1 - pct["valence"]) + pct["energy"] + pct["loudness"]) / 3
    np.testing.assert_allclose(catalog["derived_angry_score"], expected, atol=1e-6)
    out = recommend._coverage_top(catalog, [("angry", 1.0)], 30, pool_multiplier=1)
    top = catalog[catalog["derived_angry"]].nlargest(30, "derived_angry_score")
    assert set(out["track_id"]) == set(top["track_id"])


@requires_artifacts
def test_retrained_classifier_meets_held_out_floor(eval_report):
    held_out = eval_report["held_out_evaluation"]
    assert eval_report["final_model"]["trained_rows"] == SOURCE_ROWS
    # guards against a broken retrain; the legacy model scored 0.835 on this split
    assert held_out["softmax_test_accuracy"] > 0.80
    assert held_out["softmax_test_accuracy"] > held_out["naive_bayes_test_accuracy"]


@requires_artifacts
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("mood", MOODS)
def test_recommend_music_on_full_catalog_covers_every_target(catalog, mood, mode):
    targets = recommend._target_music_weighted(mood, mode)
    out = recommend.recommend_music(catalog, mood, mode, top_n=30)
    assert len(out) == 30
    assert out["track_id"].is_unique
    shares = dict(zip([t for t, _ in targets], recommend._allocate_shares([w for _, w in targets], 30)))
    assert out["target"].value_counts().to_dict() == shares
    for target, rows in out.groupby("target"):
        member, _ = recommend._target_candidates(rows, target)
        assert member.all(), f"off-target track picked for {target}"


@requires_artifacts
def test_api_returns_track_ids_when_serving_the_278k_catalog(catalog, monkeypatch):
    monkeypatch.setattr(flask_app, "load_music_catalog", lambda: catalog)
    res = flask_app.app.test_client().post(
        "/api/recommend", json={"mood": "chill", "mode": "match", "type": "music", "limit": 10})
    assert res.status_code == 200
    results = res.get_json()["results"]
    assert len(results) == 10
    ids = [r["track_id"] for r in results]
    assert len(set(ids)) == 10 and all(len(i) == 22 for i in ids)
