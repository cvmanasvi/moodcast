"""Production feature-cell routing in src/recommend.py (Phases 2K/2K.1): which requests use cells, the calm
speechiness guard on the calm cell route, deterministic cell construction and selection, exact shares,
ordering and uniqueness, fallback to the standard ranking, unchanged routes, one build per catalog,
fidelity to the validated Phase 2J.1 selection, and untouched source artifacts."""
import itertools
import json
import logging
import os

import numpy as np
import pandas as pd
import pytest

from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from tests.local_data import requires_music_catalog, requires_music_classifier, skip_if_missing

MODES = ["Match my mood", "Cheer me up"]
ROUTED = {("happy", "Match my mood"), ("sad", "Match my mood"), ("chill", "Match my mood"),
          ("energetic", "Match my mood"), ("stressed", "Cheer me up")}


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(recommend, "_cell_cache", {"catalog": None, "cells": None})


@pytest.fixture(scope="module")
def catalog():
    n = 4000
    rng = np.random.RandomState(13)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    cat = mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4) * 0.3, size=n))
    return mp.derive_moods(cat)[0]


def _seeded(fn, seed=7):
    np.random.seed(seed)
    return fn()


# 1. routing

def test_routing_for_all_twelve_requests():
    routed = {(m, md) for m in MOODS for md in MODES
              if recommend._uses_feature_cells(recommend._target_music_weighted(m, md))}
    assert routed == ROUTED
    # stressed / uplift is routed because its effective target is calm, like chill / match
    assert recommend._target_music_weighted("stressed", "Cheer me up") == \
        recommend._target_music_weighted("chill", "Match my mood")


# 2. determinism

def test_cell_construction_is_deterministic_and_partitions_each_label(catalog):
    a, b = recommend._build_feature_cells(catalog), recommend._build_feature_cells(catalog)
    assert set(a) == set(recommend.FEATURE_CELL_TARGETS)
    for target in a:
        assert all(np.array_equal(x, y) for x, y in zip(a[target], b[target]))
        member, signal = recommend._cell_candidates(catalog, target)
        allrows = np.concatenate(a[target])
        assert len(a[target]) == 30 and len(allrows) == member.sum() == len(set(allrows.tolist()))
        sizes = [len(c) for c in a[target]]
        assert max(sizes) - min(sizes) <= 2
        assert all(list(signal[c]) == sorted(signal[c], reverse=True) for c in a[target])


def test_selection_is_deterministic_for_a_seed(fresh_cache, catalog):
    for mood, mode in ROUTED:
        first = _seeded(lambda: recommend.recommend_music(catalog, mood, mode, top_n=30))
        again = _seeded(lambda: recommend.recommend_music(catalog, mood, mode, top_n=30))
        assert first["track_id"].tolist() == again["track_id"].tolist()


# 3 + 4. shares, ordering, uniqueness

@pytest.mark.parametrize("top_n", [1, 5, 30, 100])
def test_routed_lists_are_exact_unique_and_best_first(fresh_cache, catalog, top_n):
    for mood, mode in ROUTED:
        target = recommend._target_music_weighted(mood, mode)[0][0]
        out = _seeded(lambda: recommend.recommend_music(catalog, mood, mode, top_n=top_n))
        member, signal = recommend._target_candidates(catalog, target)
        rows = catalog.index.get_indexer(out.index)
        assert len(out) == top_n and out["track_id"].is_unique
        assert member[rows].all() and (out["target"] == target).all()
        assert list(out["score"]) == sorted(out["score"], reverse=True)
        assert np.allclose(out["score"], signal[rows])
        assert list(out.columns) == list(recommend._coverage_top(catalog, [(target, 1.0)], 3).columns)


def test_too_few_cell_candidates_fall_back_to_the_standard_ranking(fresh_cache, catalog, caplog):
    small = catalog[(catalog["mood"] != "sad") | (np.arange(len(catalog)) < 40)]
    with caplog.at_level(logging.WARNING, logger="src.recommend"):
        out = _seeded(lambda: recommend.recommend_music(small, "sad", "Match my mood", top_n=100))
    expected = _seeded(lambda: recommend._coverage_top(small, [("sad", 1.0)], 100))
    pd.testing.assert_frame_equal(out, expected)                     # no padding, no changed shares
    assert "Too few feature-cell candidates for sad" in caplog.text


# 5. fallback

def test_fallback_when_cell_construction_fails(fresh_cache, monkeypatch, catalog, caplog):
    def broken(df):
        raise RuntimeError("secret detail that must not be logged")
    monkeypatch.setattr(recommend, "_build_feature_cells", broken)
    with caplog.at_level(logging.WARNING, logger="src.recommend"):
        out = _seeded(lambda: recommend.recommend_music(catalog, "sad", "Match my mood", top_n=30))
    expected = _seeded(lambda: recommend._coverage_top(catalog, [("sad", 1.0)], 30))
    assert out["track_id"].tolist() == expected["track_id"].tolist()
    assert "RuntimeError" in caplog.text and "secret" not in caplog.text


def test_fallback_when_cell_selection_fails_restores_the_standard_ranking(fresh_cache, monkeypatch, catalog, caplog):
    def broken(*args, **kwargs):
        np.random.random(5)                      # consume randomness before failing
        raise ValueError("track 123 secret")
    monkeypatch.setattr(recommend, "_cell_select", broken)
    with caplog.at_level(logging.WARNING, logger="src.recommend"):
        out = _seeded(lambda: recommend.recommend_music(catalog, "happy", "Match my mood", top_n=30))
    expected = _seeded(lambda: recommend._coverage_top(catalog, [("happy", 1.0)], 30))
    assert out["track_id"].tolist() == expected["track_id"].tolist()
    assert "ValueError" in caplog.text and "123" not in caplog.text


# 6. unchanged routes

def test_every_other_request_is_exactly_the_previous_ranking(fresh_cache, catalog):
    for mood, mode in itertools.product(MOODS, MODES):
        if (mood, mode) in ROUTED:
            continue
        targets = recommend._target_music_weighted(mood, mode)
        for top_n in (10, 30):
            out = _seeded(lambda: recommend.recommend_music(catalog, mood, mode, top_n=top_n))
            old = _seeded(lambda: recommend._coverage_top(catalog, targets, top_n))
            pd.testing.assert_frame_equal(out, old)


# 7. built once per process (per catalog object)

def test_cells_are_built_once_per_catalog(fresh_cache, monkeypatch, catalog):
    calls = []
    real = recommend._build_feature_cells
    monkeypatch.setattr(recommend, "_build_feature_cells", lambda df: calls.append(1) or real(df))
    for i, (mood, mode) in enumerate(sorted(ROUTED) * 3):
        np.random.seed(i)
        recommend.recommend_music(catalog, mood, mode, top_n=10)
    assert len(calls) == 1
    for mood in MOODS:                           # unrouted requests never build cells
        recommend.recommend_music(catalog, mood, "Cheer me up", top_n=5)
    assert len(calls) == 1


def test_failed_build_is_not_retried(fresh_cache, monkeypatch, catalog):
    calls = []

    def broken(df):
        calls.append(1)
        raise MemoryError
    monkeypatch.setattr(recommend, "_build_feature_cells", broken)
    for seed in range(4):
        np.random.seed(seed)
        assert len(recommend.recommend_music(catalog, "chill", "Match my mood", top_n=10)) == 10
    assert len(calls) == 1


# fidelity to the validated Phase 2J.1 selection (real catalog)

@pytest.fixture(scope="module")
def real_catalog():
    from src.data_loader import load_music_catalog
    skip_if_missing(requires_music_catalog)
    return load_music_catalog()


def test_matches_the_validated_selection_on_the_real_catalog(fresh_cache, real_catalog):
    import evaluate_music_balanced_quality as bq
    import evaluate_music_cell_confirmation as cc
    X = real_catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    guarded_cells = bq.build_cells(real_catalog, z, guarded=True)
    unguarded_cells = bq.build_cells(real_catalog, z, guarded=False)
    for seed in (0, 1, 2):
        for mood, target in (("happy", "happy"), ("sad", "sad"), ("energetic", "energetic")):
            prod = _seeded(lambda: recommend.recommend_music(real_catalog, mood, "Match my mood", 30), seed)
            validated = _seeded(lambda: cc.recommend_rows(real_catalog, ev.request_targets(mood, "match"),
                                                          cc.T, guarded_cells), seed)
            assert prod.index.tolist() == validated.tolist(), (mood, seed)
        # calm (Phase 2K.1): the guarded calm cell route matches the validated guarded selection, both aliases
        for mood, mode in (("chill", "Match my mood"), ("stressed", "Cheer me up")):
            prod = _seeded(lambda: recommend.recommend_music(real_catalog, mood, mode, 30), seed)
            validated = _seeded(lambda: cc.recommend_rows(real_catalog, [("calm", 1.0)], cc.T, guarded_cells), seed)
            assert prod.index.tolist() == validated.tolist(), (mood, seed)
        member, signal = recommend._target_candidates(real_catalog, "calm")
        unguarded = _seeded(lambda: bq.cell_select("calm", signal, member, 30, unguarded_cells), seed)
        assert prod.index.tolist() != unguarded.tolist()          # the guard changes the calm route


# 8. source artifacts

@requires_music_catalog
@requires_music_classifier
def test_source_catalog_and_classifier_are_unchanged():
    with open(os.path.join(MODELS_DIR, "music_cell_confirmation.json"), encoding="utf-8") as f:
        recorded = json.load(f)["source_artifact_sha256"]
    assert mp.file_sha256(MUSIC_CATALOG_278K_PATH) == recorded["catalog"]
    assert mp.file_sha256(os.path.join(MODELS_DIR, "music_mood_classifier_278k.pkl")) == recorded["classifier"]


# calm speechiness guard (Phase 2K.1)

def _calm_catalog_with(speechiness_values, seed=15, n=3000):
    """Synthetic catalog with speechiness below 0.5, except the given values on the first calm tracks."""
    rng = np.random.RandomState(seed)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["speechiness"] = rng.rand(n) * 0.5
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    cat = mp.derive_moods(mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4), size=n)))[0]
    calm, _ = recommend._target_candidates(cat, "calm")
    rows = np.flatnonzero(calm)[:len(speechiness_values)]
    cat.loc[cat.index[rows], "speechiness"] = speechiness_values
    return cat, rows


def test_guard_threshold_keeps_0_66_and_excludes_anything_above():
    cat, rows = _calm_catalog_with([0.66, 0.66000001, 0.9, 0.659])
    member, _ = recommend._cell_candidates(cat, "calm")
    assert member[rows].tolist() == [True, False, False, True]
    in_cells = set(np.concatenate(recommend._build_feature_cells(cat)["calm"]).tolist())
    assert rows[0] in in_cells and rows[3] in in_cells
    assert rows[1] not in in_cells and rows[2] not in in_cells
    assert recommend.CALM_CELL_SPEECHINESS_MAX == 0.66


def test_guard_applies_to_the_calm_cell_route_only():
    cat, _ = _calm_catalog_with([])
    cat = cat.copy()
    cat["speechiness"] = 0.9                                  # every track speech-like
    for target in ("happy", "sad", "energetic"):
        assert np.array_equal(recommend._cell_candidates(cat, target)[0],
                              recommend._target_candidates(cat, target)[0])
    assert not recommend._cell_candidates(cat, "calm")[0].any()
    # the standard ranking's candidates (used by _coverage_top) ignore speechiness for calm
    assert np.array_equal(recommend._target_candidates(cat, "calm")[0], (cat["mood"] == "calm").to_numpy())


@pytest.mark.parametrize("top_n", [5, 30, 100])
def test_calm_routes_serve_no_speech_like_tracks_and_stay_deterministic(fresh_cache, top_n):
    cat, _ = _calm_catalog_with([0.9] * 60)
    for mood, mode in (("chill", "Match my mood"), ("stressed", "Cheer me up")):
        out = _seeded(lambda: recommend.recommend_music(cat, mood, mode, top_n=top_n))
        again = _seeded(lambda: recommend.recommend_music(cat, mood, mode, top_n=top_n))
        assert out["track_id"].tolist() == again["track_id"].tolist()
        assert len(out) == top_n and out["track_id"].is_unique and (out["mood"] == "calm").all()
        assert (out["speechiness"] <= 0.66).all()
        assert list(out["score"]) == sorted(out["score"], reverse=True)


def test_insufficient_guarded_calm_population_falls_back_without_padding(fresh_cache, caplog):
    cat, _ = _calm_catalog_with([])
    cat = cat.copy()
    calm = np.flatnonzero(recommend._target_candidates(cat, "calm")[0])
    cat.loc[cat.index[calm[10:]], "speechiness"] = 0.9        # only 10 guarded calm candidates remain
    with caplog.at_level(logging.WARNING, logger="src.recommend"):
        out = _seeded(lambda: recommend.recommend_music(cat, "chill", "Match my mood", top_n=30))
    pd.testing.assert_frame_equal(out, _seeded(lambda: recommend._coverage_top(cat, [("calm", 1.0)], 30)))
    assert len(out) == 30 and "Too few feature-cell candidates for calm" in caplog.text
    exact = _seeded(lambda: recommend.recommend_music(cat, "chill", "Match my mood", top_n=10))
    assert len(exact) == 10 and (exact["speechiness"] <= 0.66).all()   # exactly enough: cells still used


def test_calm_uplift_mixes_and_unrouted_requests_ignore_the_guard(fresh_cache):
    cat, _ = _calm_catalog_with([0.9] * 60)
    for mood, mode in itertools.product(MOODS, MODES):
        if (mood, mode) in ROUTED:
            continue
        targets = recommend._target_music_weighted(mood, mode)
        out = _seeded(lambda: recommend.recommend_music(cat, mood, mode, top_n=30))
        pd.testing.assert_frame_equal(out, _seeded(lambda: recommend._coverage_top(cat, targets, 30)))


def test_non_calm_cells_are_unaffected_by_speechiness(fresh_cache):
    cat, _ = _calm_catalog_with([])
    loud = cat.copy()
    loud["speechiness"] = np.where(loud["mood"] != "calm", 0.95, loud["speechiness"])
    for target in ("happy", "sad", "energetic"):
        assert np.array_equal(recommend._cell_candidates(cat, target)[0], recommend._cell_candidates(loud, target)[0])
