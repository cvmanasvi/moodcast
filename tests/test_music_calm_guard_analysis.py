"""Tests for the analysis-only calm speechiness guard (evaluate_music_calm_guard_analysis.py): the guard
touches calm only, affected requests keep exact shares when candidates suffice, shortfalls never relax
the guard and are deterministic and counted, E2 without speech-like tracks equals E1, the third seed set
is disjoint, and the report is aggregate-only."""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_calm_guard_analysis as guard
import evaluate_music_diversity_analysis as div
import evaluate_music_stratified_prototype as proto
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS

MODES = ["Match my mood", "Cheer me up"]


def _catalog(speech_like_calm=0, n=2000, seed=7):
    rng = np.random.RandomState(seed)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["speechiness"] = rng.rand(n) * 0.5                       # nothing speech-like ...
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    cat = mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4), size=n))
    cat = mp.derive_moods(cat)[0]
    if speech_like_calm:                                          # ... unless asked for, on calm tracks only
        calm, _ = recommend._target_candidates(cat, "calm")
        cat.loc[cat.index[np.flatnonzero(calm)[:speech_like_calm]], "speechiness"] = 0.9
    return cat


def _z(cat):
    X = cat[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)


def test_third_seed_set_is_disjoint_from_earlier_phases():
    assert guard.seeds_disjoint(50)
    lo, hi = guard.third_seed_range(50)
    assert (lo, hi) == (12000, 12049)
    for used in [range(2026, 2076), range(7000, 7050), range(0, 30)]:
        assert not set(range(lo, hi + 1)) & set(used)


def test_guard_only_changes_calm_candidates():
    cat = _catalog(speech_like_calm=40)
    sp = cat["speechiness"].to_numpy()
    for target in ["sad", "happy", "energetic", "calm", "angry", "stressed"]:
        before, s1 = recommend._target_candidates(cat, target)
        after, s2 = guard.guarded_candidates(cat, target)
        assert np.array_equal(s1, s2)
        if target == "calm":
            assert np.array_equal(after, before & ~(sp > 0.66)) and (before & ~after).sum() == 40
        else:
            assert np.array_equal(after, before)


def test_affected_requests_are_exactly_the_calm_containing_ones():
    expected = {f"{m} / {md}" for m in MOODS for md in ev.MODES
                if "calm" in [t for t, _ in ev.request_targets(m, md)]}
    assert set(guard.calm_requests()) == expected
    assert {"chill / match", "stressed / uplift", "sad / uplift", "angry / uplift", "chill / uplift"} == expected


def test_e2_keeps_exact_shares_and_excludes_speech_like_calm_tracks():
    cat = _catalog(speech_like_calm=60)
    z, sp = _z(cat), cat["speechiness"].to_numpy()
    for mood in MOODS:
        for mode in MODES:
            targets = recommend._target_music_weighted(mood, mode)
            shortfall = {}
            np.random.seed(4)
            rows = guard.coverage_top_guarded(cat, targets, 30, div.POOL_MULTIPLIER, proto.numpy_kmeans_select,
                                              z, shortfall)
            assert shortfall == {} and len(rows) == 30 and len(set(rows.tolist())) == 30
            for (target, _), share in zip(targets, recommend._allocate_shares([w for _, w in targets], 30)):
                member, _ = recommend._target_candidates(cat, target)
                assert int(member[rows].sum()) == share
                if target == "calm":
                    assert not (sp[rows[member[rows]]] > 0.66).any()


def test_e2_equals_e1_when_no_calm_track_is_speech_like():
    cat = _catalog()
    z = _z(cat)
    for mood in MOODS:
        for mode in MODES:
            targets = recommend._target_music_weighted(mood, mode)
            np.random.seed(2)
            e1 = div.coverage_top_selector(cat, targets, 30, div.POOL_MULTIPLIER, proto.numpy_kmeans_select, z)
            np.random.seed(2)
            e2 = guard.coverage_top_guarded(cat, targets, 30, div.POOL_MULTIPLIER, proto.numpy_kmeans_select, z)
            assert np.array_equal(e1, e2), (mood, mode)


def _scarce_calm_catalog(keep):
    """Only `keep` calm tracks survive the guard: every other calm track is speech-like."""
    cat = _catalog()
    calm, _ = recommend._target_candidates(cat, "calm")
    cat.loc[cat.index[np.flatnonzero(calm)[keep:]], "speechiness"] = 0.9
    return cat


def test_shortfall_never_relaxes_guard_and_is_deterministic_and_counted():
    cat = _scarce_calm_catalog(keep=5)
    z, sp = _z(cat), cat["speechiness"].to_numpy()
    calm, _ = recommend._target_candidates(cat, "calm")
    targets = recommend._target_music_weighted("sad", "Cheer me up")        # happy 19 + calm 11
    runs = []
    for _ in range(2):
        shortfall = {}
        np.random.seed(9)
        rows = guard.coverage_top_guarded(cat, targets, 30, div.POOL_MULTIPLIER, proto.numpy_kmeans_select,
                                          z, shortfall)
        runs.append((rows, shortfall))
    (rows, shortfall), (again, shortfall2) = runs
    assert np.array_equal(rows, again) and shortfall == shortfall2 == {"calm": 6}
    assert len(rows) == 30 and len(set(rows.tolist())) == 30
    assert int(calm[rows].sum()) == 5 and not (sp[rows] > 0.66).any()      # guard held; happy took 25 slots


def test_calm_only_request_with_too_few_guarded_candidates_returns_a_short_list():
    cat = _scarce_calm_catalog(keep=5)
    z = _z(cat)
    shortfall = {}
    np.random.seed(0)
    rows = guard.coverage_top_guarded(cat, [("calm", 1.0)], 30, div.POOL_MULTIPLIER, proto.numpy_kmeans_select,
                                      z, shortfall)
    assert len(rows) == 5 and shortfall == {"calm": 25}


ARTIFACT = os.path.join(MODELS_DIR, "music_calm_guard_analysis.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run `python evaluate_music_calm_guard_analysis.py`")
def test_artifact_is_aggregate_only():
    with open(ARTIFACT, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)
    assert not re.search(r'"(track_id|track_name|artist|artists|title|uri)"', text)
    assert report["analysis_only"] is True
    assert report["third_seed_set"]["disjoint_from_previously_used"] is True
    assert report["gates"] == proto.GATES
