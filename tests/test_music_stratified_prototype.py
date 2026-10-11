"""Tests for the analysis-only stratified-selection prototypes (evaluate_music_stratified_prototype.py):
eligibility, exact target shares, uniqueness and same-seed reproducibility for every request, the
partitioning helpers' edge cases, the gate bookkeeping, and an aggregate-only artifact."""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_diversity_analysis as div
import evaluate_music_stratified_prototype as proto
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS

MODES = ["Match my mood", "Cheer me up"]
PROTOTYPES = ["E: numpy k-means", "F: PC1 quantile strata"]


@pytest.fixture(scope="module")
def catalog():
    n = 2000
    rng = np.random.RandomState(6)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    cat = mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4), size=n))
    return mp.derive_moods(cat)[0]


@pytest.fixture(scope="module")
def z(catalog):
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)


@pytest.mark.parametrize("name", PROTOTYPES)
def test_prototypes_keep_eligibility_shares_uniqueness_and_seed(catalog, z, name):
    select = proto.strategies()[name]
    for mood in MOODS:
        for mode in MODES:
            targets = recommend._target_music_weighted(mood, mode)
            np.random.seed(8)
            rows = div.coverage_top_selector(catalog, targets, 30, div.POOL_MULTIPLIER, select, z)
            np.random.seed(8)
            again = div.coverage_top_selector(catalog, targets, 30, div.POOL_MULTIPLIER, select, z)
            assert np.array_equal(rows, again), (name, mood, mode)
            assert len(rows) == 30 and len(set(rows.tolist())) == 30
            shares = recommend._allocate_shares([w for _, w in targets], 30)
            for (target, _), share in zip(targets, shares):
                member, _ = recommend._target_candidates(catalog, target)
                assert int(member[rows].sum()) == share, (name, mood, mode, target)


@pytest.mark.parametrize("name", PROTOTYPES)
def test_prototypes_stay_inside_the_production_pool(catalog, z, name):
    member, signal = recommend._target_candidates(catalog, "happy")
    pool = set(div.candidate_pool(signal, member, 30, div.POOL_MULTIPLIER).tolist())
    np.random.seed(1)
    assert set(proto.strategies()[name](signal, member, 30, div.POOL_MULTIPLIER, z).tolist()) <= pool


@pytest.mark.parametrize("name", PROTOTYPES)
def test_prototypes_return_everything_when_fewer_candidates_than_slots(catalog, z, name):
    member, signal = recommend._target_candidates(catalog, "sad")
    few = np.zeros(len(member), dtype=bool)
    few[np.flatnonzero(member)[:7]] = True
    rows = proto.strategies()[name](signal, few, 30, div.POOL_MULTIPLIER, z)
    assert set(rows.tolist()) == set(np.flatnonzero(few).tolist())
    assert list(signal[rows]) == sorted(signal[rows], reverse=True)   # best-first


def test_kmeans_separates_well_separated_blobs_and_is_seeded():
    rng = np.random.RandomState(0)
    f = np.vstack([rng.normal(c, 0.05, size=(40, 9)) for c in (0.0, 5.0, 10.0)])
    np.random.seed(3)
    labels = proto.kmeans_labels(f, 3)
    assert sorted(np.bincount(labels).tolist()) == [40, 40, 40]
    assert all(len(set(labels[i:i + 40])) == 1 for i in (0, 40, 80))
    np.random.seed(3)
    assert np.array_equal(labels, proto.kmeans_labels(f, 3))


def test_kmeans_handles_identical_points():
    f = np.ones((50, 9))
    np.random.seed(0)
    labels = proto.kmeans_labels(f, 5)
    assert len(labels) == 50 and labels.min() >= 0 and labels.max() < 5


def test_empty_strata_are_filled_from_the_rest_of_the_pool():
    idx = np.arange(100, 160)
    signal = np.linspace(0, 1, 200)
    np.random.seed(0)
    rows = proto._one_pick_per_stratum(idx, np.zeros(60, dtype=int), 10, signal)   # one stratum only
    assert len(rows) == 10 and len(set(rows.tolist())) == 10 and set(rows.tolist()) <= set(idx.tolist())


def test_pc1_strata_are_equal_count_and_ordered():
    rng = np.random.RandomState(1)
    f = rng.normal(size=(600, 9))
    labels = proto.pc1_strata_labels(f, 30)
    assert set(np.bincount(labels).tolist()) == {20}
    centred = f - f.mean(axis=0)
    pc1 = centred @ np.linalg.svd(centred, full_matrices=False)[2][0]
    means = [pc1[labels == k].mean() for k in range(30)]
    assert means == sorted(means)


def _result(**over):
    summary = {"target_consistency": 1.0, "exact_target_shares_all_lists": True, "all_same_seed_identical": True,
               "single_target_mean_diversity": 3.5, "single_target_requests_at_least_random_eligible": 5,
               "mean_ranking_signal_of_served": 0.993, "catalog_unique_tracks": 2400, "mean_repeat_rate": 0.74,
               "latency_ms": {"p95": 10.0}}
    res = {"summary": summary, "requests": {"sad / match": {"diversity": 2.3}},
           "speech_like_share": {"calm_requests": 0.1},
           "paired_vs_production": {r: {"bootstrap_95": [0.1, 0.5]} for r in proto.GATES["paired_lower_bound_positive_for"]}}
    for key, value in over.items():
        if key in summary:
            summary[key] = value
        else:
            res[key] = value
    return res


def test_gate_status_pass_fail_and_not_established():
    assert proto.all_pass(proto.gate_status(_result()))
    assert proto.gate_status(_result(mean_repeat_rate=0.76))["repeat_rate_<=0.75"] == "FAIL"
    assert proto.gate_status(_result(latency_ms={"p95": 25.1}))["p95_latency_<=25ms"] == "FAIL"
    assert proto.gate_status(_result(speech_like_share={"calm_requests": 0.21}))["calm_speech_like_<=0.20"] == "FAIL"
    assert proto.gate_status(_result(requests={"sad / match": {"diversity": 1.99}}))["sad_match_diversity_>=2.0"] == "FAIL"
    no_paired = _result()
    del no_paired["paired_vs_production"]
    assert proto.gate_status(no_paired)["paired_lower_bounds_>0"] == "NOT ESTABLISHED"


ARTIFACT = os.path.join(MODELS_DIR, "music_stratified_prototype.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run `python evaluate_music_stratified_prototype.py`")
def test_artifact_is_aggregate_only():
    with open(ARTIFACT, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)
    assert not re.search(r'"(track_id|track_name|artist|artists|title|uri)"', text)
    assert report["analysis_only"] is True
    assert report["production_configuration"] == {"pool_multiplier": 20, "weighting": "current"}
    assert report["gates"] == proto.GATES
