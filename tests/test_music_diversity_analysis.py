"""Tests for the analysis-only single-target diversity study (evaluate_music_diversity_analysis.py):
the production replica must equal production, every experimental strategy must keep eligibility,
exact target shares, uniqueness and same-seed reproducibility, and the artifact must stay aggregate-only."""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_diversity_analysis as div
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS

MODES = ["Match my mood", "Cheer me up"]


@pytest.fixture(scope="module")
def catalog():
    n = 2000
    rng = np.random.RandomState(5)
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


def test_production_replica_equals_production(catalog, z):
    for mood in MOODS:
        for mode in MODES:
            targets = recommend._target_music_weighted(mood, mode)
            for seed in range(3):
                np.random.seed(seed)
                prod = recommend._coverage_top(catalog, targets, 30).index.to_numpy()
                np.random.seed(seed)
                rep = div.coverage_top_selector(catalog, targets, 30, div.POOL_MULTIPLIER, div.select_production, z)
                assert np.array_equal(prod, rep), (mood, mode, seed)


def test_analysis_follows_the_production_pool_default():
    assert div.POOL_MULTIPLIER == 20


def test_candidate_pool_is_the_production_pool(catalog):
    member, signal = recommend._target_candidates(catalog, "sad")
    pool = div.candidate_pool(signal, member, 5, 1)
    # pool_multiplier=1 makes production deterministic: it returns exactly the pool
    assert set(pool) == set(recommend._sample_best(signal, member, 5, 1))


@pytest.mark.parametrize("name", list(div.strategies()))
def test_strategies_keep_eligibility_shares_uniqueness_and_seed(catalog, z, name):
    select = div.strategies()[name]
    for mood in MOODS:
        for mode in MODES:
            targets = recommend._target_music_weighted(mood, mode)
            np.random.seed(3)
            rows = div.coverage_top_selector(catalog, targets, 30, div.POOL_MULTIPLIER, select, z)
            np.random.seed(3)
            again = div.coverage_top_selector(catalog, targets, 30, div.POOL_MULTIPLIER, select, z)
            assert np.array_equal(rows, again), (name, mood, mode)
            assert len(rows) == 30 and len(set(rows.tolist())) == 30
            shares = recommend._allocate_shares([w for _, w in targets], 30)
            for (target, _), share in zip(targets, shares):
                member, _ = recommend._target_candidates(catalog, target)
                assert int(member[rows].sum()) == share, (name, mood, mode, target)


def test_mmr_stays_inside_the_pool_and_redundancy_term_adds_spread(catalog, z):
    member, signal = recommend._target_candidates(catalog, "happy")
    pool = set(div.candidate_pool(signal, member, 30, div.POOL_MULTIPLIER).tolist())
    spread = {}
    for lam in (0.0, 1.0):
        lists = []
        for seed in range(10):
            np.random.seed(seed)
            rows = div.mmr_select(signal, member, 30, div.POOL_MULTIPLIER, z, lam=lam)
            assert set(rows.tolist()) <= pool
            lists.append(ev.intra_list_diversity(z[rows]))
        spread[lam] = np.mean(lists)
    assert spread[0.0] > spread[1.0]


def test_stratified_draws_one_track_per_cluster(catalog, z):
    member, signal = recommend._target_candidates(catalog, "calm")
    np.random.seed(0)
    rows = div.stratified_select(signal, member, 12, div.POOL_MULTIPLIER, z)
    assert len(rows) == 12 and len(set(rows.tolist())) == 12
    assert member[rows].all()


def test_selection_rule_requires_every_gate(catalog):
    good = {"target_consistency": 1.0, "exact_target_shares_all_lists": True, "all_same_seed_identical": True,
            "mean_ranking_signal_of_served": 0.995, "mean_repeat_rate": 0.7, "latency_ms": {"p95": 10.0}}
    assert div.passes_gates(good)[0]
    for key, bad in [("target_consistency", 0.99), ("exact_target_shares_all_lists", False),
                     ("all_same_seed_identical", False), ("mean_ranking_signal_of_served", 0.98),
                     ("mean_repeat_rate", 0.8), ("latency_ms", {"p95": 30.0})]:
        assert not div.passes_gates({**good, key: bad})[0], key


ARTIFACT = os.path.join(MODELS_DIR, "music_diversity_analysis.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run `python evaluate_music_diversity_analysis.py`")
def test_artifact_is_aggregate_only():
    with open(ARTIFACT, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)          # no track IDs
    assert not re.search(r'"(track_id|track_name|artist|artists|title|uri)"', text)
    assert report["analysis_only"] is True
    assert report["production_configuration"] == {"pool_multiplier": 20, "weighting": "current"}
    assert report["selection_rule"] == div.SELECTION_RULE
