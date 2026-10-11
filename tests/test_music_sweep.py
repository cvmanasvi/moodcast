"""Tests for the analysis-only pool/sampling sweep (evaluate_music_recommender_analysis.py):
the replica sampler must equal production with the current weighting, and the
sweep must follow (never set) the production default."""
import inspect
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_recommender_analysis as sweep
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS


def _catalog(n=1500, seed=4):
    rng = np.random.RandomState(seed)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    catalog = mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4), size=n))
    return mp.derive_moods(catalog)[0]


def test_production_default_is_pool_20():
    # Phase 2H.6 adopted pool_multiplier=20 (current weighting) from the 2H.5 sweep
    assert inspect.signature(recommend._coverage_top).parameters["pool_multiplier"].default == 20
    assert sweep.PRODUCTION_POOL_MULTIPLIER == 20
    assert {3, 20} <= set(sweep.POOL_MULTIPLIERS)  # baseline and adopted value stay in the sweep


@pytest.mark.parametrize("pool_multiplier", [1, 3, 10, 20])
def test_replica_with_current_weighting_equals_production(pool_multiplier):
    catalog = _catalog()
    for mood in MOODS:
        for mode in ["Match my mood", "Cheer me up"]:
            targets = recommend._target_music_weighted(mood, mode)
            for seed in range(3):
                np.random.seed(seed)
                prod = recommend._coverage_top(catalog, targets, 30, pool_multiplier=pool_multiplier).index.to_numpy()
                np.random.seed(seed)
                rep = sweep.coverage_top_variant(catalog, targets, 30, pool_multiplier, sweep.WEIGHTINGS["current"])
                assert np.array_equal(prod, rep), (mood, mode, seed)


@pytest.mark.parametrize("weighting", ["sqrt", "uniform"])
def test_alternative_weightings_keep_target_shares(weighting):
    catalog = _catalog()
    targets = recommend._target_music_weighted("sad", "Cheer me up")
    rows = sweep.coverage_top_variant(catalog, targets, 30, 10, sweep.WEIGHTINGS[weighting])
    assert len(set(rows.tolist())) == 30
    shares = recommend._allocate_shares([w for _, w in targets], 30)
    for (target, _), share in zip(targets, shares):
        member, _ = recommend._target_candidates(catalog, target)
        assert int(member[rows].sum()) == share


def test_pareto_front_drops_dominated_configs():
    def cfg(cov, div, rep, p95):
        return {"summary": {"target_consistency": 1.0, "catalog_coverage": cov, "mean_diversity": div,
                            "mean_repeat_rate": rep, "latency_ms": {"p95": p95}}}
    configs = {"a": cfg(0.1, 3.0, 0.9, 20), "b": cfg(0.2, 3.5, 0.8, 20), "c": cfg(0.3, 3.0, 0.7, 30)}
    assert sweep.pareto_front(configs) == ["b", "c"]


SWEEP_PATH = os.path.join(MODELS_DIR, "music_recommender_sweep.json")


@pytest.mark.skipif(not os.path.exists(SWEEP_PATH), reason="run `python evaluate_music_recommender_analysis.py`")
def test_sweep_artifact_is_aggregate_only():
    with open(SWEEP_PATH, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)          # no track IDs
    assert report["analysis_only"] is True
    # the committed sweep is the Phase 2H.5 record, generated while production used pool 3
    assert report["production_configuration"] == {"pool_multiplier": 3, "weighting": "current"}
    assert set(report["pool_size_sweep"]) == {f"pool_multiplier={m}" for m in sweep.POOL_MULTIPLIERS}
