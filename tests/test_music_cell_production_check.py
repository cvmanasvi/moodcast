"""Tests for the analysis-only Phase 2K post-implementation check (evaluate_music_cell_production.py):
fresh seeds, request mapping, the variants it compares, frozen gates and an aggregate-only report."""
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_balanced_quality as bq
import evaluate_music_cell_production as cp
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from tests.local_data import requires_music_catalog


def test_seeds_are_fresh():
    assert cp.seeds_disjoint()
    fresh = set(range(36000, 36050)) | set(range(39000, 39050))
    for a, b in cp.USED_SEEDS.values():
        assert not fresh & set(range(a, b + 1))
    assert (30000, 30049) in cp.USED_SEEDS.values() and (33000, 33049) in cp.USED_SEEDS.values()


def test_request_mapping_round_trips_targets():
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        m, app_mode = cp.request_for(targets)
        assert recommend._target_music_weighted(m, app_mode) == targets


def test_variants_call_the_intended_code_paths():
    n = 3000
    rng = np.random.RandomState(14)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    cat = mp.derive_moods(mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4), size=n)))[0]
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(36000)
        b = cp.recommender_for(cp.B, cat, None)(targets)
        np.random.seed(36000)
        assert np.array_equal(b, recommend._coverage_top(cat, targets, 30).index.to_numpy())
        np.random.seed(36000)
        nn = cp.recommender_for(cp.N, cat, None)(targets)
        np.random.seed(36000)
        assert np.array_equal(nn, recommend.recommend_music(cat, mood, ev.MODES[mode], top_n=30).index.to_numpy())


def test_gates_are_the_frozen_phase_2j1_gates():
    assert cp.GATES == bq.GATES


ARTIFACT = os.path.join(MODELS_DIR, "music_cell_production_check.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run `python evaluate_music_cell_production.py`")
def test_report_is_aggregate_only():
    with open(ARTIFACT, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)
    assert not re.search(r'"(track_id|track_name|artist|artists|title|uri|name)"\s*:', text)
    assert report["analysis_only"] is True and report["seeds"]["disjoint"] is True
    assert report["preregistration"] == cp.PREREGISTRATION and report["gates"] == cp.GATES


@requires_music_catalog
def test_report_pins_the_current_catalog():
    with open(ARTIFACT, encoding="utf-8") as f:
        report = json.load(f)
    assert report["source_artifact_sha256"]["catalog"] == mp.file_sha256(MUSIC_CATALOG_278K_PATH)
