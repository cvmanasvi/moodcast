"""Tests for the analysis-only Phase 2K.1 check (evaluate_music_calm_cell_guard.py): fresh seeds, the
variants it compares (K differs from current production only on the two calm-only requests), and an
aggregate-only report."""
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_balanced_quality as bq
import evaluate_music_calm_cell_guard as cg
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from tests.local_data import requires_music_catalog


def test_seeds_are_fresh():
    assert cg.seeds_disjoint()
    fresh = set(range(42000, 42050)) | set(range(45000, 45050))
    for a, b in cg.USED_SEEDS.values():
        assert not fresh & set(range(a, b + 1))
    assert (36000, 36049) in cg.USED_SEEDS.values() and (39000, 39049) in cg.USED_SEEDS.values()


def test_k_differs_from_current_production_only_on_calm_only_requests(monkeypatch):
    monkeypatch.setattr(recommend, "_cell_cache", {"catalog": None, "cells": None})
    n = 3000
    rng = np.random.RandomState(16)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    cat = mp.derive_moods(mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4), size=n)))[0]
    X = cat[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    cells = {"calm": bq.build_cells(cat, (X - X.mean(axis=0)) / X.std(axis=0), guarded=False)["calm"]}
    k1, k = cg.recommender_for(cg.K1, cat, cells), cg.recommender_for(cg.K, cat, cells)
    sp = cat["speechiness"].to_numpy()
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(42000)
        a = k1(targets)
        np.random.seed(42000)
        b = k(targets)
        if targets == [("calm", 1.0)]:
            assert not (sp[a] > 0.66).any() and (sp[b] > 0.66).any()   # guarded vs unguarded calm cells
        else:
            assert np.array_equal(a, b), (mood, mode)


ARTIFACT = os.path.join(MODELS_DIR, "music_calm_cell_guard_check.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run `python evaluate_music_calm_cell_guard.py`")
def test_report_is_aggregate_only():
    with open(ARTIFACT, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)
    assert not re.search(r'"(track_id|track_name|artist|artists|title|uri|name)"\s*:', text)
    assert report["analysis_only"] is True and report["seeds"]["disjoint"] is True
    assert report["preregistration"] == cg.PREREGISTRATION and report["prior_gates"] == bq.GATES


@requires_music_catalog
def test_report_pins_the_current_catalog():
    with open(ARTIFACT, encoding="utf-8") as f:
        report = json.load(f)
    assert report["source_artifact_sha256"]["catalog"] == mp.file_sha256(MUSIC_CATALOG_278K_PATH)
