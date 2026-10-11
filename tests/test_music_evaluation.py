"""Tests for the music recommender evaluation (src/music_evaluation.py) and its
report (evaluate_music_recommender.py -> models/music_recommender_eval.json)."""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from src import music_evaluation as ev
from src import music_pipeline as mp
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH

REPORT_PATH = os.path.join(MODELS_DIR, "music_recommender_eval.json")


def _catalog(n=1200, seed=1):
    """Full-shape synthetic catalog (random features and probabilities, derived moods)."""
    rng = np.random.RandomState(seed)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    catalog = mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4), size=n))
    return mp.derive_moods(catalog)[0]


# ---- metric definitions ----

def test_precision_recall_ndcg_on_known_lists():
    rel = np.array([1, 0, 1, 0, 0], dtype=float)
    assert ev.precision_at_k(rel, 2) == 0.5 and ev.precision_at_k(rel, 5) == 0.4
    assert ev.recall_at_k(rel, 5, n_relevant=4) == 0.5
    assert ev.recall_at_k(rel, 5, n_relevant=0) == 0.0
    expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))  # ideal: 2 relevant at the top (k=3)
    assert ev.ndcg_at_k(rel, 3, n_relevant=2) == pytest.approx(expected)
    assert ev.ndcg_at_k(np.ones(5), 5, n_relevant=100) == pytest.approx(1.0)
    assert ev.ndcg_at_k(np.zeros(5), 5, n_relevant=3) == 0.0


def test_diversity_and_jaccard():
    assert ev.intra_list_diversity(np.zeros((4, 9))) == 0.0
    assert ev.intra_list_diversity(np.array([[0.0, 0.0], [3.0, 4.0]])) == pytest.approx(5.0)
    assert ev.intra_list_diversity(np.zeros((1, 9))) == 0.0
    assert ev.jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)
    assert ev.jaccard([], []) == 1.0


def test_random_eligible_baseline_keeps_shares_and_candidates():
    catalog = _catalog()
    names = ["happy", "calm"]
    members = [ev.target_members(catalog, t) for t in names]
    rows = ev.random_eligible_list(members, [0.625, 0.375], 30, np.random.RandomState(0))
    assert len(rows) == len(set(rows.tolist())) == 30
    assert int(members[0][rows].sum()) == 19 and int(members[1][rows].sum()) == 11


# ---- the evaluation itself ----

def test_evaluate_on_a_synthetic_catalog():
    catalog = _catalog()
    out = ev.evaluate(catalog, runs=3, moods=["happy", "stressed"])
    assert set(out["requests"]) == {"happy / match", "happy / uplift", "stressed / match", "stressed / uplift"}
    for name, req in out["requests"].items():
        assert req["moodcast"]["duplicates"] == 0, name
        assert req["moodcast"]["precision@30"] == 1.0, name           # gating holds
        assert req["random_eligible_baseline"]["precision@30"] == 1.0
        assert 0 <= req["random_catalog_baseline"]["precision@30"] <= 1
        assert sum(req["mood_consistency"].values()) == pytest.approx(1.0)
        assert req["stability"]["same_seed_identical"] is True
    assert 0 < out["catalog_coverage"]["coverage"] <= 1
    assert "track_id" not in json.dumps(out)                         # aggregates only


requires_catalog = pytest.mark.skipif(not os.path.exists(MUSIC_CATALOG_278K_PATH),
                                      reason="run `python train_music_278k.py` to build the 278k catalog")


@requires_catalog
def test_catalog_checks_on_the_real_catalog():
    from src.data_loader import load_music_catalog
    checks = ev.catalog_checks(load_music_catalog())
    assert checks["rows"] == checks["unique_track_ids"] == 277_938
    assert checks["audio_features_missing_values"] == 0
    assert checks["labels"] == ["calm", "energetic", "happy", "sad"]
    assert checks["derived_columns_present"] and not checks["has_popularity_column"]


@pytest.mark.skipif(not os.path.exists(REPORT_PATH), reason="run `python evaluate_music_recommender.py`")
def test_committed_report_is_aggregate_only_and_complete():
    with open(REPORT_PATH, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)                  # no track IDs
    for key in ["evaluation_date", "catalog", "ks", "definitions", "requests", "overall_mean_over_requests",
                "catalog_coverage", "performance", "limitations"]:
        assert key in report, key
    assert report["catalog"]["rows"] == 277_938
    assert len(report["requests"]) == 2 * len(MOODS)
    assert all(r["moodcast"]["duplicates"] == 0 for r in report["requests"].values())
