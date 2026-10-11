"""Tests for the analysis-only Phase 2J balanced-quality study (evaluate_music_balanced_quality.py):
seed disjointness, configuration families, the production replica, the fixed feature cells and cell
selection, exact shares / guard / determinism for every variant, the diversity decomposition, relevance
and gate bookkeeping, unchanged source artifacts and an aggregate-only report."""
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_balanced_quality as bq
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from tests.local_data import requires_music_catalog, requires_music_classifier


@pytest.fixture(scope="module")
def catalog():
    n = 4000
    rng = np.random.RandomState(11)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    cat = mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4) * 0.3, size=n))
    cat = mp.derive_moods(cat)[0]
    calm, _ = recommend._target_candidates(cat, "calm")
    cat.loc[cat.index[np.flatnonzero(calm)[:40]], "speechiness"] = 0.9
    return cat


@pytest.fixture(scope="module")
def z(catalog):
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)


@pytest.fixture(scope="module")
def cells(catalog, z):
    return bq.build_cells(catalog, z, guarded=True)


def test_seed_sets_are_fresh_and_disjoint():
    assert bq.seeds_disjoint()
    fresh = set(range(24000, 24050)) | set(range(27000, 27050))
    for a, b in bq.USED_SEEDS.values():
        assert not fresh & set(range(a, b + 1))


def test_families_have_six_single_and_five_multi_configurations():
    single, multi = bq.families()
    assert len(single) == 6 and len(multi) == 5
    assert single[(("calm", 30),)] == ["chill / match", "stressed / uplift"]
    assert all(len(c) == 2 for c in multi)
    assert sum(len(n) for n in single.values()) + sum(len(n) for n in multi.values()) == 12


def test_production_variant_equals_production(catalog):
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(1)
        a = recommend._coverage_top(catalog, targets, 30).index.to_numpy()
        np.random.seed(1)
        assert np.array_equal(a, bq.recommend_rows(catalog, targets, bq.P, None)), (mood, mode)


def test_cells_partition_each_eligible_population_evenly_and_deterministically(catalog, z, cells):
    again = bq.build_cells(catalog, z, guarded=True)
    sp = catalog["speechiness"].to_numpy()
    for target, v in cells.items():
        member, signal = bq.guard.guarded_candidates(catalog, target)
        assert np.array_equal(v["labels"] >= 0, member)
        assert np.array_equal(v["labels"], again[target]["labels"])
        sizes = [len(c) for c in v["by_cell"]]
        assert len(sizes) == 30 and sum(sizes) == member.sum() and max(sizes) - min(sizes) <= 2
        for rows in v["by_cell"]:
            assert list(signal[rows]) == sorted(signal[rows], reverse=True)
        if target == "calm":
            assert not (sp[v["labels"] >= 0] > 0.66).any()


def test_cell_select_uses_distinct_cells_and_each_cells_top_candidates(catalog, cells):
    member, signal = bq.guard.guarded_candidates(catalog, "happy")
    np.random.seed(3)
    rows = bq.cell_select("happy", signal, member, 25, cells)
    labels = cells["happy"]["labels"][rows]
    assert len(rows) == 25 and len(set(labels.tolist())) == 25
    for r, c in zip(rows, labels):
        assert r in set(cells["happy"]["by_cell"][c][:bq.CELL_TOP].tolist())
    np.random.seed(3)
    assert np.array_equal(rows, bq.cell_select("happy", signal, member, 25, cells))


@pytest.mark.parametrize("variant", bq.VARIANTS)
def test_variants_keep_exact_shares_guard_and_seed(catalog, cells, variant):
    sp = catalog["speechiness"].to_numpy()
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(24000)
        rows = bq.recommend_rows(catalog, targets, variant, cells)
        np.random.seed(24000)
        assert np.array_equal(rows, bq.recommend_rows(catalog, targets, variant, cells))
        assert len(rows) == 30 and len(set(rows.tolist())) == 30
        for (target, _), share in zip(targets, recommend._allocate_shares([w for _, w in targets], 30)):
            member, _ = recommend._target_candidates(catalog, target)
            assert int(member[rows].sum()) == share
            if variant != bq.P and target == "calm":
                assert not (sp[rows[member[rows]]] > 0.66).any()


def test_decompose_matches_hand_computation():
    z = np.array([[0.0, 0], [3, 0], [0, 4], [10, 0]])
    rows = np.array([0, 1, 2, 3])
    a = np.array([True, True, False, False])
    b = ~a
    within, between = bq.decompose(rows, z, [a, b])
    assert within == pytest.approx(np.mean([3.0, np.hypot(10, 4)]))
    assert between == pytest.approx(np.mean([4.0, 10.0, 5.0, 7.0]))
    assert bq.decompose(rows, z, [np.ones(4, dtype=bool)])[1] is None


def test_relevance_counts_only_base_label_slots(catalog, cells):
    lists = {}
    for mood, mode in itertools.product(MOODS, ev.MODES):
        np.random.seed(0)
        lists[f"{mood} / {mode}"] = [bq.recommend_rows(catalog, ev.request_targets(mood, mode), bq.S, cells)]
    out = bq.relevance(catalog, lists)
    labels = catalog["mood"].to_numpy()
    expected = sum(int(np.isin(labels[l[0]], [t for t, _ in ev.request_targets(*n.split(" / "))
                                              if t in bq.LABEL_TARGETS]).sum()) for n, l in lists.items())
    assert sum(v["slots"] for v in out["by_target"].values()) == expected
    assert 0 <= out["classifier_agreement"] <= 1 and 0 <= out["mean_probability"] <= 1


def _fake(single_lb=0.1, multi_lb=0.0, agree=0.97, prob=0.97, speech=0.0):
    single, multi = bq.families()
    paired = {bq.config_name(c): {"within_target_diversity": {"ci95": [single_lb, 0.5]},
                                  "list_diversity": {"ci95": [multi_lb, 0.2]}} for c in list(single) + list(multi)}
    return {"summary": {"target_consistency": 1.0, "exact_target_shares_all_lists": True,
                        "all_same_seed_identical": True, "catalog_unique_tracks": 5000, "mean_repeat_rate": 0.5,
                        "latency_ms": {"p95": 5.0}},
            "relevance": {"classifier_agreement": agree, "mean_probability": prob},
            "speech_like_share": {"calm_requests": speech},
            "paired_vs_reference": paired,
            "family_mean_change_vs_reference": {"single": {"ci95": [single_lb, 0.4]},
                                                "multi": {"ci95": [multi_lb, 0.2]}}}, single, multi


def test_gate_status_logic():
    res, single, multi = _fake()
    assert all(v == "PASS" for v in bq.gate_status(res, 0.18, single, multi).values())
    res, single, multi = _fake(multi_lb=-0.3)
    st = bq.gate_status(res, 0.18, single, multi)
    assert st["multi_family_list_diversity_noninferior_lb_>-0.10"] == "FAIL"
    assert st["each_multi_config_list_diversity_lb_>-0.25"] == "FAIL"
    res, single, multi = _fake(single_lb=-0.01)
    st = bq.gate_status(res, 0.18, single, multi)
    assert st["single_family_within_gain_lb_>0"] == "FAIL" and st[">=4_of_6_single_configs_within_gain_lb_>0"] == "FAIL"
    assert bq.gate_status(_fake(agree=0.94)[0], 0.18, single, multi)["classifier_agreement_>=0.95"] == "FAIL"
    assert bq.gate_status(_fake(speech=0.19)[0], 0.18, single, multi)["calm_speech_like_<=0.20_and_<=production"] == "FAIL"


@requires_music_catalog
def test_source_artifacts_are_unchanged():
    with open(os.path.join(MODELS_DIR, "music_recommender_eval.json"), encoding="utf-8") as f:
        recorded = json.load(f)["catalog"]["sha256"]
    assert mp.file_sha256(MUSIC_CATALOG_278K_PATH) == recorded


ARTIFACT = os.path.join(MODELS_DIR, "music_balanced_quality_analysis.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run `python evaluate_music_balanced_quality.py`")
def test_report_is_aggregate_only():
    with open(ARTIFACT, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)
    assert not re.search(r'"(track_id|track_name|artist|artists|title|uri|name)"\s*:', text)
    assert report["analysis_only"] is True and report["seeds"]["disjoint"] is True
    assert report["preregistration"] == bq.PREREGISTRATION and report["gates"] == bq.GATES


@requires_music_catalog
@requires_music_classifier
def test_report_matches_current_artifacts():
    with open(ARTIFACT, encoding="utf-8") as f:
        report = json.load(f)
    assert report["source_artifact_sha256"] == {"catalog": mp.file_sha256(MUSIC_CATALOG_278K_PATH),
                                                "classifier": mp.file_sha256(bq.CLASSIFIER_PATH)}
