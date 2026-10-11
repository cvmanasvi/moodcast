"""Tests for the analysis-only Phase 2J.1 confirmation (evaluate_music_cell_confirmation.py): strategy
routing, determinism, exact shares and the calm guard, the frozen gates, seed separation, unchanged
source artifacts and an aggregate-only report."""
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_balanced_quality as bq
import evaluate_music_cell_confirmation as cc
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from tests.local_data import requires_music_catalog, requires_music_classifier


@pytest.fixture(scope="module")
def catalog():
    n = 4000
    rng = np.random.RandomState(12)
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
def cells(catalog):
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    return bq.build_cells(catalog, (X - X.mean(axis=0)) / X.std(axis=0), guarded=True)


def test_seeds_are_fresh_and_separate():
    assert cc.seeds_disjoint()
    fresh_p, fresh_r = set(range(30000, 30050)), set(range(33000, 33050))
    assert not fresh_p & fresh_r
    for a, b in cc.USED_SEEDS.values():
        assert not (fresh_p | fresh_r) & set(range(a, b + 1))
    assert (24000, 24049) in cc.USED_SEEDS.values() and (27000, 27049) in cc.USED_SEEDS.values()


def test_routing_uses_cells_only_for_base_label_single_target_requests():
    routed = {f"{m} / {md}": cc.uses_cells(ev.request_targets(m, md)) for m in MOODS for md in ev.MODES}
    assert {k for k, v in routed.items() if v} == {"happy / match", "sad / match", "chill / match",
                                                    "energetic / match", "stressed / uplift"}


def test_t_equals_s_when_routed_and_g_otherwise(catalog, cells):
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(30000)
        t = cc.recommend_rows(catalog, targets, cc.T, cells)
        np.random.seed(30000)
        expected = bq.recommend_rows(catalog, targets, bq.S if cc.uses_cells(targets) else bq.G, cells)
        assert np.array_equal(t, expected), (mood, mode)
        if not cc.uses_cells(targets):
            np.random.seed(30000)
            assert np.array_equal(t, cc.recommend_rows(catalog, targets, cc.G, cells))


def test_production_variant_equals_production(catalog):
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(2)
        a = recommend._coverage_top(catalog, targets, 30).index.to_numpy()
        np.random.seed(2)
        assert np.array_equal(a, cc.recommend_rows(catalog, targets, cc.P, None))


@pytest.mark.parametrize("variant", cc.VARIANTS)
def test_variants_are_deterministic_with_exact_shares_and_guard(catalog, cells, variant):
    sp = catalog["speechiness"].to_numpy()
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(33000)
        rows = cc.recommend_rows(catalog, targets, variant, cells)
        np.random.seed(33000)
        assert np.array_equal(rows, cc.recommend_rows(catalog, targets, variant, cells))
        assert len(rows) == 30 and len(set(rows.tolist())) == 30
        for (target, _), share in zip(targets, recommend._allocate_shares([w for _, w in targets], 30)):
            member, _ = recommend._target_candidates(catalog, target)
            assert int(member[rows].sum()) == share
            if variant != cc.P and target == "calm":
                assert not (sp[rows[member[rows]]] > 0.66).any()


def test_gates_are_the_frozen_phase_2j_gates():
    assert cc.GATES == {
        "integrity": "100% target consistency, exact shares in every list, same-seed determinism on all 12 requests",
        "single_family_mean_gain_lower_bound_gt": 0.0,
        "single_family_configs_with_lower_bound_gt_0_min": 4,
        "single_config_regression_upper_bound_lt": -0.10,
        "multi_family_mean_change_lower_bound_gt": -0.10,
        "multi_config_change_lower_bound_gt": -0.25,
        "min_classifier_agreement_base_label_slots": 0.95,
        "min_mean_classifier_probability_base_label_slots": 0.95,
        "min_catalog_unique_tracks": 2000,
        "max_mean_repeat_rate": 0.75,
        "max_latency_p95_ms": 25.0,
        "max_calm_speech_like_share": 0.20,
        "calm_speech_like_not_worse_than_production": True,
    }


def test_zero_differences_do_not_count_as_gains_but_pass_noninferiority():
    single, multi = bq.families()
    zero = {"ci95": [0.0, 0.0]}
    gain = {"ci95": [0.1, 0.3]}
    paired = {bq.config_name(c): {"within_target_diversity": gain if cc.uses_cells(list(c)) else zero,
                                  "list_diversity": zero} for c in list(single) + list(multi)}
    res = {"summary": {"target_consistency": 1.0, "exact_target_shares_all_lists": True,
                       "all_same_seed_identical": True, "catalog_unique_tracks": 3000, "mean_repeat_rate": 0.7,
                       "latency_ms": {"p95": 5.0}},
           "relevance": {"classifier_agreement": 1.0, "mean_probability": 0.99},
           "speech_like_share": {"calm_requests": 0.0}, "paired_vs_reference": paired,
           "family_mean_change_vs_reference": {"single": gain, "multi": zero}}
    st = bq.gate_status(res, 0.18, single, multi)
    assert st[">=4_of_6_single_configs_within_gain_lb_>0"] == "PASS"   # exactly happy, sad, calm, energetic
    assert st["no_single_config_regression"] == "PASS"
    assert st["multi_family_list_diversity_noninferior_lb_>-0.10"] == "PASS"
    paired["happy:30"]["within_target_diversity"] = zero
    assert bq.gate_status(res, 0.18, single, multi)[">=4_of_6_single_configs_within_gain_lb_>0"] == "FAIL"


@requires_music_catalog
@requires_music_classifier
def test_source_artifacts_are_unchanged():
    with open(os.path.join(MODELS_DIR, "music_recommender_eval.json"), encoding="utf-8") as f:
        assert mp.file_sha256(MUSIC_CATALOG_278K_PATH) == json.load(f)["catalog"]["sha256"]
    with open(os.path.join(MODELS_DIR, "music_balanced_quality_analysis.json"), encoding="utf-8") as f:
        recorded = json.load(f)["source_artifact_sha256"]
    assert recorded == {"catalog": mp.file_sha256(MUSIC_CATALOG_278K_PATH),
                        "classifier": mp.file_sha256(bq.CLASSIFIER_PATH)}


ARTIFACT = os.path.join(MODELS_DIR, "music_cell_confirmation.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run `python evaluate_music_cell_confirmation.py`")
def test_report_is_aggregate_only_and_records_preregistration():
    with open(ARTIFACT, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)
    assert not re.search(r'"(track_id|track_name|artist|artists|title|uri|name)"\s*:', text)
    assert report["analysis_only"] is True and report["seeds"]["disjoint"] is True
    assert report["preregistration"] == cc.PREREGISTRATION and report["gates"] == cc.GATES


@requires_music_catalog
@requires_music_classifier
def test_report_matches_current_artifacts():
    with open(ARTIFACT, encoding="utf-8") as f:
        report = json.load(f)
    assert report["source_artifact_sha256"] == {"catalog": mp.file_sha256(MUSIC_CATALOG_278K_PATH),
                                                "classifier": mp.file_sha256(bq.CLASSIFIER_PATH)}
