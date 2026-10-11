"""Tests for the pre-registered Phase 2I.1 confirmation (evaluate_music_pool_confirmation.py): seed
disjointness, the confidence-matched baseline draws from exactly the strategy's pool, exact shares,
determinism, calm guard, served-confidence reporting, gate bookkeeping and an aggregate-only report."""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_phase_2h9 as h9
import evaluate_music_pool_confirmation as pc
import evaluate_music_pool_redesign as pr
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS


@pytest.fixture(scope="module")
def catalog():
    n = 3000
    rng = np.random.RandomState(10)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    cat = mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4) * 0.2, size=n))
    cat = mp.derive_moods(cat)[0]
    calm, _ = recommend._target_candidates(cat, "calm")
    cat.loc[cat.index[np.flatnonzero(calm)[:50]], "speechiness"] = 0.9
    return cat


@pytest.fixture(scope="module")
def z(catalog):
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)


def test_seed_set_is_fresh():
    assert pc.seed_range() == (21000, 21049) and pc.seeds_disjoint()
    fresh = set(range(21000, 21050))
    for used in [range(0, 50), [2024], range(2026, 2076), range(7000, 7050), range(12000, 12050),
                 range(15000, 15050), range(18000, 18050)]:
        assert not fresh & set(used)


@pytest.mark.parametrize("name", list(pc.VARIANTS))
def test_variants_keep_exact_shares_guard_and_seed(catalog, name):
    _, guarded = pc.VARIANTS[name]
    sp = catalog["speechiness"].to_numpy()
    for mood in MOODS:
        for mode in ev.MODES:
            targets = ev.request_targets(mood, mode)
            np.random.seed(21000)
            rows = pc.fn(catalog, name)(targets)
            np.random.seed(21000)
            assert np.array_equal(rows, pc.fn(catalog, name)(targets))
            assert len(rows) == 30 and len(set(rows.tolist())) == 30
            for (target, _), share in zip(targets, recommend._allocate_shares([w for _, w in targets], 30)):
                member, _ = recommend._target_candidates(catalog, target)
                assert int(member[rows].sum()) == share
                if guarded and target == "calm":
                    assert not (sp[rows[member[rows]]] > 0.66).any()


def test_baseline_population_is_the_strategys_pool(catalog):
    spec, guarded = pc.VARIANTS[pc.PRIMARY]
    sp = catalog["speechiness"].to_numpy()
    for target in pr.LABEL_TARGETS:
        mask = pc.baseline_population(catalog, target, 30, spec, guarded)
        member, signal = pr.guard.guarded_candidates(catalog, target)
        pool, fell_back = pr.pool_rows(target, signal, member, 30, spec)
        assert set(np.flatnonzero(mask)) == set(pool.tolist())
        if not fell_back:
            assert (signal[mask] >= 0.99).all()
        if target == "calm":
            assert not (sp[mask] > 0.66).any()
    for target in pc.DERIVED:                      # no invented probability floor
        member, _ = recommend._target_candidates(catalog, target)
        assert np.array_equal(pc.baseline_population(catalog, target, 30, spec, guarded), member)


def test_matched_baseline_without_restriction_equals_the_unrestricted_one(catalog, z):
    open_spec = {"kind": "threshold", "tau": -1.0, "cap_multiplier": None}
    assert pc.matched_baseline(catalog, z, 21000, 5, open_spec, False) == \
        h9.random_eligible_baseline(catalog, z, 21000, 5, recommend._target_candidates)


def test_served_confidence_reports_every_base_label_slot(catalog):
    np.random.seed(0)
    lists = {f"{m} / {md}": [pc.fn(catalog, pc.PRIMARY)(ev.request_targets(m, md))] for m in MOODS for md in ev.MODES}
    out = pc.served_confidence(catalog, lists)
    assert set(out["by_target"]) == set(pr.LABEL_TARGETS)
    for t, v in out["by_target"].items():
        assert v["min"] <= v["p5"] <= v["mean"] <= 1.0 and v["slots"] > 0
    labels = catalog["mood"].to_numpy()
    # slots of base-label targets only: derived (angry/stressed) slots also carry a base label but are excluded
    total = 0
    for name, l in lists.items():
        mood, mode = name.split(" / ")
        base_targets = [t for t, _ in ev.request_targets(mood, mode) if t in pr.LABEL_TARGETS]
        total += int(np.isin(labels[np.concatenate(l)], base_targets).sum())
    assert sum(v["slots"] for v in out["by_target"].values()) == total


def test_paired_gate_components_and_combination():
    paired = {r: {"bootstrap_95": [0.1, 0.3]} for r in pc.PAIRED_REQUESTS}
    assert pc.paired_component(pc.PRIMARY, "stressed / match", paired)[0] == "NOT ESTABLISHED"
    assert pc.paired_component(pc.PRIMARY, "sad / match", paired)[0] == "PASS"
    paired["happy / match"] = {"bootstrap_95": [-0.01, 0.2]}
    assert pc.paired_component(pc.PRIMARY, "happy / match", paired)[0] == "FAIL"
    assert pc.combine(["PASS", "NOT ESTABLISHED"]) == "NOT ESTABLISHED"
    assert pc.combine(["PASS", "NOT ESTABLISHED", "FAIL"]) == "FAIL"
    assert pc.combine(["PASS", "PASS"]) == "PASS"


def _res(speech=0.0, conf=0.995, calm=3.0):
    divs = {"happy / match": 3.0, "sad / match": 2.5, "angry / match": 3.2, "chill / match": calm,
            "energetic / match": 3.4, "stressed / match": 3.3, "stressed / uplift": calm}
    return {"summary": {"target_consistency": 1.0, "exact_target_shares_all_lists": True,
                        "all_same_seed_identical": True, "catalog_unique_tracks": 5000, "mean_repeat_rate": 0.3,
                        "latency_ms": {"p95": 4.0}},
            "requests": {n: {"diversity": d} for n, d in divs.items()},
            "speech_like_share": {"calm_requests": speech},
            "served_confidence": {"mean_all_base_label_slots": conf}}


def test_score_uses_unique_configs_matched_baselines_and_calm_vs_production():
    configs = h9.unique_single_target_configurations()
    matched = {n: 2.9 for n in _res()["requests"]}
    paired = {r: {"bootstrap_95": [0.1, 0.3]} for r in pc.PAIRED_REQUESTS}
    out = pc.score(pc.PRIMARY, _res(), matched, configs, paired, prod_speech=0.18)
    assert out["unique_single_target_mean_diversity"] == round(np.mean([3.0, 2.5, 3.2, 3.0, 3.4, 3.3]), 4)
    assert out["unique_configs_at_least_confidence_matched_baseline"] == "5/6"
    assert out["gates"]["paired_lower_bounds_>0"] == "NOT ESTABLISHED"          # stressed component
    assert out["gates"]["calm_speech_like_<=0.20_and_not_worse_than_production"] == "PASS"
    worse = pc.score(pc.PRIMARY, _res(speech=0.19), matched, configs, paired, prod_speech=0.18)
    assert worse["gates"]["calm_speech_like_<=0.20_and_not_worse_than_production"] == "FAIL"
    low = pc.score(pc.PRIMARY, _res(conf=0.989), matched, configs, paired, prod_speech=0.18)
    assert low["gates"]["mean_served_classifier_probability_>=0.99"] == "FAIL"


ARTIFACT = os.path.join(MODELS_DIR, "music_pool_confirmation.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run `python evaluate_music_pool_confirmation.py`")
def test_report_is_aggregate_only_and_records_preregistration():
    with open(ARTIFACT, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)
    assert not re.search(r'"(track_id|track_name|artist|artists|title|uri)"', text)
    assert report["analysis_only"] is True and report["seeds"]["disjoint_from_previously_used"] is True
    assert report["preregistration"] == pc.PREREGISTRATION and report["gates"] == pc.GATES
