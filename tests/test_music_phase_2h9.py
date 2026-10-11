"""Tests for the pre-registered Phase 2H.9 evaluation (evaluate_music_phase_2h9.py): seed disjointness,
unique-configuration aggregation, guarded random-eligible baselines, gate bookkeeping, exact shares and
determinism of the evaluated variants, and an aggregate-only report."""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_calm_guard_analysis as guard
import evaluate_music_phase_2h9 as h9
import evaluate_music_recommender_analysis as sweep
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS


@pytest.fixture(scope="module")
def catalog():
    n = 2000
    rng = np.random.RandomState(8)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    cat = mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4), size=n))
    cat = mp.derive_moods(cat)[0]
    calm, _ = recommend._target_candidates(cat, "calm")
    cat.loc[cat.index[np.flatnonzero(calm)[:80]], "speechiness"] = 0.9     # speech-like calm tracks
    return cat


@pytest.fixture(scope="module")
def z(catalog):
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)


def test_fourth_seed_set_is_disjoint_from_every_earlier_seed():
    assert h9.seed_range() == (15000, 15049) and h9.seeds_disjoint()
    fourth = set(range(15000, 15050))
    for used in [range(0, 50), [2024], range(2026, 2076), range(7000, 7050), range(12000, 12050)]:
        assert not fourth & set(used)


def test_unique_configurations_merge_the_calm_aliases_only():
    configs = h9.unique_single_target_configurations()
    assert len(configs) == 6
    assert configs[(("calm", 30),)] == ["chill / match", "stressed / uplift"]
    assert all(len(names) == 1 for config, names in configs.items() if config != (("calm", 30),))
    assert {c[0][0] for c in configs} == {"happy", "sad", "angry", "calm", "energetic", "stressed"}


def test_configuration_key_is_targets_and_shares_not_display_name():
    assert h9.request_configuration("chill", "match") == h9.request_configuration("stressed", "uplift")
    assert h9.request_configuration("sad", "uplift") != h9.request_configuration("angry", "uplift")  # same targets, swapped shares


def test_production_eligibility_baseline_reproduces_the_existing_reference(catalog, z):
    ours = h9.random_eligible_baseline(catalog, z, 123, 5, recommend._target_candidates)
    theirs = sweep.random_eligible_reference(catalog, z, 5, 123)["requests"]
    assert ours == {n: v["diversity"] for n, v in theirs.items()}


def test_guarded_baseline_samples_only_guarded_calm_and_changes_only_calm_requests(catalog, z):
    plain = h9.random_eligible_baseline(catalog, z, 5, 10, recommend._target_candidates)
    guarded = h9.random_eligible_baseline(catalog, z, 5, 10, guard.guarded_candidates)
    calm_requests = set(guard.calm_requests())
    for name in plain:
        if name not in calm_requests:
            assert plain[name] == guarded[name], name
    assert any(plain[n] != guarded[n] for n in calm_requests)
    # the guarded population really excludes speech-like calm tracks
    member, _ = guard.guarded_candidates(catalog, "calm")
    rng = np.random.RandomState(0)
    rows = ev.random_eligible_list([member], [1.0], 30, rng)
    assert not (catalog["speechiness"].to_numpy()[rows] > 0.66).any()


def _fake_result(divs, speech=0.1, **summary):
    base = {"target_consistency": 1.0, "exact_target_shares_all_lists": True, "all_same_seed_identical": True,
            "mean_ranking_signal_of_served": 0.993, "catalog_unique_tracks": 2400, "mean_repeat_rate": 0.74,
            "latency_ms": {"p95": 10.0}}
    base.update(summary)
    return {"summary": base, "requests": {n: {"diversity": d} for n, d in divs.items()},
            "speech_like_share": {"calm_requests": speech}}


def _divs(calm=4.0, **over):
    d = {"happy / match": 3.0, "sad / match": 2.5, "angry / match": 3.5, "chill / match": calm,
         "energetic / match": 3.5, "stressed / match": 3.2, "stressed / uplift": calm}
    d.update(over)
    return d


def test_score_counts_the_calm_configuration_once():
    configs = h9.unique_single_target_configurations()
    baseline = {n: 3.0 for n in _divs()}
    paired = {r: {"bootstrap_95": [0.1, 0.2]} for r in h9.GATES["paired_lower_bound_positive_for"]}
    out = h9.score(_fake_result(_divs(calm=6.0)), baseline, configs, paired, 0.15)
    assert out["unique_single_target_mean_diversity"] == round(np.mean([3.0, 2.5, 3.5, 6.0, 3.5, 3.2]), 4)
    assert out["unique_configs_at_least_baseline"] == "5/6"        # only sad (2.5) is below 3.0
    assert all(v == "PASS" for v in out["gates"].values())


def test_score_gates_fail_and_flag_regressions():
    configs = h9.unique_single_target_configurations()
    baseline = {n: 3.0 for n in _divs()}
    paired = {r: {"bootstrap_95": [0.1, 0.2]} for r in h9.GATES["paired_lower_bound_positive_for"]}
    low = h9.score(_fake_result(_divs(calm=2.0, **{"happy / match": 2.0})), baseline, configs, paired, 0.15)
    assert low["gates"][">=5_of_6_unique_configs_at_least_baseline"] == "FAIL"   # 3/6
    assert low["gates"]["unique_single_target_mean_diversity_>=3.3"] == "FAIL"
    worse = h9.score(_fake_result(_divs(), speech=0.19), baseline, configs, paired, 0.15)
    assert worse["gates"]["calm_speech_like_<=0.20"] == "PASS"
    assert worse["calm_speech_like_vs_production"]["worse_than_production"] is True
    assert h9.score(_fake_result(_divs()), baseline, configs, None, 0.1)["gates"]["paired_lower_bounds_>0"] == \
        "NOT ESTABLISHED"


@pytest.mark.parametrize("name", h9.VARIANTS)
def test_variants_keep_exact_shares_and_are_deterministic(catalog, z, name):
    sp = catalog["speechiness"].to_numpy()
    for mood in MOODS:
        for mode in ev.MODES:
            targets = ev.request_targets(mood, mode)
            fn = guard.variants(catalog, z, {})[name]
            np.random.seed(15000)
            rows = fn(targets)
            np.random.seed(15000)
            assert np.array_equal(rows, fn(targets))
            assert len(rows) == 30 and len(set(rows.tolist())) == 30
            for (target, _), share in zip(targets, recommend._allocate_shares([w for _, w in targets], 30)):
                member, _ = recommend._target_candidates(catalog, target)
                assert int(member[rows].sum()) == share
                if name in h9.GUARDED_VARIANTS and target == "calm":
                    assert not (sp[rows[member[rows]]] > 0.66).any()


ARTIFACT = os.path.join(MODELS_DIR, "music_phase_2h9_evaluation.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run `python evaluate_music_phase_2h9.py`")
def test_report_is_aggregate_only_and_records_the_preregistration():
    with open(ARTIFACT, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)
    assert not re.search(r'"(track_id|track_name|artist|artists|title|uri)"', text)
    assert report["analysis_only"] is True
    assert report["preregistration"] == h9.PREREGISTRATION and report["gates"] == h9.GATES
    assert report["seeds"]["disjoint_from_previously_used"] is True
