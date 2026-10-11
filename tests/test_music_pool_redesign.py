"""Tests for the analysis-only candidate-pool redesign (evaluate_music_pool_redesign.py): seed
disjointness, the control equals production, every strategy keeps eligibility, exact shares, uniqueness
and same-seed reproducibility, thresholds only ever widen the pool, baselines are comparable with Phase
2H.9, and the report is aggregate-only."""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import evaluate_music_calm_guard_analysis as guard
import evaluate_music_phase_2h9 as h9
import evaluate_music_pool_redesign as pr
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS


@pytest.fixture(scope="module")
def catalog():
    n = 3000
    rng = np.random.RandomState(9)
    raw = pd.DataFrame({f: rng.rand(n) for f in mp.AUDIO_FEATURES})
    raw["uri"] = [f"spotify:track:T{i:021d}" for i in range(n)]
    raw["labels"] = [i % 4 for i in range(n)]
    raw["duration (ms)"] = 200000.0
    cat = mp.build_catalog(mp.build_training_frame(raw), rng.dirichlet(np.ones(4) * 0.3, size=n))
    cat = mp.derive_moods(cat)[0]
    calm, _ = recommend._target_candidates(cat, "calm")
    cat.loc[cat.index[np.flatnonzero(calm)[:60]], "speechiness"] = 0.9
    return cat


@pytest.fixture(scope="module")
def z(catalog):
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)


def test_held_out_seeds_are_new_and_disjoint():
    assert pr.held_out_disjoint()
    held = set(range(18000, 18050))
    for used in [range(0, 50), [2024], range(2026, 2076), range(7000, 7050), range(12000, 12050),
                 range(15000, 15050)]:
        assert not held & set(used)


def test_control_equals_production(catalog):
    for mood in MOODS:
        for mode in ev.MODES:
            targets = ev.request_targets(mood, mode)
            for seed in range(2):
                np.random.seed(seed)
                a = recommend._coverage_top(catalog, targets, 30).index.to_numpy()
                np.random.seed(seed)
                b = pr.coverage_top_pool(catalog, targets, 30, pr.STRATEGIES[pr.CONTROL])
                assert np.array_equal(a, b), (mood, mode)


@pytest.mark.parametrize("name", pr.variant_names("C: threshold 0.95"))
def test_every_variant_keeps_shares_eligibility_and_seed(catalog, name):
    spec, guarded = pr.variant(name)
    sp = catalog["speechiness"].to_numpy()
    for mood in MOODS:
        for mode in ev.MODES:
            targets = ev.request_targets(mood, mode)
            np.random.seed(5)
            rows = pr.coverage_top_pool(catalog, targets, 30, spec, guarded)
            np.random.seed(5)
            assert np.array_equal(rows, pr.coverage_top_pool(catalog, targets, 30, spec, guarded))
            assert len(rows) == 30 and len(set(rows.tolist())) == 30
            for (target, _), share in zip(targets, recommend._allocate_shares([w for _, w in targets], 30)):
                member, _ = recommend._target_candidates(catalog, target)
                assert int(member[rows].sum()) == share, (name, mood, mode, target)
                if guarded and target == "calm":
                    assert not (sp[rows[member[rows]]] > 0.66).any()


def test_thresholds_only_widen_the_pool_and_count_fallbacks(catalog):
    member, signal = recommend._target_candidates(catalog, "happy")
    production = set(pr.pool_rows("happy", signal, member, 30, pr.STRATEGIES[pr.CONTROL])[0].tolist())
    for tau in (0.99, 0.5, 0.0):
        idx, fell_back = pr.pool_rows("happy", signal, member, 30, {"kind": "threshold", "tau": tau,
                                                                    "cap_multiplier": None})
        assert len(idx) >= len(production)
        if fell_back:
            assert set(idx.tolist()) == production
        else:
            assert (signal[idx] >= tau).all() and len(idx) == int((member & (signal >= tau)).sum())
    _, fell_back = pr.pool_rows("happy", signal, member, 30, {"kind": "threshold", "tau": 1.01, "cap_multiplier": None})
    assert fell_back


def test_cap_keeps_the_best_by_signal_and_derived_targets_are_untouched(catalog):
    member, signal = recommend._target_candidates(catalog, "sad")
    idx, _ = pr.pool_rows("sad", signal, member, 3, {"kind": "threshold", "tau": 0.0, "cap_multiplier": 100})
    assert len(idx) == 300
    assert signal[idx].min() >= np.sort(signal[member])[::-1][299]
    m2, s2 = recommend._target_candidates(catalog, "angry")
    spec = {"kind": "threshold", "tau": 0.0, "cap_multiplier": None}
    assert set(pr.pool_rows("angry", s2, m2, 5, spec)[0]) == set(
        pr.pool_rows("angry", s2, m2, 5, pr.STRATEGIES[pr.CONTROL])[0])


def test_baselines_and_scoring_are_the_phase_2h9_ones():
    assert pr.GATES == h9.GATES
    assert len(h9.unique_single_target_configurations()) == 6


def test_gap_closed_is_relative_to_the_control():
    configs = {(("sad", 30),): ["sad / match"], (("angry", 30),): ["angry / match"]}
    control = {"requests": {"sad / match": {"diversity": 1.0}, "angry / match": {"diversity": 3.5}}}
    res = {"requests": {"sad / match": {"diversity": 2.0}, "angry / match": {"diversity": 3.6}}}
    assert pr.gap_closed(res, control, {"sad / match": 3.0, "angry / match": 3.0}, configs) == {"sad": 0.5}


def test_choose_prefers_full_passes_and_ignores_the_control():
    def r(div, ok, soft=True):
        gates = {k: "PASS" for k in ["ranking_signal_>=0.99", "repeat_rate_<=0.75", "p95_latency_<=25ms", "x"]}
        if not ok:
            gates["x"] = "FAIL"
        if not soft:
            gates["ranking_signal_>=0.99"] = "FAIL"
        return {"unique_single_target_mean_diversity": div, "gates": gates}
    res = {pr.CONTROL: r(9.0, True), "B1: top 40x": r(3.0, False), "B2: top 80x": r(3.2, False, soft=False)}
    assert pr.choose(res) == ("B1: top 40x", False)
    res["C: threshold 0.99"] = r(2.9, True)
    assert pr.choose(res) == ("C: threshold 0.99", True)


ARTIFACT = os.path.join(MODELS_DIR, "music_pool_redesign_analysis.json")


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run `python evaluate_music_pool_redesign.py`")
def test_report_is_aggregate_only():
    with open(ARTIFACT, encoding="utf-8") as f:
        text = f.read()
    report = json.loads(text)
    assert not re.search(r'"[0-9A-Za-z]{22}"', text)
    assert not re.search(r'"(track_id|track_name|artist|artists|title|uri)"', text)
    assert report["analysis_only"] is True and report["seeds"]["held_out_disjoint"] is True
    assert report["gates"] == h9.GATES and report["strategies"] == json.loads(json.dumps(pr.STRATEGIES))
