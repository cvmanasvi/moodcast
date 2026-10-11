"""Tests for the Phase 2L.1 listening-pilot preparation (prepare_music_listening_pilot.py): the exact
pre-Phase-2K baseline from Git, reproducibility, A/B anonymisation, exact sample counts, answer-key
integrity, honest metadata fallbacks, aggregate-only reporting and untouched production source."""
import csv
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

import prepare_music_listening_pilot as lp
from src import music_pipeline as mp
from src import recommend
from src.config import MODELS_DIR, MOODS
from src.data_loader import load_music_catalog
from tests.local_data import requires_music_catalog, requires_pilot_baseline, skip_if_missing

STRATEGY_WORDS = re.compile(r"previous|current|baseline|feature|cell|phase|2k|production|strategy|seed",
                            re.IGNORECASE)


@pytest.fixture(scope="module")
def catalog():
    skip_if_missing(requires_music_catalog)
    return load_music_catalog().reset_index(drop=True)


@pytest.fixture(scope="module")
def baseline():
    skip_if_missing(requires_pilot_baseline)
    return lp.load_baseline_module()


def _run(tmp_path, monkeypatch, name, metadata=True):
    skip_if_missing(requires_music_catalog, requires_pilot_baseline)
    local, share, report = tmp_path / name / "local", tmp_path / name / "share", tmp_path / name / "report.json"
    if not metadata:
        monkeypatch.setattr(lp, "load_music_metadata", lambda: None)
    monkeypatch.setattr(sys, "argv", ["prepare", "--local-dir", str(local), "--share-dir", str(share),
                                      "--report", str(report)])
    lp.main()
    return local, share, report


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    mp_ = pytest.MonkeyPatch()
    try:
        yield _run(tmp_path_factory.mktemp("pack"), mp_, "a")
    finally:
        mp_.undo()


# ---- baseline correctness ----

@requires_pilot_baseline
def test_baseline_is_the_exact_pre_phase_2k_source_from_git():
    content = lp.baseline_source()
    assert lp.git_blob_sha(content) == lp.BASELINE_BLOB
    text = content.decode("utf-8")
    assert "FEATURE_CELL_TARGETS" not in text and "return _coverage_top(df_music, music_targets, top_n)" in text


def test_baseline_reproduces_the_previous_ranking_for_every_request(catalog, baseline):
    for mood in MOODS:
        for mode in ("Match my mood", "Cheer me up"):
            targets = recommend._target_music_weighted(mood, mode)
            np.random.seed(48001)
            old = baseline.recommend_music(catalog, mood, mode, top_n=30)
            np.random.seed(48001)
            pd.testing.assert_frame_equal(old, recommend._coverage_top(catalog, targets, 30))


def test_pilot_moods_differ_between_strategies(catalog, baseline):
    lists = lp.generate_lists(catalog, baseline)
    for target, app in lp.PILOT:
        assert len(lists[(target, "previous")]) == len(lists[(target, "current")]) == 30
        assert not np.array_equal(lists[(target, "previous")], lists[(target, "current")])
        np.random.seed(lp.LIST_SEEDS[target])
        cur = recommend.recommend_music(catalog, app, "Match my mood", top_n=30)
        assert np.array_equal(lists[(target, "current")], catalog.index.get_indexer(cur.index))


# ---- counts, anonymisation, integrity ----

def test_exact_sample_counts(pack):
    local, share, _ = pack
    sheet = _read(local / "listening_sheet.csv")
    key = _read(local / "answer_key" / "answer_key_items.csv")
    assert len(sheet) == len(key) == 64
    per_list = {}
    for r in key:
        per_list.setdefault((r["target"], r["strategy"]), []).append(r)
    assert len(per_list) == 8 and all(len(v) == 8 for v in per_list.values())
    assert {r["pair_id"] for r in sheet} == {"P1", "P2", "P3", "P4"}
    for p in ("P1", "P2", "P3", "P4"):
        assert sorted(r["list_label"] for r in sheet if r["pair_id"] == p) == ["A"] * 8 + ["B"] * 8
    assert len(_read(share / "ratings_tracks.csv")) == 64
    assert len(_read(share / "ratings_lists.csv")) == 8 and len(_read(share / "ratings_pairs.csv")) == 4


def test_listener_materials_do_not_reveal_the_strategy(pack):
    local, share, _ = pack
    for path in [local / "listening_sheet.csv", share / "ratings_tracks.csv", share / "ratings_lists.csv",
                 share / "ratings_pairs.csv"]:
        rows = _read(path)
        assert not STRATEGY_WORDS.search(" ".join(rows[0].keys())), path
        for r in rows:
            # titles/artists are real song metadata; links are random base62 track ids (checked separately)
            assert all(not STRATEGY_WORDS.search(v) for k, v in r.items()
                       if k not in ("title", "artist", "listen_url")), path
            if "listen_url" in r:
                assert re.fullmatch(r"https://open\.spotify\.com/track/[0-9A-Za-z]{22}", r["listen_url"])
    for r in _read(local / "listening_sheet.csv"):
        assert re.fullmatch(r"P[1-4]-[AB]0[1-8]", r["item_id"])
        assert r["item_id"].startswith(f'{r["pair_id"]}-{r["list_label"]}')


def test_ab_labels_and_orders_are_randomised(pack):
    local, _, _ = pack
    lists = _read(local / "answer_key" / "answer_key_lists.csv")
    previous_labels = [r["list_label"] for r in lists if r["strategy"].startswith("previous")]
    assert len(previous_labels) == 4
    key = _read(local / "answer_key" / "answer_key_items.csv")
    positions = [int(r["position_in_list_of_30"]) for r in sorted(key, key=lambda r: r["item_id"])]
    assert positions != sorted(positions)                     # display order is not rank order
    assert max(positions) > 8                                 # not just the top of each list


def test_answer_key_integrity(pack, catalog, baseline):
    local, _, _ = pack
    sheet = {r["item_id"]: r for r in _read(local / "listening_sheet.csv")}
    key = _read(local / "answer_key" / "answer_key_items.csv")
    assert sorted(sheet) == sorted(r["item_id"] for r in key)
    lists = lp.generate_lists(catalog, baseline)
    strategy_key = {v: k for k, v in lp.STRATEGY_NAMES.items()}
    seen = {}
    list_map = {(r["pair_id"], r["list_label"]): r["strategy"] for r in _read(local / "answer_key" / "answer_key_lists.csv")}
    for r in key:
        s = strategy_key[r["strategy"]]
        row = lists[(r["target"], s)][int(r["position_in_list_of_30"]) - 1]
        assert catalog.iloc[row]["track_id"] == r["track_id"]
        assert catalog.iloc[row]["mood"] == r["target"]
        assert r["listen_url"] == sheet[r["item_id"]]["listen_url"] == f"https://open.spotify.com/track/{r['track_id']}"
        assert int(r["list_seed"]) == lp.LIST_SEEDS[r["target"]]
        assert list_map[(r["pair_id"], r["list_label"])] == r["strategy"]
        seen.setdefault((r["target"], s), set()).add(r["position_in_list_of_30"])
    assert all(len(v) == 8 for v in seen.values())             # 8 distinct positions per list


# ---- reproducibility and metadata honesty ----

def test_materials_are_reproducible(pack, tmp_path, monkeypatch):
    local, share, _ = pack
    local2, share2, _ = _run(tmp_path, monkeypatch, "b")
    for a, b in [(local, local2), (local / "answer_key", local2 / "answer_key"), (share, share2)]:
        names = sorted(f for f in os.listdir(a) if os.path.isfile(a / f))
        assert names == sorted(f for f in os.listdir(b) if os.path.isfile(b / f))
        for n in names:
            assert (a / n).read_bytes() == (b / n).read_bytes(), n


def test_missing_metadata_is_marked_not_invented(tmp_path, monkeypatch):
    local, _, report = _run(tmp_path, monkeypatch, "nometa", metadata=False)
    for r in _read(local / "listening_sheet.csv"):
        assert r["title"].startswith("unavailable") and r["artist"].startswith("unavailable")
    assert json.loads(report.read_text(encoding="utf-8"))["local_metadata_available"] is False


# ---- privacy and production safety ----

def _assert_aggregate_only(text):
    assert not re.search(r"[0-9A-Za-z]{22}", re.sub(r"[0-9a-f]{40,64}", "", text))   # no track ids (hashes removed)
    assert "open.spotify.com" not in text and "spotify:track" not in text
    assert not re.search(r'"(track_id|track_name|title|artist|artists|uri|name)"\s*:', text)


def test_report_is_aggregate_only(pack):
    _, _, report = pack
    text = report.read_text(encoding="utf-8")
    _assert_aggregate_only(text)
    data = json.loads(text)
    assert data["ratings_collected"] == 0 and data["ratings_simulated"] is False
    assert data["design"]["slots"] == 64 and data["baseline"]["git_blob_src_recommend"] == lp.BASELINE_BLOB


REPORT = os.path.join(MODELS_DIR, "music_listening_pilot_pack.json")


@pytest.mark.skipif(not os.path.exists(REPORT), reason="run `python prepare_music_listening_pilot.py`")
def test_committed_report_is_aggregate_only():
    with open(REPORT, encoding="utf-8") as f:
        _assert_aggregate_only(f.read())


@requires_music_catalog
def test_production_source_is_unchanged():
    with open(os.path.join(MODELS_DIR, "music_calm_cell_guard_check.json"), encoding="utf-8") as f:
        recorded = json.load(f)["source_artifact_sha256"]
    assert mp.file_sha256(recommend.__file__) == recorded["src_recommend"]
    assert mp.file_sha256(lp.MUSIC_CATALOG_278K_PATH) == recorded["catalog"]
