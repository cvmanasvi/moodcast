"""Tests for the Phase 2L.2 supplementary calm pair (prepare_music_listening_pilot_calm.py): seed
separation, reproducibility, exact counts and uniqueness, speechiness exclusion, answer-key integrity,
anonymisation, the shortage stop, aggregate-only reporting, and untouched Phase 2L.1 materials."""
import csv
import hashlib
import json
import os
import re
import sys

import numpy as np
import pytest

import prepare_music_listening_pilot as lp
import prepare_music_listening_pilot_calm as cs
from src.config import MODELS_DIR
from src.data_loader import load_music_catalog
from tests.local_data import requires_music_catalog, requires_pilot_baseline, skip_if_missing

STRATEGY_WORDS = re.compile(r"previous|current|baseline|feature|cell|phase|2k|production|strategy|seed|speech",
                            re.IGNORECASE)


@pytest.fixture(scope="module")
def catalog():
    skip_if_missing(requires_music_catalog)
    return load_music_catalog().reset_index(drop=True)


def _run(base, monkeypatch, metadata=True):
    skip_if_missing(requires_music_catalog, requires_pilot_baseline)
    local, share, report = base / "local", base / "share", base / "report.json"
    if not metadata:
        monkeypatch.setattr(cs, "load_music_metadata", lambda: None)
    monkeypatch.setattr(sys, "argv", ["prep", "--local-dir", str(local), "--share-dir", str(share),
                                      "--report", str(report)])
    cs.main()
    return local, share, report


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    m = pytest.MonkeyPatch()
    try:
        yield _run(tmp_path_factory.mktemp("calm"), m)
    finally:
        m.undo()


def test_seeds_are_fresh_and_disjoint_from_every_earlier_set():
    assert cs.seeds_disjoint()
    for s in (cs.LIST_SEED, cs.SAMPLE_SEED, cs.BLIND_SEED):
        assert s not in set(lp.LIST_SEEDS.values()) | {lp.SAMPLE_SEED, lp.BLIND_SEED}
        for a, b in cs.USED_SEEDS.values():
            assert not a <= s <= b
    assert len({cs.LIST_SEED, cs.SAMPLE_SEED, cs.BLIND_SEED}) == 3


def test_counts_uniqueness_and_labels(pack):
    local, share, _ = pack
    sheet = _read(local / "calm_supplement_listening_sheet.csv")
    key = _read(local / "answer_key" / "calm_supplement_answer_key_items.csv")
    assert len(sheet) == len(key) == 16
    for strategy in lp.STRATEGY_NAMES.values():
        group = [r["track_id"] for r in key if r["strategy"] == strategy]
        assert len(group) == 8 and len(set(group)) == 8
    assert sorted(r["list_label"] for r in sheet) == ["A"] * 8 + ["B"] * 8
    assert {r["pair_id"] for r in sheet} == {"S1"}
    assert len(_read(share / "calm_supplement_ratings_tracks.csv")) == 16
    assert len(_read(share / "calm_supplement_ratings_lists.csv")) == 2
    assert len(_read(share / "calm_supplement_ratings_pairs.csv")) == 1


def test_speech_like_tracks_are_excluded_from_both_groups(pack, catalog):
    local, _, report = pack
    key = _read(local / "answer_key" / "calm_supplement_answer_key_items.csv")
    sp = catalog.set_index("track_id")["speechiness"]
    assert all(sp[r["track_id"]] <= lp.SPEECH_LIKE for r in key)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert all(v["sampled_speech_like"] == 0 for v in data["per_strategy"].values())


def test_answer_key_integrity(pack, catalog):
    local, _, _ = pack
    sheet = {r["item_id"]: r for r in _read(local / "calm_supplement_listening_sheet.csv")}
    key = _read(local / "answer_key" / "calm_supplement_answer_key_items.csv")
    lists_key = {r["list_label"]: r["strategy"] for r in _read(local / "answer_key" / "calm_supplement_answer_key_lists.csv")}
    assert sorted(sheet) == sorted(r["item_id"] for r in key)
    lists = cs.generate_lists(catalog, lp.load_baseline_module())
    by_name = {v: k for k, v in lp.STRATEGY_NAMES.items()}
    for r in key:
        row = lists[by_name[r["strategy"]]][int(r["position_in_list_of_30"]) - 1]
        assert catalog.iloc[row]["track_id"] == r["track_id"] and catalog.iloc[row]["mood"] == "calm"
        assert r["listen_url"] == sheet[r["item_id"]]["listen_url"] == f"https://open.spotify.com/track/{r['track_id']}"
        assert lists_key[r["list_label"]] == r["strategy"] and int(r["list_seed"]) == cs.LIST_SEED
    assert sorted(set(lists_key.values())) == sorted(lp.STRATEGY_NAMES.values())


def test_listener_materials_are_anonymous_and_labelled_supplementary(pack):
    local, share, _ = pack
    for path in [local / "calm_supplement_listening_sheet.csv", share / "calm_supplement_ratings_tracks.csv",
                 share / "calm_supplement_ratings_lists.csv", share / "calm_supplement_ratings_pairs.csv"]:
        rows = _read(path)
        for r in rows:
            assert all(not STRATEGY_WORDS.search(v) for k, v in r.items()
                       if k not in ("title", "artist", "listen_url")), path
            if "listen_url" in r:
                assert re.fullmatch(r"https://open\.spotify\.com/track/[0-9A-Za-z]{22}", r["listen_url"])
    for r in _read(local / "calm_supplement_listening_sheet.csv"):
        assert re.fullmatch(r"S1-[AB]0[1-8]", r["item_id"]) and "supplementary" in r["request"]
    assert "exploratory" in (share / "README.md").read_text(encoding="utf-8").lower()


def test_materials_are_reproducible(pack, tmp_path, monkeypatch):
    local, share, _ = pack
    local2, share2, _ = _run(tmp_path, monkeypatch)
    for a, b in [(local, local2), (local / "answer_key", local2 / "answer_key"), (share, share2)]:
        names = sorted(f for f in os.listdir(a) if os.path.isfile(a / f))
        assert names == sorted(f for f in os.listdir(b) if os.path.isfile(b / f))
        for n in names:
            assert (a / n).read_bytes() == (b / n).read_bytes(), n


def test_shortage_stops_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "eligible_positions", lambda catalog, rows: np.arange(7))
    with pytest.raises(SystemExit, match="Shortage"):
        _run(tmp_path, monkeypatch)
    assert not (tmp_path / "local").exists() and not (tmp_path / "share").exists()
    assert not (tmp_path / "report.json").exists()


def test_missing_metadata_is_marked_not_invented(tmp_path, monkeypatch):
    local, _, _ = _run(tmp_path, monkeypatch, metadata=False)
    assert all(r["title"].startswith("unavailable") for r in _read(local / "calm_supplement_listening_sheet.csv"))


def _assert_aggregate_only(text):
    assert not re.search(r"[0-9A-Za-z]{22}", re.sub(r"[0-9a-f]{40,64}", "", text))
    assert "open.spotify.com" not in text
    assert not re.search(r'"(track_id|track_name|title|artist|artists|uri|name)"\s*:', text)


def test_report_is_aggregate_only(pack):
    _, _, report = pack
    text = report.read_text(encoding="utf-8")
    _assert_aggregate_only(text)
    data = json.loads(text)
    assert data["ratings_collected"] == 0 and data["ratings_simulated"] is False
    assert data["exploratory_secondary_comparison"] is True and data["design"]["slots"] == 16


REPORT = os.path.join(MODELS_DIR, "music_listening_pilot_calm_supplement.json")


@pytest.mark.skipif(not os.path.exists(REPORT), reason="run `python prepare_music_listening_pilot_calm.py`")
def test_committed_report_is_aggregate_only():
    with open(REPORT, encoding="utf-8") as f:
        _assert_aggregate_only(f.read())


PACK_REPORT = os.path.join(MODELS_DIR, "music_listening_pilot_pack.json")


@pytest.mark.skipif(not os.path.exists(PACK_REPORT), reason="run `python prepare_music_listening_pilot.py`")
def test_phase_2l1_materials_are_unchanged():
    with open(PACK_REPORT, encoding="utf-8") as f:
        recorded = json.load(f)["files_sha256"]
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel, digest in recorded.items():
        path = os.path.join(base, rel)
        if rel.startswith("data/metadata/") and not os.path.exists(path):
            continue                                   # local-only files are absent on other machines
        with open(path, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, rel
