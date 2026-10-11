"""Tests for the preferred-artist layer (src/artist_priority.py, config/artist_priority.json).

Covers exact artist matching from local metadata only, mood relevance (same
target, never a less relevant track), the share and per-artist caps, honest
handling of tracks without metadata, and fallback to the recommender's list.
Synthetic fixtures use made-up track ids and artist names and are labelled so.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import flask_app
from src import artist_priority as ap
from src import recommend
from src.data_loader import load_music_catalog, load_music_metadata
from src.music_metadata import METADATA_COLUMNS, MetadataStore
from tests.local_data import requires_music_catalog, skip_if_missing

MOODS = ["happy", "sad", "chill", "energetic", "angry", "stressed"]
MODES = ["Match my mood", "Cheer me up"]


@pytest.fixture(scope="module")
def catalog():
    skip_if_missing(requires_music_catalog)
    return load_music_catalog()


@pytest.fixture(scope="module")
def metadata():
    md = load_music_metadata()
    if md is None:
        pytest.skip("local music metadata not built (python build_music_metadata.py)")
    return md


@pytest.fixture(scope="module")
def config():
    """The shipped config, switched on explicitly (it is off by default)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MOODCAST_ARTIST_PRIORITY", "1")
        cfg = ap.load_config()
    assert cfg is not None
    return cfg


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("MOODCAST_ARTIST_PRIORITY", "1")


@pytest.fixture(scope="module")
def index(catalog, metadata, config):
    return ap.get_index(catalog, metadata, config)


def _store(rows):
    """A MetadataStore from (track_id, title, [artists]) -- SYNTHETIC test data."""
    table = pd.DataFrame([{"track_id": t, "track_name": title, "artist_name": ", ".join(artists),
                           "artist_names_json": json.dumps(artists), "release_year": "2020", "metadata_source": "test"}
                          for t, title, artists in rows], columns=METADATA_COLUMNS)
    return MetadataStore(table)


# ---------------------------------------------------------------- config + matching

def test_every_configured_artist_is_really_in_the_local_metadata(metadata, config):
    present = {name for tid in metadata.track_ids() for name in metadata.artists_of(tid)}
    missing = [a for cat in config["categories"].values() for a in cat["artists"] if a not in present]
    assert missing == [], f"configured artists with no tracks in the local metadata: {missing}"


def test_ambiguous_names_are_not_configured(config):
    names = {a for cat in config["categories"].values() for a in cat["artists"]}
    for ambiguous in ["Jin", "King", "Mukesh", "V", "RM"]:
        assert ambiguous not in names


def test_shipped_config_is_off_by_default_and_env_switches_it(monkeypatch, tmp_path):
    with open(ap.CONFIG_PATH, encoding="utf-8") as f:
        assert json.load(f)["enabled"] is False                     # off unless explicitly switched on
    monkeypatch.delenv("MOODCAST_ARTIST_PRIORITY", raising=False)
    assert ap.load_config() is None                                 # unset: the file decides (off)
    for on in ["1", "true", "YES", " on "]:
        monkeypatch.setenv("MOODCAST_ARTIST_PRIORITY", on)
        assert ap.load_config() is not None, on                     # explicit enable
    on_file = tmp_path / "on.json"
    on_file.write_text(json.dumps({"enabled": True, "categories": {"x": {"artists": ["A"]}}}), encoding="utf-8")
    monkeypatch.setenv("MOODCAST_ARTIST_PRIORITY", "0")
    assert ap.load_config(str(on_file)) is None                     # explicit disable wins over the file
    monkeypatch.delenv("MOODCAST_ARTIST_PRIORITY")
    assert ap.load_config(str(on_file)) is not None
    monkeypatch.setenv("MOODCAST_ARTIST_PRIORITY", "1")
    assert ap.load_config(str(tmp_path / "missing.json")) is None   # unreadable config: off, even if asked


def test_matching_is_exact_and_uses_only_local_metadata():
    ids = [f"{i:022d}" for i in range(5)]
    catalog = pd.DataFrame({"track_id": ids})
    store = _store([(ids[0], "Synthetic A", ["Synthetic Star"]),
                    (ids[1], "Synthetic B", ["synthetic star"]),                 # different case: not a match
                    (ids[2], "Synthetic C", ["Someone Else", "Synthetic Idol"]),  # collaboration: matched via 2nd artist
                    (ids[3], "Synthetic D", ["Someone Else"])])                   # ids[4] has no metadata at all
    cfg = {"max_share": 0.3, "max_per_artist": 2, "categories": {
        "international": {"label": "International pop", "artists": ["Synthetic Star"]},
        "kpop": {"label": "K-pop", "artists": ["Synthetic Idol"]}}}
    idx = ap.ArtistIndex(catalog, store, cfg)
    assert idx.category.tolist() == ["international", "", "kpop", "", ""]
    assert idx.artist.tolist() == ["Synthetic Star", "", "Synthetic Idol", "", ""]
    assert ap.ArtistIndex(catalog, None, cfg).category.tolist() == [""] * 5   # no metadata: nothing is a priority track


def test_requested_anchor_artists_are_found(index):
    for name, category in [("Taylor Swift", "international"), ("BTS", "kpop"), ("BLACKPINK", "kpop"),
                           ("Arijit Singh", "indian"), ("A.R. Rahman", "indian"), ("Shreya Ghoshal", "indian")]:
        rows = np.flatnonzero(index.artist == name)
        assert len(rows) > 0 and set(index.category[rows]) == {category}, name


# ---------------------------------------------------------------- ranking behaviour on the real catalog

def _lists(catalog, index, config, seeds=range(4), **overrides):
    cfg = {**config, **overrides}
    for mood in MOODS:
        for mode in MODES:
            for seed in seeds:
                np.random.seed(seed)
                recs = recommend.recommend_music(catalog, mood, mode, top_n=30)
                yield mood, mode, recs, ap.prioritize(catalog, recs, index, cfg)


def test_mood_relevance_is_preserved_slot_by_slot(catalog, index, config):
    for mood, mode, recs, out in _lists(catalog, index, config):
        assert len(out) == len(recs) and out.track_id.is_unique, (mood, mode)
        assert out.target.tolist() == recs.target.tolist(), (mood, mode)            # same target in every slot
        assert (out.score.to_numpy() >= recs.score.to_numpy() - 1e-12).all(), (mood, mode)   # never less relevant
        added = ~out.track_id.isin(set(recs.track_id)).to_numpy()
        for target in set(out.target):
            member, _ = recommend._target_candidates(catalog, target)     # the recommender's own membership rule
            guarded, _ = recommend._cell_candidates(catalog, target)      # swaps also keep calm's speech-like guard
            in_target = (out.target == target).to_numpy()
            assert member[catalog.index.get_indexer(out.index[in_target])].all(), (mood, mode, target)
            assert guarded[catalog.index.get_indexer(out.index[in_target & added])].all(), (mood, mode, target)


def test_swaps_are_capped_and_artists_stay_diverse(catalog, index, config, metadata):
    for mood, mode, recs, out in _lists(catalog, index, config):
        swapped = (~out.track_id.isin(set(recs.track_id))).sum()
        assert swapped <= int(config["max_share"] * 30), (mood, mode)
        counts = Counter(a for tid in out.track_id for a in metadata.artists_of(tid))
        new_artists = {a for tid in out.track_id[~out.track_id.isin(set(recs.track_id))] for a in metadata.artists_of(tid)}
        assert all(counts[a] <= config["max_per_artist"] for a in new_artists), (mood, mode)
    for _, _, recs, out in _lists(catalog, index, config, seeds=[0], max_per_artist=1):
        added = out.track_id[~out.track_id.isin(set(recs.track_id))]
        counts = Counter(a for tid in out.track_id for a in metadata.artists_of(tid))
        assert all(counts[a] <= 1 for tid in added for a in metadata.artists_of(tid))


def test_preferred_artists_appear_where_the_mood_allows(catalog, index, config):
    seen = {}
    for mood, mode, recs, out in _lists(catalog, index, config):
        seen.setdefault((mood, mode), Counter()).update(out.artist_category.dropna())
    for key in [("happy", "Match my mood"), ("sad", "Match my mood"), ("energetic", "Match my mood")]:
        assert len(seen[key]) >= 2 and sum(seen[key].values()) > 0, key       # several categories, not one artist


def test_categories_are_only_shown_with_verified_artists(catalog, index, config, metadata):
    names = {k: set(v["artists"]) for k, v in config["categories"].items()}
    label_key = {v["label"]: k for k, v in config["categories"].items()}
    for _, _, _, out in _lists(catalog, index, config, seeds=[1]):
        for tid, label in zip(out.track_id, out.artist_category):
            if label is not None:
                assert set(metadata.artists_of(tid)) & names[label_key[label]], tid
            else:                                                             # unmarked: no configured artist on it
                assert not set(metadata.artists_of(tid)) & set().union(*names.values()), tid


def test_order_within_targets_sorts_each_target_in_its_own_slots():
    targets = np.array(["a", "b", "a", "a", "b"], dtype=object)
    scores = np.array([0.5, 0.2, 0.9, 0.5, 0.7])
    order = ap._order_within_targets(targets, scores)
    assert targets[order].tolist() == targets.tolist()                 # every slot keeps its target
    assert scores[order].tolist() == [0.9, 0.7, 0.5, 0.5, 0.2]         # best first per target, ties keep slot order
    assert order.tolist() == [2, 4, 0, 3, 1]


def test_each_target_is_best_first_with_the_recommenders_interleaving(catalog, index, config):
    for mood, mode, recs, out in _lists(catalog, index, config):
        assert out.target.tolist() == recs.target.tolist(), (mood, mode)        # interleaving untouched
        for target in set(out.target):
            scores = out.score.to_numpy()[(out.target == target).to_numpy()]
            assert (np.diff(scores) <= 1e-12).all(), (mood, mode, target)       # non-increasing within target


def test_resort_only_reorders_the_selected_tracks(catalog, index, config, monkeypatch):
    moved = 0
    for mood in MOODS:
        for mode in MODES:
            np.random.seed(11)
            recs = recommend.recommend_music(catalog, mood, mode, top_n=30)
            state = np.random.get_state()
            out = ap.prioritize(catalog, recs, index, config)
            np.random.set_state(state)                                          # same swap draws, no re-sort
            with monkeypatch.context() as mp:
                mp.setattr(ap, "_order_within_targets", lambda targets, scores: np.arange(len(targets)))
                raw = ap.prioritize(catalog, recs, index, config)
            assert Counter(out.track_id) == Counter(raw.track_id), (mood, mode)  # same multiset
            for target in set(raw.target):                                      # same tracks per target
                assert set(out.track_id[out.target == target]) == set(raw.track_id[raw.target == target])
            assert set(out.columns) == set(raw.columns)
            moved += out.track_id.tolist() != raw.track_id.tolist()
    assert moved > 0                                                            # the re-sort really moves swaps up


def test_list_without_swaps_is_returned_in_its_order(catalog, index, config):
    np.random.seed(3)
    recs = recommend.recommend_music(catalog, "chill", "Match my mood", top_n=30)   # no eligible priority track
    out = ap.prioritize(catalog, recs, index, config)
    pd.testing.assert_frame_equal(out.drop(columns=["artist_category", "priority_artist"]), recs)


def test_zero_share_or_no_metadata_leaves_the_list_unchanged(catalog, index, config):
    np.random.seed(5)
    recs = recommend.recommend_music(catalog, "happy", "Match my mood", top_n=30)
    unchanged = ap.prioritize(catalog, recs, index, {**config, "max_share": 0.0})
    assert unchanged.track_id.tolist() == recs.track_id.tolist()
    bare = ap.prioritize(catalog, recs, ap.ArtistIndex(catalog, None, config), config)
    assert bare.track_id.tolist() == recs.track_id.tolist() and bare.artist_category.isna().all()
    assert ap.prioritize(catalog, recs, index, None).track_id.tolist() == recs.track_id.tolist()


# ---------------------------------------------------------------- API integration + fallback

def test_api_serves_the_recommender_list_when_the_layer_is_off_or_fails(monkeypatch, metadata):
    client = flask_app.app.test_client()
    catalog = flask_app.load_music_catalog()
    np.random.seed(9)
    expected = recommend.recommend_music(catalog, "happy", "Match my mood", top_n=20).track_id.tolist()

    def served():
        np.random.seed(9)
        return client.post("/api/recommend", json={"mood": "happy", "type": "music", "limit": 20}).get_json()["results"]

    monkeypatch.delenv("MOODCAST_ARTIST_PRIORITY", raising=False)     # default: off
    default = served()
    assert [r["track_id"] for r in default] == expected and all(r["artist_category"] is None for r in default)

    monkeypatch.setenv("MOODCAST_ARTIST_PRIORITY", "0")               # explicitly off
    assert [r["track_id"] for r in served()] == expected

    monkeypatch.setenv("MOODCAST_ARTIST_PRIORITY", "1")               # on: it really changes the list...
    assert [r["track_id"] for r in served()] != expected
    calls = []
    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("boom")
    monkeypatch.setattr(ap, "prioritize", boom)                       # ...and if it fails, the list is served as-is
    assert [r["track_id"] for r in served()] == expected and calls


def test_api_marks_only_verified_preferred_artists(config, enabled, metadata):
    client = flask_app.app.test_client()
    labels = {v["label"]: set(v["artists"]) for v in config["categories"].values()}
    marked = 0
    for mood in MOODS:
        for mode in ["match", "uplift"]:
            for r in client.post("/api/recommend", json={"mood": mood, "mode": mode, "type": "music", "limit": 30}).get_json()["results"]:
                if r["artist_category"]:
                    marked += 1
                    assert r["has_metadata"] and set(r["artists"]) & labels[r["artist_category"]], r["track_id"]
                    assert r["priority_artist"] in r["artists"] and r["priority_artist"] in labels[r["artist_category"]]
                    assert r["spotify_url"] == f"https://open.spotify.com/track/{r['track_id']}"
    assert marked > 0
