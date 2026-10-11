"""Tests for the Popular Artists mode (src/popular_artists.py, POST /api/popular).

Covers whole-entry artist matching (never naive comma splitting, never a
partial name), loading the prepared file, mood eligibility for all six moods
in both modes, per-artist limits, internal category balancing, refresh
exclusions, short and empty results, determinism, the API (no category ever
exposed, exact Spotify links, honest "unavailable" when the data isn't
prepared) and separation from /api/recommend. Synthetic fixtures use made-up
track ids and artist names and are labelled so. Real-data tests run only when
the local dataset has been prepared (python prepare_popular_artists.py).
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import flask_app
from src import popular_artists as pa
from src import recommend
from tests.local_data import requires_music_catalog

MOODS = ["happy", "sad", "energetic", "chill", "angry", "stressed"]
MODES = {"match": "Match my mood", "uplift": "Cheer me up"}
CATEGORY_WORDS = re.compile(r"international|k-?pop|indian|category", re.IGNORECASE)


# ---------------------------------------------------------------- matching

def test_matching_uses_whole_entries_and_keeps_comma_names():
    names = ["Pritam", "Tyler, The Creator", "BTS", "Arijit Singh"]
    assert pa.match_artists("Pritam Chakraborty", names) == []                      # partial name: no match
    assert pa.match_artists("Pritam, Arijit Singh", names) == ["Pritam", "Arijit Singh"]
    assert pa.match_artists("Tyler, The Creator, Kali Uchis", names) == ["Tyler, The Creator"]
    assert pa.match_artists("Tyler", names) == [] and pa.match_artists("The Creator", names) == []
    assert pa.match_artists("bts", names) == []                                      # case-sensitive
    assert pa.match_artists("Arijit Singh,Pritam", names) == ["Arijit Singh", "Pritam"]   # no space after comma
    assert pa.match_artists("BTS, BTS", names) == ["BTS"]


def test_shipped_config_is_curated_and_unambiguous():
    cfg = pa.load_config()
    names = [n for v in cfg["categories"].values() for n in v]
    assert set(cfg["categories"]) == {"international", "kpop", "indian"}
    assert len(names) == len(set(names)) == 60
    for ambiguous in ["V", "Jin", "RM", "King", "Atif Aslam"]:
        assert ambiguous not in names


# ---------------------------------------------------------------- synthetic prepared data

def _track_id(i):
    return f"{i:022d}"


def _frame(rows):
    """A prepared frame (as load_prepared returns it) from row dicts -- SYNTHETIC test data."""
    base = {"year": 2020, "popularity": 50, "duration_ms": 200000, "danceability": 0.5, "energy": 0.5,
            "loudness": -6.0, "speechiness": 0.05, "acousticness": 0.2, "instrumentalness": 0.0,
            "liveness": 0.1, "valence": 0.5, "tempo": 120.0, "proba_sad": 0.1, "proba_happy": 0.1,
            "proba_energetic": 0.1, "proba_calm": 0.1, "derived_angry": False, "derived_angry_score": 0.0,
            "derived_stressed": False, "derived_stressed_score": 0.0}
    out = []
    for i, r in enumerate(rows):
        tid = _track_id(i)
        row = {**base, "track_id": tid, "track_name": f"Synthetic Song {i}", "track_url": pa.SPOTIFY_TRACK_URL + tid, **r}
        row.setdefault("artist_name", ", ".join(row["matched_artists"]))
        out.append(row)
    return pd.DataFrame(out)[pa.PREPARED_COLUMNS]


def _catalogue(per_artist=6, artists_per_cat=5, **probas):
    """Three categories x `artists_per_cat` synthetic artists x `per_artist` happy tracks."""
    rows = []
    for cat in ["international", "kpop", "indian"]:
        for a in range(artists_per_cat):
            for k in range(per_artist):
                rows.append({"matched_artists": [f"Synthetic {cat} {a}"], "category": cat,
                             "popularity": 20 + 10 * k, "proba_happy": 0.9, **probas})
    return _frame(rows)


def _ids(rows):
    return rows["track_id"].tolist()


def test_lists_are_capped_per_artist_and_balanced_across_categories():
    df = _catalogue()
    np.random.seed(0)
    rows, info = pa.recommend_popular(df, "happy", "Match my mood")
    assert len(rows) == 30 and not info["limited"] and info["targets"] == ["happy"]
    assert max(Counter(a for m in rows.matched_artists for a in m).values()) <= pa.MAX_PER_ARTIST
    assert Counter(rows.category) == {"international": 10, "kpop": 10, "indian": 10}
    assert rows.track_id.is_unique and (rows.proba_happy >= pa.CONFIDENCE_MIN).all()


def test_a_collaboration_counts_toward_every_matched_artist():
    rows = [{"matched_artists": ["Synthetic A", "Synthetic B"], "category": "kpop", "proba_happy": 0.9, "popularity": 90}
            for _ in range(3)]
    rows += [{"matched_artists": ["Synthetic B"], "category": "kpop", "proba_happy": 0.9} for _ in range(3)]
    out, _ = pa.recommend_popular(_frame(rows), "happy", "Match my mood")
    counts = Counter(a for m in out.matched_artists for a in m)
    assert counts["Synthetic A"] <= 2 and counts["Synthetic B"] <= 2


def test_only_confident_tracks_are_eligible_and_short_lists_say_so():
    rows = [{"matched_artists": [f"Synthetic {i}"], "category": "indian", "proba_happy": 0.9} for i in range(4)]
    rows += [{"matched_artists": [f"Synthetic low {i}"], "category": "indian", "proba_happy": 0.59} for i in range(20)]
    out, info = pa.recommend_popular(_frame(rows), "happy", "Match my mood")
    assert len(out) == 4 and info == {"targets": ["happy"], "eligible": 4, "returned": 4, "limited": True,
                                      "limited_by": "pool"}
    empty, info = pa.recommend_popular(_frame(rows), "sad", "Match my mood")
    assert len(empty) == 0 and info["limited"] and info["eligible"] == 0


def test_every_mood_and_mode_uses_the_apps_target_mapping():
    df = _catalogue(proba_sad=0.9, proba_energetic=0.9, proba_calm=0.9,
                    derived_angry=True, derived_angry_score=0.8, derived_stressed=True, derived_stressed_score=0.7)
    for mood in MOODS:
        for mode in MODES.values():
            np.random.seed(1)
            rows, info = pa.recommend_popular(df, mood, mode)
            expected = [t for t, _ in recommend._target_music_weighted(mood, mode)]
            assert info["targets"] == expected and set(rows.target) <= set(expected), (mood, mode)
            for target in set(rows.target):
                member, _ = pa.target_candidates(df, target)
                assert member[df.index.get_indexer(rows.index[rows.target == target])].all(), (mood, mode, target)


def test_derived_moods_use_the_prepared_rule_flags_and_calm_keeps_its_speech_guard():
    rows = [{"matched_artists": ["Synthetic angry"], "category": "kpop", "derived_angry": True, "derived_angry_score": 0.9},
            {"matched_artists": ["Synthetic not angry"], "category": "kpop", "derived_angry": False, "derived_angry_score": 0.99},
            {"matched_artists": ["Synthetic calm"], "category": "indian", "proba_calm": 0.9, "speechiness": 0.1},
            {"matched_artists": ["Synthetic speech"], "category": "indian", "proba_calm": 0.9, "speechiness": 0.8}]
    df = _frame(rows)
    assert pa.target_candidates(df, "angry")[0].tolist() == [True, False, False, False]
    assert pa.target_candidates(df, "calm")[0].tolist() == [False, False, True, False]
    with pytest.raises(ValueError):
        pa.target_candidates(df, "romantic")


def test_refresh_excludes_recent_tracks_and_reports_exhaustion():
    df = _catalogue(per_artist=2)                       # 30 eligible tracks in all
    np.random.seed(2)
    first, _ = pa.recommend_popular(df, "happy", "Match my mood")
    again, info = pa.recommend_popular(df, "happy", "Match my mood", exclude=_ids(first)[:20])
    assert not set(_ids(again)) & set(_ids(first)[:20])
    assert len(again) == 10 and info["limited"]


def test_a_list_cut_short_by_the_per_artist_limit_says_so():
    rows = [{"matched_artists": ["Synthetic Prolific"], "category": "indian", "proba_calm": 0.9} for _ in range(10)]
    rows += [{"matched_artists": [f"Synthetic {i}"], "category": "kpop", "proba_calm": 0.9} for i in range(3)]
    out, info = pa.recommend_popular(_frame(rows), "chill", "Match my mood")
    assert len(out) == 5 and info["eligible"] == 13 and info["limited_by"] == "artist_cap"
    full, info = pa.recommend_popular(_catalogue(), "happy", "Match my mood")
    assert len(full) == 30 and info["limited"] is False and info["limited_by"] is None


def test_repeated_refreshes_never_repeat_until_the_pool_runs_out():
    df = _catalogue(per_artist=3, artists_per_cat=4)          # 36 eligible; 2 per artist -> at most 24 per list
    np.random.seed(8)
    seen, sizes = [], []
    for _ in range(4):
        rows, info = pa.recommend_popular(df, "happy", "Match my mood", exclude=seen)
        assert not set(_ids(rows)) & set(seen)
        sizes.append((len(rows), info["limited_by"]))
        seen += _ids(rows)
    assert sizes[0] == (24, "artist_cap")                    # 36 eligible, but only 2 per artist fit in one list
    assert sizes[1] == (12, "pool")                          # the 12 never shown
    assert sizes[2] == (0, "pool") and sizes[3] == (0, "pool")
    assert len(set(seen)) == 36


def test_refresh_keeps_the_cheer_me_up_targets():
    df = _catalogue(proba_calm=0.9)                          # happy and calm both eligible
    np.random.seed(9)
    first, info = pa.recommend_popular(df, "sad", "Cheer me up")
    assert info["targets"] == ["happy", "calm"]
    again, info = pa.recommend_popular(df, "sad", "Cheer me up", exclude=_ids(first))
    assert info["targets"] == ["happy", "calm"] and set(again.target) <= {"happy", "calm"}
    assert not set(_ids(again)) & set(_ids(first))


def test_lists_are_deterministic_for_a_fixed_seed_and_never_exceed_the_cap():
    df = _catalogue(per_artist=10)
    np.random.seed(5)
    a, _ = pa.recommend_popular(df, "happy", "Match my mood")
    np.random.seed(5)
    b, _ = pa.recommend_popular(df, "happy", "Match my mood")
    assert _ids(a) == _ids(b)
    np.random.seed(6)
    c, _ = pa.recommend_popular(df, "happy", "Match my mood")
    assert _ids(c) != _ids(a)
    assert len(pa.recommend_popular(df, "happy", "Match my mood", limit=100)[0]) == pa.MAX_RESULTS


# ---------------------------------------------------------------- loading the prepared file

def _write_prepared(path, df):
    df.assign(matched_artists=df.matched_artists.map(json.dumps)).to_csv(path, index=False)


def test_load_prepared_parses_lists_and_rejects_missing_or_broken_files(tmp_path):
    assert pa.load_prepared(str(tmp_path / "missing.csv")) is None
    df = _catalogue(per_artist=1)
    df.loc[0, "artist_name"] = "Synthetic, With Comma"
    good = tmp_path / "good.csv"
    _write_prepared(good, df)
    loaded = pa.load_prepared(str(good))
    assert loaded.matched_artists.iloc[0] == df.matched_artists.iloc[0] and loaded.artist_name.iloc[0] == "Synthetic, With Comma"
    assert loaded.derived_angry.dtype == bool and loaded.track_id.tolist() == df.track_id.tolist()
    bad_url = tmp_path / "bad_url.csv"
    _write_prepared(bad_url, df.assign(track_url="https://example.com/x"))
    assert pa.load_prepared(str(bad_url)) is None
    no_col = tmp_path / "no_col.csv"
    _write_prepared(no_col, df.drop(columns=["proba_happy"]))
    assert pa.load_prepared(str(no_col)) is None


# ---------------------------------------------------------------- API

@pytest.fixture
def client():
    return flask_app.app.test_client()


@pytest.fixture
def synthetic(monkeypatch):
    df = _catalogue(proba_sad=0.9, derived_angry=True, derived_angry_score=0.8)
    df.loc[0, "artist_name"] = "Synthetic international 0, The Comma Band"
    monkeypatch.setattr(pa, "load_prepared", lambda path=None: df)
    return df


def test_api_returns_cards_with_exact_links_and_no_categories(client, synthetic):
    np.random.seed(3)
    body = client.post("/api/popular", json={"mood": "happy", "mode": "match"}).get_json()
    assert body["available"] is True and body["limit"] == 30 and len(body["results"]) == 30
    assert body["eligible"] == 90 and body["limited"] is False
    for r in body["results"]:
        assert r["spotify_url"] == f"https://open.spotify.com/track/{r['track_id']}"
        assert r["has_metadata"] is True and r["image"] == "" and r["mood"] == "happy"
        assert r["title"] == synthetic.set_index("track_id").loc[r["track_id"], "track_name"]
        assert r["artist"] == synthetic.set_index("track_id").loc[r["track_id"], "artist_name"]   # verbatim
        assert 0.6 <= r["mood_confidence"] <= 1
        assert not any(CATEGORY_WORDS.search(k) for k in r)
        assert not CATEGORY_WORDS.search(json.dumps({k: v for k, v in r.items() if k not in ("title", "artist")}))


def test_api_rule_based_moods_have_no_model_confidence(client, synthetic):
    body = client.post("/api/popular", json={"mood": "angry", "mode": "match"}).get_json()
    assert body["results"] and all(r["mood_confidence"] is None for r in body["results"])


def test_api_refresh_excludes_and_reports(client, synthetic):
    np.random.seed(4)
    first = [r["track_id"] for r in client.post("/api/popular", json={"mood": "happy"}).get_json()["results"]]
    body = client.post("/api/popular", json={"mood": "happy", "exclude": first}).get_json()
    assert not set(first) & {r["track_id"] for r in body["results"]}
    assert body["refresh"] == {"excluded": 30, "returned": 30, "exhausted": False} and body["limited_by"] is None


def test_api_refresh_respects_the_selected_mode(client, synthetic):
    np.random.seed(12)
    first = [r["track_id"] for r in client.post("/api/popular", json={"mood": "angry", "mode": "uplift"}).get_json()["results"]]
    body = client.post("/api/popular", json={"mood": "angry", "mode": "uplift", "exclude": first}).get_json()
    assert body["mode"] == "uplift" and not set(first) & {r["track_id"] for r in body["results"]}
    targets = {t for t, _ in recommend._target_music_weighted("angry", "Cheer me up")}
    for r in body["results"]:                                # each track fits one of the Cheer me up targets
        assert any(pa.target_candidates(synthetic, t)[0][synthetic.index[synthetic.track_id == r["track_id"]][0]]
                   for t in targets), r["track_id"]


def test_api_validates_input_and_caps_the_limit(client, synthetic):
    assert client.post("/api/popular", json={"mood": "romantic"}).status_code == 400
    assert client.post("/api/popular", json={"mood": "happy", "mode": "party"}).status_code == 400
    assert client.post("/api/popular", json={"mood": "happy", "exclude": ["not-an-id"]}).status_code == 400
    assert client.post("/api/popular", data="x").status_code == 400
    body = client.post("/api/popular", json={"mood": "happy", "limit": 100}).get_json()
    assert body["limit"] == 30 and len(body["results"]) == 30


def test_api_reports_unavailable_without_prepared_data(client, monkeypatch):
    monkeypatch.setattr(pa, "load_prepared", lambda path=None: None)
    r = client.post("/api/popular", json={"mood": "chill", "mode": "uplift"})
    body = r.get_json()
    assert r.status_code == 200 and body["available"] is False and body["results"] == []
    assert "prepare_popular_artists.py" in body["message"]


@requires_music_catalog
def test_normal_recommendations_never_touch_the_popular_pipeline(client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("popular pipeline used by /api/recommend")
    monkeypatch.setattr(pa, "load_prepared", boom)
    monkeypatch.setattr(pa, "recommend_popular", boom)
    r = client.post("/api/recommend", json={"mood": "happy", "type": "music", "limit": 30})
    assert r.status_code == 200 and len(r.get_json()["results"]) == 30


# ---------------------------------------------------------------- real local data (skipped when not prepared)

requires_data = pytest.mark.skipif(not os.path.exists(pa.PREPARED_PATH),
                                   reason="run `python prepare_popular_artists.py` to prepare the local dataset")


@pytest.fixture(scope="module")
def real():
    df = pa.load_prepared()
    assert df is not None
    with open(pa.MANIFEST_PATH, encoding="utf-8") as f:
        return df, json.load(f)


@requires_data
def test_manifest_pins_the_source_and_carries_no_track_ids(real):
    import prepare_popular_artists as prep
    from src.music_pipeline import file_sha256
    df, manifest = real
    assert manifest["raw_sha256"] == prep.EXPECTED_SHA256 and manifest["source"]["version"] == 2
    assert manifest["prepared_sha256"] == file_sha256(pa.PREPARED_PATH)
    assert manifest["contains_track_ids"] is False and manifest["local_only"] is True
    with open(pa.MANIFEST_PATH, encoding="utf-8") as f:
        text = f.read()
    assert not set(re.findall(r"\b[0-9A-Za-z]{22}\b", text)) & set(df.track_id)
    assert "artwork_url" not in df.columns


@requires_data
def test_every_curated_artist_and_every_mood_is_covered(real):
    df, manifest = real
    assert manifest["curated_artists_missing"] == []
    found = set(df.matched_artists.explode())
    assert found == {n for v in pa.load_config()["categories"].values() for n in v}
    for mood, c in manifest["coverage_match_my_mood"].items():
        assert c["tracks"] > 0 and len(c["by_category"]) == 3, mood


@requires_data
def test_real_data_is_clean(real):
    df, _ = real
    assert df.track_id.is_unique and (df.track_url == pa.SPOTIFY_TRACK_URL + df.track_id).all()
    feats = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
    assert ((df[feats] >= 0) & (df[feats] <= 1)).all().all() and (df.duration_ms >= 60000).all()
    assert (df.speechiness <= 0.66).all() and (df.tempo > 0).all()
    names = [n for v in pa.load_config()["categories"].values() for n in v]
    for s, m in zip(df.artist_name, df.matched_artists):
        assert m == pa.match_artists(s, names)


@requires_data
def test_real_lists_for_every_mood_and_mode(real):
    df, _ = real
    for mood in MOODS:
        for mode in MODES.values():
            np.random.seed(11)
            rows, info = pa.recommend_popular(df, mood, mode)
            assert 0 < len(rows) <= 30 and rows.track_id.is_unique, (mood, mode)
            assert max(Counter(a for m in rows.matched_artists for a in m).values()) <= 2, (mood, mode)
            assert rows.category.nunique() == 3, (mood, mode)
            for target in set(rows.target):
                member, _ = pa.target_candidates(df, target)
                assert member[df.index.get_indexer(rows.index[rows.target == target])].all(), (mood, mode, target)
            again, info = pa.recommend_popular(df, mood, mode, exclude=_ids(rows))
            assert not set(_ids(again)) & set(_ids(rows)), (mood, mode)
            if info["limited"]:                              # the reason given matches what ran out
                assert info["limited_by"] == ("artist_cap" if info["eligible"] > len(again) else "pool"), (mood, mode)


@requires_data
def test_real_chill_refresh_is_short_because_of_the_artist_limit_not_the_pool(real):
    df, _ = real
    np.random.seed(0)
    first, _ = pa.recommend_popular(df, "chill", "Match my mood")
    again, info = pa.recommend_popular(df, "chill", "Match my mood", exclude=_ids(first))
    member, _ = pa.target_candidates(df, "calm")
    unseen = df[member & ~df.track_id.isin(first.track_id)]
    bound = sum(min(2, n) for n in Counter(a for m in unseen.matched_artists for a in m).values())
    assert len(again) <= bound < info["eligible"] and info["limited_by"] == "artist_cap"
