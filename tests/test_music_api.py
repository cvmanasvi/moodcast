"""The music recommendation API, exercised end to end through Flask on the real
278k catalog (built locally; every test here skips without it) -- the app's only music catalog since Phase 2H.4.
Metadata is supplied synthetically so the metadata-present / metadata-missing
cases are exact."""
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import flask_app
from src import music_metadata as mm
from src import recommend
from src.config import BASE_DIR, MOODS
from src.data_loader import load_music_catalog
from tests.local_data import requires_music_catalog

pytestmark = requires_music_catalog

MODES = {"match": "Match my mood", "uplift": "Cheer me up"}
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def catalog():
    return load_music_catalog()


@pytest.fixture(scope="module")
def by_id(catalog):
    return catalog.set_index("track_id")


@pytest.fixture
def client():
    return flask_app.app.test_client()


@pytest.fixture
def no_metadata(monkeypatch):
    """The out-of-the-box configuration: no local metadata built."""
    monkeypatch.setattr(flask_app, "load_music_metadata", lambda: None)


def _recommend(client, mood, mode="match", limit=30):
    res = client.post("/api/recommend", json={"mood": mood, "mode": mode, "type": "music", "limit": limit})
    assert res.status_code == 200, res.get_data(as_text=True)
    return res.get_json()["results"]


def _store_for(tmp_path, rows):
    """A real MetadataStore file for the given (track_id, title, artists, date) rows."""
    source = pd.DataFrame(rows, columns=["id", "name", "artists", "release_date"])
    meta = mm.build_metadata(pd.Series([r[0] for r in rows]), source, "TEST")
    path = tmp_path / "music_metadata.csv"
    mm.write_metadata(meta, str(path))
    return mm.load_metadata_store(str(path))


# ---- the single catalog ----

def test_api_serves_the_278k_catalog(catalog):
    assert len(catalog) == 277_938
    assert not hasattr(flask_app, "load_active_music_catalog")  # no catalog switch left
    assert not hasattr(flask_app, "MUSIC_CATALOG_FLAG")


def test_old_rollback_flag_is_ignored(monkeypatch, no_metadata, client):
    monkeypatch.setenv("USE_278K_MUSIC_CATALOG", "false")
    assert all(len(r["track_id"]) == 22 for r in _recommend(client, "happy", limit=5))


def test_recommend_music_rejects_a_catalog_without_track_ids(catalog):
    with pytest.raises(ValueError, match="278k catalog"):
        recommend.recommend_music(catalog.drop(columns=["track_id"]), "happy", "Match my mood", top_n=5)


# ---- all six moods, identity, integrity ----

@pytest.mark.parametrize("mode", ["match", "uplift"])
@pytest.mark.parametrize("mood", MOODS)
def test_every_mood_returns_its_targets(no_metadata, client, by_id, mood, mode):
    results = _recommend(client, mood, mode, limit=30)
    ids = [r["track_id"] for r in results]
    assert len(results) == 30
    assert all(isinstance(i, str) and len(i) == 22 for i in ids)
    assert len(set(ids)) == 30
    for r in results:
        ms = int(by_id.loc[r["track_id"], "duration_ms"])
        assert r["duration_ms"] == ms and r["duration"] == f"{ms // 60000}:{(ms // 1000) % 60:02d}"
        assert (r["title"], r["artist"], r["artists"], r["has_metadata"]) == ("Unknown title", "Unknown artist", [], False)
        assert r["spotify_url"] == f"https://open.spotify.com/track/{r['track_id']}"
        assert "popularity" not in r and r["image"] == "" and r["album"] is None
        assert r["mood"] and r["mood_confidence"] is not None
    # each track belongs to a target of the requested mood, in the exact shares
    targets = recommend._target_music_weighted(mood, MODES[mode])
    shares = dict(zip([t for t, _ in targets], recommend._allocate_shares([w for _, w in targets], 30)))
    counts = {}
    for tid in ids:
        row = by_id.loc[tid]
        matched = [t for t in shares if (row[f"derived_{t}"] if t in ("angry", "stressed") else row["mood"] == t)]
        assert matched, f"{tid} is not a {mood}/{mode} target"
        counts[matched[0]] = counts.get(matched[0], 0) + 1
    assert counts == shares


def test_api_preserves_recommendation_order_and_values(no_metadata, client, catalog):
    for mood in MOODS:
        for mode, rmode in MODES.items():
            np.random.seed(2024)
            results = _recommend(client, mood, mode, limit=30)
            np.random.seed(2024)
            recs = recommend.recommend_music(catalog, mood, rmode, top_n=30)
            assert [r["track_id"] for r in results] == recs["track_id"].tolist(), (mood, mode)
            assert [r["mood"] for r in results] == recs["mood"].tolist()
            assert [r["mood_confidence"] for r in results] == pytest.approx(recs["mood_confidence"].tolist())


def test_recommend_music_uses_pool_20_or_feature_cells_and_is_seed_deterministic(catalog):
    cells = recommend._get_feature_cells(catalog)
    for mood in MOODS:
        for rmode in MODES.values():
            targets = recommend._target_music_weighted(mood, rmode)
            np.random.seed(11)
            served = recommend.recommend_music(catalog, mood, rmode, top_n=30)["track_id"].tolist()
            np.random.seed(11)
            again = recommend.recommend_music(catalog, mood, rmode, top_n=30)["track_id"].tolist()
            np.random.seed(11)
            if recommend._uses_feature_cells(targets):   # single base-label target (Phase 2K)
                target = targets[0][0]
                expected = recommend._feature_cell_top(catalog, target, 30, cells[target])["track_id"].tolist()
            else:                                        # every other request: the pool-20 ranking
                expected = recommend._coverage_top(catalog, targets, 30, pool_multiplier=20)["track_id"].tolist()
            assert served == again == expected, (mood, rmode)


def test_pool_20_reaches_past_the_old_pool_3_candidates(catalog):
    # a single-target pool: every run samples from the same fixed pool of pool_multiplier * 30
    # candidates, so more than 90 distinct tracks is only possible above pool 3. Called directly
    # because sad / match itself is now served from feature cells.
    served = set()
    for seed in range(30):
        np.random.seed(seed)
        served |= set(recommend._coverage_top(catalog, [("sad", 1.0)], 30).index.tolist())
    assert 3 * 30 < len(served) <= 20 * 30


def test_music_recommendations_make_no_network_calls(monkeypatch, no_metadata, client):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for mood in MOODS:
        for mode in MODES:
            for r in _recommend(client, mood, mode, limit=30):
                assert r["spotify_url"] == f"https://open.spotify.com/track/{r['track_id']}"


def test_requested_limit_is_respected(no_metadata, client):
    for limit in [1, 7, 30, 100]:
        assert len(_recommend(client, "sad", "uplift", limit)) == limit


# ---- metadata, duration, popularity, artwork, links ----

def test_metadata_present_and_missing(monkeypatch, tmp_path, no_metadata, client, catalog):
    np.random.seed(7)
    first = _recommend(client, "sad", "match", 30)
    with_meta = [r["track_id"] for r in first[:3]]
    store = _store_for(tmp_path, [(tid, f"Title {i}", repr([f"Artist {i}", "Earth, Wind & Fire"]), "1999-01-01")
                                  for i, tid in enumerate(with_meta)])
    monkeypatch.setattr(flask_app, "load_music_metadata", lambda: store)
    np.random.seed(7)
    results = _recommend(client, "sad", "match", 30)
    assert [r["track_id"] for r in results] == [r["track_id"] for r in first]  # metadata never changes membership
    for i, r in enumerate(results[:3]):
        assert (r["title"], r["artist"], r["artists"], r["release_year"], r["has_metadata"]) == (
            f"Title {i}", f"Artist {i}, Earth, Wind & Fire", [f"Artist {i}", "Earth, Wind & Fire"], 1999, True)
    for r in results[3:]:
        assert (r["title"], r["artist"], r["artists"], r["release_year"], r["has_metadata"]) == (
            "Unknown title", "Unknown artist", [], None, False)


def test_real_duration_popularity_artwork_and_link(no_metadata, client, by_id):
    for r in _recommend(client, "energetic", "match", 30):
        ms = int(by_id.loc[r["track_id"], "duration_ms"])
        assert r["duration_ms"] == ms
        assert r["duration"] == f"{ms // 60000}:{(ms // 1000) % 60:02d}"
        assert "popularity" not in r
        assert r["image"] == "" and r["album"] is None and r["genre"] is None
        assert r["spotify_url"] == f"https://open.spotify.com/track/{r['track_id']}"


def test_missing_duration_is_null_not_a_fake_default(monkeypatch, no_metadata, client, catalog):
    sample = catalog.head(400).copy()
    sample["duration_ms"] = np.nan
    monkeypatch.setattr(flask_app, "load_music_catalog", lambda: sample)
    for r in _recommend(client, "happy", "match", 5):
        assert r["duration_ms"] is None and r["duration"] is None


# ---- the real music.js consumes a real 278k response ----

@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_music_page_renders_a_278k_response(monkeypatch, tmp_path, no_metadata, client):
    np.random.seed(3)
    ids = [r["track_id"] for r in _recommend(client, "happy", "match", 30)]
    # give two different tracks identical display fields (a re-release), leave the rest unknown
    store = _store_for(tmp_path, [(ids[0], "Same Song", repr(["Same Artist"]), "2001"),
                                  (ids[1], "Same Song", repr(["Same Artist"]), "2011")])
    monkeypatch.setattr(flask_app, "load_music_metadata", lambda: store)
    np.random.seed(3)
    body = client.post("/api/recommend", json={"mood": "happy", "type": "music", "limit": 30}).get_json()
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(body), encoding="utf-8")
    # Spotify oEmbed stand-in: a real-looking title + cover for one track, "not found" for another
    cover = "https://i.scdn.co/image/ab67616d0000b2730123456789abcdef01234567"
    details = {ids[2]: {"title": "Real oEmbed Title", "artwork": cover}, ids[3]: {"title": None, "artwork": None}}
    details_path = tmp_path / "details.json"
    details_path.write_text(json.dumps(details), encoding="utf-8")
    # the backend's real answer to a refresh that excludes everything just shown
    refresh = client.post("/api/recommend", json={"mood": "happy", "type": "music", "limit": 30, "exclude": ids}).get_json()
    refresh_path = tmp_path / "refresh.json"
    refresh_path.write_text(json.dumps(refresh), encoding="utf-8")

    out = subprocess.run([NODE, os.path.join("tests", "js", "music_page_harness.js"), BASE_DIR,
                          str(response_path), str(details_path), str(refresh_path)],
                         cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    page = json.loads(out.stdout)
    assert page["rows"] == 30
    assert page["rowTrackIds"] == ids                           # every card keeps its track_id
    assert page["badText"] == 0                                 # no undefined / null / [object Object]
    assert page["durationsShown"] == [r["duration"] for r in body["results"]]
    assert page["favorites"] == [ids[-1]]                       # favorite stored by track_id
    # each row opens its own exact track on Spotify (twins with identical titles included)
    assert page["spotifyLinks"] == [f"https://open.spotify.com/track/{i}" for i in ids]
    assert not any(page["linkUnavailable"])
    # local metadata shows at once; the others wait for oEmbed instead of showing a fake title
    assert page["headingsBeforeDetails"] == ["Same Song", "Same Song"] + ["Loading title…"] * 28
    # after oEmbed: a real title where Spotify has one (no artist label invented), honest fallback elsewhere
    assert page["headingsAfterDetails"] == ["Same Song", "Same Song", "Real oEmbed Title"] + ["Track details unavailable"] * 27
    assert page["artistLines"][0].startswith("Same Artist") and page["artistLines"][2] is None
    assert page["covers"] == [False, False, True] + [False] * 27
    assert page["placeholderText"] == 0
    # MoodCast doesn't play audio: no in-page player API, rows aren't "play" buttons
    assert page["playbackApi"] == [] and page["rowsAreButtons"] == 0
