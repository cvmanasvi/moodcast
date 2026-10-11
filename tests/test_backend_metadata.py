"""Backend metadata integration (Phase 2E): loading the optional local metadata,
enriching 278k recommendation results by exact track_id, placeholder fallbacks,
local song/artist search, and Spotify links built from the ID alone.

Everything here runs on synthetic catalogs/metadata, so nothing depends on the
gitignored local files."""
import json

import numpy as np
import pandas as pd
import pytest

import flask_app
from src import data_loader
from src import music_metadata as mm
from src.recommend import recommend_music

MOODS4 = ["sad", "happy", "energetic", "calm"]


def _tid(i):
    return f"T{i:021d}"


def _catalog(n=60, seed=0):
    """Synthetic 278k-style catalog (true labels cycle sad/happy/energetic/calm)."""
    rng = np.random.RandomState(seed)
    proba = rng.dirichlet(np.ones(4), size=n)
    df = pd.DataFrame({
        "track_id": [_tid(i) for i in range(n)],
        "label": [i % 4 for i in range(n)],
        "mood": [MOODS4[i % 4] for i in range(n)],
        "duration_ms": 180000 + np.arange(n) * 1000,
        **{f: rng.rand(n) for f in ["valence", "energy", "danceability", "acousticness"]},
        "tempo": 120.0,
    })
    for i, m in enumerate(MOODS4):
        df[f"proba_{m}"] = np.round(proba[:, i], 4)
    df["mood_confidence"] = df["proba_sad"]
    return df


def _source(rows):
    return pd.DataFrame(rows, columns=["id", "name", "artists", "release_date"])


def _store(tmp_path, rows, catalog):
    """Build + write + load metadata through the real pipeline functions."""
    meta = mm.build_metadata(catalog["track_id"], _source(rows), "TEST")
    path = tmp_path / "music_metadata.csv"
    mm.write_metadata(meta, str(path))
    return mm.load_metadata_store(str(path))


@pytest.fixture
def catalog():
    return _catalog()


@pytest.fixture
def store(tmp_path, catalog):
    rows = [(_tid(i), f"Song {i}", repr([f"Artist {i}"]), "2001") for i in range(0, 60, 2)]  # even ids only
    rows += [
        # same title + artist as each other, different track_ids (e.g. a re-release)
        (_tid(1), "Twin Song", repr(["Twin Artist"]), "1990"),
        (_tid(3), "Twin Song", repr(["Twin Artist"]), "2010"),
        # comma inside an artist name
        (_tid(5), "SMUCKERS", repr(["Tyler, The Creator", "Lil Wayne", "Kanye West"]), "2015"),
        (_tid(7), "Fantasy", repr(["Earth, Wind & Fire"]), "1978-01-01"),
    ]
    return _store(tmp_path, rows, catalog)


@pytest.fixture
def client(monkeypatch, catalog, store):
    monkeypatch.setattr(flask_app, "load_music_catalog", lambda: catalog)
    monkeypatch.setattr(flask_app, "load_music_metadata", lambda: store)
    return flask_app.app.test_client()


def _recommend(client, mood="sad", mode="uplift", limit=20):
    res = client.post("/api/recommend", json={"mood": mood, "mode": mode, "type": "music", "limit": limit})
    assert res.status_code == 200
    return res.get_json()["results"]


# ---- 1. loading ----

def test_store_loads_from_file(store):
    assert len(store) == 34 and _tid(0) in store and _tid(9) not in store


def test_missing_file_means_no_store(tmp_path):
    assert mm.load_metadata_store(str(tmp_path / "absent.csv")) is None


def test_backend_loader_reads_once_and_caches(monkeypatch, tmp_path, catalog):
    path = tmp_path / "music_metadata.csv"
    mm.write_metadata(mm.build_metadata(catalog["track_id"], _source([(_tid(0), "S", "['A']", "2000")]), "T"),
                      str(path))
    calls = []
    real = data_loader.load_metadata_store
    monkeypatch.setattr(data_loader, "load_metadata_store", lambda p: calls.append(p) or real(p))
    monkeypatch.setattr(data_loader, "MUSIC_METADATA_PATH", str(path))
    monkeypatch.setattr(data_loader, "_music_metadata_loaded", False)
    monkeypatch.setattr(data_loader, "_music_metadata", None)
    first = data_loader.load_music_metadata()
    assert data_loader.load_music_metadata() is first and len(first) == 1
    assert calls == [str(path)]


@pytest.mark.parametrize("content", [None, "not,a,metadata\nfile,at,all\n"])
def test_backend_loader_degrades_to_none(monkeypatch, tmp_path, content):
    path = tmp_path / "music_metadata.csv"
    if content is not None:
        path.write_text(content)
    monkeypatch.setattr(data_loader, "MUSIC_METADATA_PATH", str(path))
    monkeypatch.setattr(data_loader, "_music_metadata_loaded", False)
    monkeypatch.setattr(data_loader, "_music_metadata", None)
    assert data_loader.load_music_metadata() is None


# ---- 2. lookup by track_id ----

def test_lookup_is_exact_by_track_id(store):
    assert store.display(_tid(2)) == {"title": "Song 2", "artist": "Artist 2", "artists": ["Artist 2"],
                                      "release_year": 2001, "has_metadata": True}
    assert store.display(_tid(2).lower())["has_metadata"] is False  # no case-folding / fuzzy lookup
    assert store.display(_tid(9)) == mm.missing_display()


# ---- 3. + 11. enrichment keeps order, ids and ranking values ----

def test_enrichment_does_not_change_recommendations(client, catalog):
    np.random.seed(11)
    results = _recommend(client, "sad", "uplift", 20)
    np.random.seed(11)
    recs = recommend_music(catalog, "sad", "Cheer me up", top_n=20)
    assert [r["track_id"] for r in results] == recs["track_id"].tolist()           # same tracks, same order
    assert [r["mood"] for r in results] == recs["mood"].tolist()
    assert [r["mood_confidence"] for r in results] == pytest.approx(recs["mood_confidence"].tolist())
    assert len(results) == 20


def test_enriched_fields_come_from_the_tracks_own_metadata(client):
    for r in _recommend(client, "sad", "uplift", 20):
        idx = int(r["track_id"][1:])
        if r["has_metadata"] and idx % 2 == 0:
            assert (r["title"], r["artist"], r["artists"], r["release_year"]) == \
                   (f"Song {idx}", f"Artist {idx}", [f"Artist {idx}"], 2001)
        assert r["duration_ms"] == 180000 + idx * 1000          # still from the catalog
        assert r["spotify_url"] == f"https://open.spotify.com/track/{r['track_id']}"


def test_metadata_is_never_borrowed_from_another_track(client, store):
    """Every returned title/artist is exactly what that track_id has in the store."""
    for mood in ["sad", "happy", "energetic", "chill"]:
        for mode in ["match", "uplift"]:
            for r in _recommend(client, mood, mode, 30):
                expected = store.display(r["track_id"])
                assert (r["title"], r["artist"], r["artists"]) == \
                       (expected["title"], expected["artist"], expected["artists"])


# ---- 4. missing metadata ----

def test_tracks_without_metadata_keep_their_place(client, catalog):
    np.random.seed(3)
    results = _recommend(client, "happy", "match", 15)  # happy labels are odd ids: mostly no metadata
    np.random.seed(3)
    recs = recommend_music(catalog, "happy", "Match my mood", top_n=15)
    assert [r["track_id"] for r in results] == recs["track_id"].tolist()
    missing = [r for r in results if not r["has_metadata"]]
    assert missing, "fixture should include tracks without metadata"
    for r in missing:
        assert r["track_id"] and r["title"] == "Unknown title" and r["artist"] == "Unknown artist"
        assert r["artists"] == [] and r["release_year"] is None and r["year"] is None
        assert r["spotify_url"] == f"https://open.spotify.com/track/{r['track_id']}"


def test_recommendations_work_with_no_metadata_at_all(monkeypatch, catalog):
    monkeypatch.setattr(flask_app, "load_music_catalog", lambda: catalog)
    monkeypatch.setattr(flask_app, "load_music_metadata", lambda: None)
    results = _recommend(flask_app.app.test_client(), "sad", "match", 10)
    assert len(results) == 10
    assert all(r["title"] == "Unknown title" and r["has_metadata"] is False for r in results)


# ---- 5. duplicate display identities stay separate ----

def test_same_title_and_artist_remain_separate_tracks(client, store):
    twins = [store.display(_tid(1)), store.display(_tid(3))]
    assert twins[0]["title"] == twins[1]["title"] and twins[0]["artist"] == twins[1]["artist"]
    assert [t["release_year"] for t in twins] == [1990, 2010]
    hits = [h for h in store.search_tracks("twin song") if h["title"] == "Twin Song"]
    assert sorted(h["track_id"] for h in hits) == [_tid(1), _tid(3)]
    results = _recommend(client, "happy", "match", 15)  # ids 1 and 3 are happy-labelled
    ids = [r["track_id"] for r in results]
    assert len(ids) == len(set(ids))
    for r in results:
        if r["track_id"] in (_tid(1), _tid(3)):
            assert r["release_year"] == {_tid(1): 1990, _tid(3): 2010}[r["track_id"]]


# ---- 6. comma-containing artist names ----

def test_comma_artist_names_are_not_split(store):
    d = store.display(_tid(5))
    assert d["artists"] == ["Tyler, The Creator", "Lil Wayne", "Kanye West"]
    assert d["artist"] == "Tyler, The Creator, Lil Wayne, Kanye West"
    assert store.display(_tid(7))["artists"] == ["Earth, Wind & Fire"]


def test_api_returns_decoded_artist_list(client):
    # id 5 is happy-labelled and id 7 calm-labelled, so both can be recommended
    results = {r["track_id"]: r for mood in ["happy", "chill"] for r in _recommend(client, mood, "match", 15)}
    assert all(isinstance(r["artists"], list) for r in results.values())
    assert results[_tid(5)]["artists"] == ["Tyler, The Creator", "Lil Wayne", "Kanye West"]
    assert results[_tid(7)]["artists"] == ["Earth, Wind & Fire"]
    assert results[_tid(7)]["artist"] == "Earth, Wind & Fire"
    res = client.get("/api/search?q=smuckers").get_json()
    assert res["songs"][0]["artists"] == ["Tyler, The Creator", "Lil Wayne", "Kanye West"]


# ---- 7. + 8. structured, case-insensitive search ----

def test_artist_search_returns_whole_names(client):
    names = [a["name"] for a in client.get("/api/search?q=the creator").get_json()["artists"]]
    assert names == ["Tyler, The Creator"]                       # not "The Creator"
    names = [a["name"] for a in client.get("/api/search?q=earth, wind").get_json()["artists"]]
    assert names == ["Earth, Wind & Fire"]                       # not "Earth" / "Wind & Fire"


def test_song_search_matches_title_and_artist(client):
    by_artist = client.get("/api/search?q=earth, wind").get_json()["songs"]
    assert [s["track_id"] for s in by_artist] == [_tid(7)]
    by_title = client.get("/api/search?q=fantasy").get_json()["songs"]
    assert [s["track_id"] for s in by_title] == [_tid(7)]
    assert by_title[0]["spotify_url"] == f"https://open.spotify.com/track/{_tid(7)}"


@pytest.mark.parametrize("q", ["TWIN SONG", "twin song", "Twin Song", "tWiN sOnG"])
def test_search_is_case_insensitive(client, q):
    songs = client.get(f"/api/search?q={q}").get_json()["songs"]
    assert sorted(s["track_id"] for s in songs) == [_tid(1), _tid(3)]


def test_search_order_is_deterministic(store):
    first = store.search_tracks("song", limit=50)
    assert first == store.search_tracks("song", limit=50)
    titles = [h["title"].lower() for h in first]
    prefix = [t for t in titles if t.startswith("song")]
    assert titles[:len(prefix)] == sorted(prefix)                 # title-prefix hits first, alphabetical


def test_search_limits_results(client):
    body = client.get("/api/search?q=song").get_json()
    assert len(body["songs"]) == 6 and len(body["artists"]) <= 6


def test_search_only_returns_tracks_with_metadata(client, store):
    for s in client.get("/api/search?q=song").get_json()["songs"]:
        assert s["track_id"] in store


# ---- 9. search without metadata ----

def test_search_without_metadata_is_empty_not_an_error(monkeypatch):
    monkeypatch.setattr(flask_app, "load_music_metadata", lambda: None)
    res = flask_app.app.test_client().get("/api/search?q=love")
    assert res.status_code == 200
    body = res.get_json()
    assert body["songs"] == [] and body["artists"] == []
    assert set(body) == {"movies", "songs", "artists", "genres"}


# ---- 10. Spotify URL ----

def test_spotify_url_is_built_from_the_id_only():
    assert mm.spotify_track_url("3v6sBj3swihU8pXQQHhDZo") == "https://open.spotify.com/track/3v6sBj3swihU8pXQQHhDZo"
    assert mm.spotify_track_url("3v6sBj3swihU8pXQQHhDZo") == mm.spotify_track_url("3v6sBj3swihU8pXQQHhDZo")
    for bad in [None, "", "short", "3v6sBj3swihU8pXQQHhDZo/../x", 42]:
        assert mm.spotify_track_url(bad) is None
