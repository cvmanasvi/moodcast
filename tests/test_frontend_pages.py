"""Frontend + integration regression tests for the Music and Movies pages.

The page scripts run against real /api/recommend responses through
tests/js/movies_page_harness.js and tests/js/music_page_harness.js (fake DOM,
real ui-utils.js / movies.js / music.js); these tests assert what they
rendered and requested. Also covered: the new display-only backend pieces
(TMDB movie details, Spotify oEmbed titles, refresh with `exclude`), all with
outbound HTTP mocked -- no test calls TMDB or Spotify. Fixture provider names,
logos and titles below are SYNTHETIC test data, not claims about real films.
"""
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest
import requests

import flask_app
from src import movie_details, spotify_oembed
from src.config import BASE_DIR
from tests.local_data import requires_movie_data, requires_music_catalog

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def _read(*parts):
    with open(os.path.join(BASE_DIR, *parts), encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def client():
    return flask_app.app.test_client()


def _sized(url):
    """The display-size poster URL the page requests (same image, see posterDisplayUrl)."""
    m = re.match(r"^(https://m\.media-amazon\.com/images/M/[^?#]+?)\.(jpe?g|png)$", url, re.IGNORECASE)
    return f"{m.group(1)}._V1_SX400.{m.group(2)}" if m and "._V1_" not in m.group(1) else url


def _run(harness, *args):
    out = subprocess.run([NODE, os.path.join("tests", "js", harness), BASE_DIR, *map(str, args)],
                         cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    """Answers requests.Session().get by URL; records every call."""
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        answer = self.answers(url, params or {})
        if isinstance(answer, Exception):
            raise answer
        return answer


# ================================================================ backend: display fields

def test_poster_url_keeps_real_urls_and_drops_placeholders():
    real = "https://m.media-amazon.com/images/M/MV5BMTc@._V1_.jpg"
    assert flask_app.poster_url(real) == real
    for bad in ["https://via.placeholder.com/300x450?text=No+Image", "http://placehold.co/300x450",
                "", None, float("nan"), "not a url", "javascript:alert(1)", "/static/poster.jpg"]:
        assert flask_app.poster_url(bad) == "", bad


def test_missing_genres_and_overviews_are_not_given_made_up_values():
    assert flask_app.parse_genre_list(None) == ""
    assert flask_app.parse_genre_list(float("nan")) == ""
    assert flask_app.parse_genre_list("[]") == ""
    assert flask_app.parse_genre_list("['Comedy', 'Drama']") == "Comedy, Drama"
    assert flask_app.full_overview(float("nan")) == "" and flask_app.full_overview("  ") == ""
    long = "x" * 500
    assert flask_app.full_overview(long) == long                  # the card back gets the whole storyline


@pytest.mark.parametrize("mood", sorted(flask_app.VALID_MOODS))
@requires_movie_data
def test_movie_results_never_serve_placeholder_posters(client, mood):
    for mode in ("match", "uplift"):
        body = client.post("/api/recommend", json={"mood": mood, "mode": mode, "type": "movies", "limit": 100}).get_json()
        assert body["results"]
        for m in body["results"]:
            assert m["image"] == "" or re.match(r"^https?://", m["image"])
            assert "placeholder" not in m["image"].lower()
            assert isinstance(m["movie_id"], int)


@requires_movie_data
def test_movie_ranking_is_unchanged_by_the_display_fields(client):
    """The API returns movies in recommend_movies' order; only display fields were touched."""
    movie_df = flask_app.load_movie_data()
    np.random.seed(11)
    expected = flask_app.recommend_movies(movie_df, "sad", "Match my mood", top_n=20)
    np.random.seed(11)
    body = client.post("/api/recommend", json={"mood": "sad", "mode": "match", "type": "movies", "limit": 20}).get_json()
    assert [r["title"] for r in body["results"]] == [flask_app.clean_movie_title(t) for t in expected["title"]]
    assert [r["movie_id"] for r in body["results"]] == [int(i) for i in expected.index]


# ================================================================ backend: refresh / rotation

@requires_music_catalog
def test_a_plain_request_is_one_unmodified_recommender_call(client, monkeypatch):
    calls = []
    real = flask_app.recommend_music
    monkeypatch.setattr(flask_app, "recommend_music", lambda *a, **k: calls.append((a[1:], k)) or real(*a, **k))
    body = client.post("/api/recommend", json={"mood": "happy", "mode": "match", "type": "music", "limit": 30}).get_json()
    assert len(calls) == 1 and calls[0] == (("happy", "Match my mood"), {"top_n": 30})
    assert "refresh" not in body and len(body["results"]) == 30


@pytest.mark.parametrize("media_type,key", [pytest.param("music", "track_id", marks=requires_music_catalog),
                                            pytest.param("movies", "movie_id", marks=requires_movie_data)])
def test_refresh_returns_unseen_items_from_the_same_recommender(client, monkeypatch, media_type, key):
    first = client.post("/api/recommend", json={"mood": "sad", "type": media_type, "limit": 24}).get_json()["results"]
    seen = [r[key] for r in first]
    calls = []
    fn = "recommend_music" if media_type == "music" else "recommend_movies"
    real = getattr(flask_app, fn)
    monkeypatch.setattr(flask_app, fn, lambda *a, **k: calls.append((a[1:], k)) or real(*a, **k))
    body = client.post("/api/recommend", json={"mood": "sad", "type": media_type, "limit": 24, "exclude": seen}).get_json()
    fresh = [r[key] for r in body["results"]]
    assert fresh and not set(fresh) & set(seen)                   # nothing already shown
    assert len(fresh) == len(set(fresh))                          # no duplicates
    assert all(c == (("sad", "Match my mood"), {"top_n": 24}) for c in calls)   # same mood/mode, unmodified calls
    assert body["refresh"]["excluded"] == 24 and body["refresh"]["returned"] == len(fresh)


def test_refresh_reports_exhaustion_instead_of_repeating(client, monkeypatch):
    pool = [{"track_id": f"{i:022d}", "title": "x"} for i in range(5)]
    monkeypatch.setattr(flask_app, "get_music_recommendations", lambda mood, mode, top_n: pool[:top_n])
    body = client.post("/api/recommend", json={"mood": "sad", "type": "music", "limit": 5,
                                                "exclude": [p["track_id"] for p in pool[:3]]}).get_json()
    assert [r["track_id"] for r in body["results"]] == [p["track_id"] for p in pool[3:]]
    assert body["refresh"] == {"excluded": 3, "draws": flask_app.MAX_REFRESH_DRAWS, "returned": 2, "exhausted": True}


@pytest.mark.parametrize("body", [
    {"type": "music", "exclude": "abc"},
    {"type": "music", "exclude": ["not-a-track-id"]},
    {"type": "music", "exclude": ["0" * 22] * (flask_app.MAX_EXCLUDE + 1)},
    {"type": "movies", "exclude": ["12"]},
    {"type": "movies", "exclude": [True]},
    {"type": "movies", "exclude": [-1]},
])
def test_invalid_exclude_lists_are_rejected(client, body):
    res = client.post("/api/recommend", json={"mood": "happy", **body})
    assert res.status_code == 400 and "exclude" in res.get_json()["error"]


# ================================================================ backend: Spotify oEmbed titles

def test_oembed_parse_keeps_only_valid_titles_and_spotify_covers():
    good = {"title": "  Maria Jesus ", "thumbnail_url": "https://image-cdn-ak.spotifycdn.com/image/ab67616d00001e02950abc"}
    assert spotify_oembed.parse(good) == {"title": "Maria Jesus", "artwork": good["thumbnail_url"]}
    assert spotify_oembed.parse({"title": "", "thumbnail_url": "https://evil.example/image/abc"}) == {"title": None, "artwork": None}
    assert spotify_oembed.parse({"title": "x" * 400}) == {"title": None, "artwork": None}
    assert spotify_oembed.parse(None) == {"title": None, "artwork": None}


def test_oembed_lookup_asks_for_the_exact_track_and_caches():
    spotify_oembed.clear_cache()
    tid = "0PBbGtt09Jefz2FqHx99fp"
    session = FakeSession(lambda url, params: FakeResponse({"title": "Real Title", "thumbnail_url": "https://i.scdn.co/image/ab67"}))
    assert spotify_oembed.lookup(tid, session)["title"] == "Real Title"
    assert spotify_oembed.lookup(tid, session)["title"] == "Real Title"        # cached
    assert session.calls == [(spotify_oembed.OEMBED_URL, {"url": f"https://open.spotify.com/track/{tid}"})]
    assert spotify_oembed.lookup("not-an-id", session) is None and len(session.calls) == 1
    spotify_oembed.clear_cache()


def test_oembed_failures_give_no_title(monkeypatch):
    spotify_oembed.clear_cache()
    missing = FakeSession(lambda url, params: FakeResponse({}, status=404))
    assert spotify_oembed.lookup("1" * 22, missing) == {"title": None, "artwork": None, "status": "not_found"}
    down = FakeSession(lambda url, params: requests.ConnectionError("offline"))
    assert spotify_oembed.lookup("2" * 22, down) == {"title": None, "artwork": None, "status": "error"}
    monkeypatch.setenv("MOODCAST_SPOTIFY_OEMBED", "0")
    assert spotify_oembed.lookup("3" * 22, down)["status"] == "disabled"
    spotify_oembed.clear_cache()


def test_track_details_route_validates_ids_and_returns_lookups(client, monkeypatch):
    tid = "0PBbGtt09Jefz2FqHx99fp"
    monkeypatch.setattr(spotify_oembed, "lookup_many",
                        lambda ids: {i: {"title": "Real Title", "artwork": None, "status": "ok"} for i in ids})
    body = client.get(f"/api/track-details?ids={tid},{tid}").get_json()
    assert body["tracks"] == {tid: {"title": "Real Title", "artwork": None}}
    for bad in ["", "?ids=", "?ids=abc", "?ids=" + ",".join(["0" * 22] * 61)]:
        assert client.get("/api/track-details" + bad).status_code == 400


# ================================================================ backend: TMDB posters + watch providers

def _search(*results):
    return {"results": [{"id": i, "title": t, "original_title": o or t, "release_date": d} for i, t, o, d in results]}


def test_tmdb_match_needs_same_title_and_year_and_one_result():
    exact = _search((1, "Ladies & Gentlemen, the Rolling Stones", None, "1973-05-01"))
    assert movie_details.match_movie(exact["results"], "Ladies and Gentlemen: The Rolling Stones", 1973)[0]["id"] == 1
    near = _search((2, "Truman & Tennessee: An Intimate Conversation", None, "2021-06-01"))
    assert movie_details.match_movie(near["results"], "Truman & Tennessee: An Intimate Conversation", 2020)[0]["id"] == 2
    assert movie_details.match_movie(_search((3, "Dhappa", None, "2019-01-01"))["results"], "Dhappa", 2022) == (None, "unmatched")
    assert movie_details.match_movie(_search((4, "Other Film", None, "2020-01-01"))["results"], "Film", 2020) == (None, "unmatched")
    twins = _search((5, "Same", None, "2020-01-01"), (6, "Same", None, "2020-09-01"))
    assert movie_details.match_movie(twins["results"], "Same", 2020) == (None, "ambiguous")
    exact_wins = _search((7, "Film", None, "2019-01-01"), (8, "Film", None, "2020-01-01"))
    assert movie_details.match_movie(exact_wins["results"], "Film", 2020)[0]["id"] == 8


def test_provider_parsing_keeps_india_only_and_never_invents_links():
    payload = {"results": {
        "IN": {"link": "https://www.themoviedb.org/movie/278-x/watch?locale=IN",
               "flatrate": [{"provider_name": "Synthetic Stream", "logo_path": "/abc.png"}],
               "rent": [{"provider_name": "Synthetic Rent", "logo_path": "javascript:x"}, {"provider_name": ""}]},
        "US": {"flatrate": [{"provider_name": "Only In US", "logo_path": "/us.png"}]}}}
    parsed = movie_details.parse_providers(payload)
    assert parsed["region"] == "IN" and parsed["link"].startswith("https://www.themoviedb.org/")
    assert parsed["groups"] == [
        {"type": "flatrate", "label": "Stream", "providers": [{"name": "Synthetic Stream", "logo": "https://image.tmdb.org/t/p/w92/abc.png"}]},
        {"type": "rent", "label": "Rent", "providers": [{"name": "Synthetic Rent", "logo": None}]},
    ]
    assert parsed["verified"] is True
    assert movie_details.parse_providers({"results": {"US": {}}}) == {"region": "IN", "verified": True, "link": None, "groups": []}
    for missing in [None, {}, {"results": None}, "oops"]:                  # no answer is not "no providers"
        assert movie_details.parse_providers(missing) == {"region": "IN", "verified": False, "link": None, "groups": []}
    assert movie_details.parse_providers({"results": {"IN": {"link": "https://evil.example/x"}}})["link"] is None


def test_tmdb_lookup_returns_poster_and_providers_for_a_unique_match(monkeypatch):
    movie_details.clear_cache()
    monkeypatch.setattr(movie_details, "TMDB_API_KEY", "test-key")

    def answers(url, params):
        if url.endswith("/search/movie"):
            return FakeResponse(_search((278, "Synthetic Film", None, "1994-09-23")))
        if url.endswith("/search/tv"):
            return FakeResponse({"results": []})
        assert url.endswith("/movie/278")
        return FakeResponse({"poster_path": "/poster.jpg", "watch/providers": {"results": {"IN": {
            "link": "https://www.themoviedb.org/movie/278/watch?locale=IN",
            "buy": [{"provider_name": "Synthetic Store", "logo_path": "/s.png"}]}}}})
    session = FakeSession(answers)
    out = movie_details.lookup("Synthetic Film", 1994, session)
    assert out["status"] == "matched" and out["tmdb_id"] == 278
    assert out["poster"] == "https://image.tmdb.org/t/p/w342/poster.jpg"
    assert [g["label"] for g in out["providers"]["groups"]] == ["Buy"]
    assert session.calls[-1][1]["append_to_response"] == "watch/providers"
    assert all(p["api_key"] == "test-key" for _, p in session.calls)      # key only in server-side requests
    movie_details.clear_cache()


def test_tmdb_lookup_without_match_or_key_or_network_gives_nothing(monkeypatch):
    movie_details.clear_cache()
    monkeypatch.setattr(movie_details, "TMDB_API_KEY", "test-key")
    none = FakeSession(lambda url, params: FakeResponse({"results": []}))
    assert movie_details.lookup("Nothing Like It", 2001, none) == {
        "status": "unmatched", "media_type": None, "tmdb_id": None, "poster": None,
        "providers": {"region": "IN", "verified": False, "link": None, "groups": []}}
    down = FakeSession(lambda url, params: requests.Timeout("slow"))
    assert movie_details.lookup("Another", 2002, down)["status"] == "error"
    monkeypatch.setattr(movie_details, "TMDB_API_KEY", None)
    assert movie_details.lookup("Third", 2003, down)["status"] == "disabled"
    movie_details.clear_cache()


# Synthetic TMDB answers for the regression tests below. Provider names are
# SYNTHETIC; only the response *shapes* mirror TMDB's watch/providers API.
_SERIES_IN = {"results": {"IN": {
    "link": "https://www.themoviedb.org/tv/1668-x/watch?locale=IN",
    "flatrate": [{"provider_name": "Synthetic Stream A", "logo_path": "/a.png"},
                 {"provider_name": "Synthetic Stream B", "logo_path": "/b.png"}],
    "ads": [{"provider_name": "Synthetic Stream B", "logo_path": "/b.png"}]}}}


def _route(routes):
    """A FakeSession answer function: the first (url suffix -> payload or exception) that matches."""
    def answer(url, params):
        for suffix, payload in routes.items():
            if url.endswith(suffix):
                return payload if isinstance(payload, Exception) else FakeResponse(payload)
        raise AssertionError(f"unexpected TMDB request {url}")
    return answer


def test_tv_series_with_indian_providers_is_resolved_as_tv_not_a_same_named_film(monkeypatch):
    """Regression: Friends (1994) is a TV series. Searching movies only matched an
    unrelated 1994 film with no Indian listing and showed "no options"."""
    movie_details.clear_cache()
    monkeypatch.setattr(movie_details, "TMDB_API_KEY", "test-key")
    by_imdb = FakeSession(_route({
        "/find/tt0108778": {"movie_results": [], "tv_results": [{"id": 1668, "name": "Synthetic Series"}]},
        "/tv/1668": {"poster_path": "/series.jpg", "watch/providers": _SERIES_IN}}))
    out = movie_details.lookup("Synthetic Series", 1994, by_imdb, imdb_id=108778)
    assert out["status"] == "matched" and out["media_type"] == "tv" and out["tmdb_id"] == 1668
    assert by_imdb.calls[0][1]["external_source"] == "imdb_id"
    assert not any("/movie/" in url or "/search/" in url for url, _ in by_imdb.calls)
    groups = {g["type"]: [p["name"] for p in g["providers"]] for g in out["providers"]["groups"]}
    assert groups == {"flatrate": ["Synthetic Stream A", "Synthetic Stream B"], "ads": ["Synthetic Stream B"]}
    assert out["providers"]["verified"] and out["providers"]["link"].startswith("https://www.themoviedb.org/tv/")

    # without an IMDb id: the same-named film exists too, but only the series has this exact title + year
    movie_details.clear_cache()
    by_title = FakeSession(_route({
        "/search/movie": _search((196924, "Friends Forever", None, "1994-01-14")),
        "/search/tv": {"results": [{"id": 1668, "name": "Synthetic Series", "original_name": "Synthetic Series",
                                    "first_air_date": "1994-09-22"}]},
        "/tv/1668": {"poster_path": "/series.jpg", "watch/providers": _SERIES_IN}}))
    assert movie_details.lookup("Synthetic Series", 1994, by_title)["media_type"] == "tv"

    # a film and a series with the same exact title + year: not guessed
    movie_details.clear_cache()
    clash = FakeSession(_route({
        "/search/movie": _search((5, "Synthetic Series", None, "1994-01-14")),
        "/search/tv": {"results": [{"id": 1668, "name": "Synthetic Series", "first_air_date": "1994-09-22"}]}}))
    assert movie_details.lookup("Synthetic Series", 1994, clash)["status"] == "ambiguous"
    movie_details.clear_cache()


def test_movie_with_providers_by_imdb_id_reads_every_category(monkeypatch):
    movie_details.clear_cache()
    monkeypatch.setattr(movie_details, "TMDB_API_KEY", "test-key")
    providers = {"results": {"IN": {
        "link": "https://www.themoviedb.org/movie/603-x/watch?locale=IN",
        **{key: [{"provider_name": f"Synthetic {key}", "logo_path": f"/{key}.png"}]
           for key in ["flatrate", "free", "ads", "rent", "buy"]}}}}
    session = FakeSession(_route({
        "/find/tt0133093": {"movie_results": [{"id": 603}], "tv_results": []},
        "/movie/603": {"poster_path": "/m.jpg", "watch/providers": providers}}))
    out = movie_details.lookup("Synthetic Film", 1999, session, imdb_id=133093)
    assert out["status"] == "matched" and out["media_type"] == "movie" and out["tmdb_id"] == 603
    assert [g["label"] for g in out["providers"]["groups"]] == ["Stream", "Free", "Free with ads", "Rent", "Buy"]
    movie_details.clear_cache()


def test_successful_response_without_indian_providers_is_verified_empty(monkeypatch):
    movie_details.clear_cache()
    monkeypatch.setattr(movie_details, "TMDB_API_KEY", "test-key")
    session = FakeSession(_route({
        "/find/tt0000001": {"movie_results": [{"id": 9}], "tv_results": []},
        "/movie/9": {"poster_path": None, "watch/providers": {"results": {"US": {"flatrate": [{"provider_name": "US Only"}]}}}}}))
    out = movie_details.lookup("Synthetic Film", 2000, session, imdb_id=1)
    assert out["status"] == "matched"
    assert out["providers"] == {"region": "IN", "verified": True, "link": None, "groups": []}
    # TMDB answered but left out watch/providers: unverified, not "none"
    movie_details.clear_cache()
    partial = FakeSession(_route({"/find/tt0000002": {"movie_results": [{"id": 10}]}, "/movie/10": {"poster_path": None}}))
    assert movie_details.lookup("Other", 2000, partial, imdb_id=2)["providers"]["verified"] is False
    movie_details.clear_cache()


def test_api_failure_is_an_error_not_an_empty_listing_and_is_retried(monkeypatch):
    movie_details.clear_cache()
    monkeypatch.setattr(movie_details, "TMDB_API_KEY", "test-key")
    failing = FakeSession(_route({"/find/tt0108778": {"movie_results": [], "tv_results": [{"id": 1668}]},
                                  "/tv/1668": FakeResponse({}, status=503)}))
    out = movie_details.lookup("Synthetic Series", 1994, failing, imdb_id=108778)
    assert out["status"] == "error" and out["providers"]["verified"] is False and out["providers"]["groups"] == []
    # an error is cached only briefly, so a later request asks TMDB again
    working = FakeSession(_route({"/find/tt0108778": {"movie_results": [], "tv_results": [{"id": 1668}]},
                                  "/tv/1668": {"watch/providers": _SERIES_IN}}))
    monkeypatch.setattr(movie_details.time, "monotonic", lambda: 10 ** 9)
    assert movie_details.lookup("Synthetic Series", 1994, working, imdb_id=108778)["status"] == "matched"
    movie_details.clear_cache()


@requires_movie_data
def test_route_passes_the_imdb_id_of_a_dataset_tv_series(client, monkeypatch):
    seen = []
    monkeypatch.setattr(movie_details, "lookup_many", lambda items: seen.extend(items) or {})
    movie_df = flask_app.load_movie_data()
    friends = int(movie_df.index[movie_df["title"] == "Friends: (1994)"][0])
    client.post("/api/movie-details", json={"movie_ids": [friends]})
    assert seen == [(friends, "Friends", 1994, 108778)]                 # IMDb tt0108778, the TV series


@requires_movie_data
def test_movie_details_route_looks_up_dataset_titles_server_side(client, monkeypatch):
    seen = []
    monkeypatch.setattr(movie_details, "lookup_many", lambda items: seen.extend(items) or
                        {mid: {"status": "unmatched", "tmdb_id": None, "poster": None,
                               "providers": {"region": "IN", "link": None, "groups": []}} for mid, *_ in items})
    movie_df = flask_app.load_movie_data()
    body = client.post("/api/movie-details", json={"movie_ids": [0, 1]}).get_json()
    assert body["region"] == "IN" and set(body["movies"]) == {"0", "1"}
    assert seen[0] == (0, flask_app.clean_movie_title(movie_df.loc[0, "title"]), int(movie_df.loc[0, "release_year"]),
                       271023)                                         # IMDb id of row 0 in the raw dataset
    for bad in [{}, {"movie_ids": []}, {"movie_ids": ["0"]}, {"movie_ids": [10 ** 9]}, {"movie_ids": list(range(61))}]:
        assert client.post("/api/movie-details", json=bad).status_code == 400
    assert "TMDB_API_KEY" not in json.dumps(body) and "api_key" not in json.dumps(body)


# ================================================================ movies page (real movies.js)

# TMDB details the page renders directly (watchHtml), for the "Where to watch" regressions.
WATCH_CASES = {
    "series": {"status": "matched", "media_type": "tv", "providers": {"region": "IN", "verified": True,
               "link": "https://www.themoviedb.org/tv/1668-x/watch?locale=IN",
               "groups": [{"type": "flatrate", "label": "Stream", "providers": [{"name": "Synthetic Stream A", "logo": None}]},
                          {"type": "ads", "label": "Free with ads", "providers": [{"name": "Synthetic Stream B", "logo": None}]}]}},
    "movie": {"status": "matched", "media_type": "movie", "providers": {"region": "IN", "verified": True, "link": None,
              "groups": [{"type": "rent", "label": "Rent", "providers": [{"name": "Synthetic Rent", "logo": None}]},
                         {"type": "buy", "label": "Buy", "providers": [{"name": "Synthetic Buy", "logo": None}]}]}},
    "none": {"status": "matched", "providers": {"region": "IN", "verified": True,
             "link": "https://www.themoviedb.org/movie/9-x/watch?locale=IN", "groups": []}},
    "unverified": {"status": "matched", "providers": {"region": "IN", "verified": False, "link": None, "groups": []}},
    "legacy": {"status": "matched", "providers": {"region": "IN", "link": None, "groups": []}},   # cached before `verified`
    "error": {"status": "error", "providers": {"region": "IN", "verified": False, "link": None, "groups": []}},
}


@pytest.fixture(scope="module")
def movies_page(tmp_path_factory):
    client = flask_app.app.test_client()
    response = client.post("/api/recommend", json={"mood": "happy", "mode": "match", "type": "movies", "limit": 60}).get_json()
    pool = response["results"]
    refresh = client.post("/api/recommend", json={"mood": "happy", "mode": "match", "type": "movies", "limit": 60,
                                                   "exclude": [m["movie_id"] for m in pool]}).get_json()
    details, broken = {}, []
    tmdb_poster = lambda i: f"https://image.tmdb.org/t/p/w342/synthetic{i}.jpg"
    for i, m in enumerate(pool):
        providers = {"region": "IN", "verified": i % 8 != 1, "link": None, "groups": []}
        if i % 4 == 0:
            providers = {"region": "IN", "link": f"https://www.themoviedb.org/movie/{i}/watch?locale=IN",
                         "groups": [{"type": "flatrate", "label": "Stream",
                                     "providers": [{"name": "Synthetic Stream", "logo": "https://image.tmdb.org/t/p/w92/s.png"}]}]}
        status = "unmatched" if i % 4 == 2 else "matched"
        details[str(m["movie_id"])] = {"status": status, "tmdb_id": i if status == "matched" else None,
                                       "poster": tmdb_poster(i) if status == "matched" else None, "providers": providers}
        if i % 3 == 1 and m["image"]:
            broken.append(_sized(m["image"]))          # dataset poster dead...
            if i % 2 == 1:
                broken.append(tmdb_poster(i))          # ...and for some, the TMDB one too
    fixture = tmp_path_factory.mktemp("movies_page") / "fixture.json"
    fixture.write_text(json.dumps({"response": response, "refresh": refresh, "details": details, "broken": broken,
                                   "watchCases": WATCH_CASES}), encoding="utf-8")
    return response, refresh, details, set(broken), _run("movies_page_harness.js", fixture)


def _poster_for(movie, details, broken):
    """The poster the page should show for a movie, or None (dataset first, then a matched TMDB poster)."""
    sized = _sized(movie["image"]) if movie["image"] else None
    if sized and sized not in broken:
        return sized
    d = details.get(str(movie["movie_id"]))
    if d and d["poster"] and d["poster"] not in broken:
        return d["poster"]
    return None


@requires_node
@requires_movie_data
def test_movies_page_requests_recommendations_then_details(movies_page):
    response, _, _, _, page = movies_page
    assert page["requests"][0] == {"path": "/api/recommend", "body": {"mood": "happy", "mode": "match", "type": "movies", "limit": 60}}
    assert page["requests"][1]["path"] == "/api/movie-details"
    assert page["requests"][1]["body"]["movie_ids"] == [m["movie_id"] for m in response["results"]]
    assert page["loadingBeforePosters"] is True


@requires_node
@requires_movie_data
def test_movies_render_as_a_grid_of_poster_backed_cards_in_ranked_order(movies_page):
    response, _, details, broken, page = movies_page
    expected = [(i, _poster_for(m, details, broken)) for i, m in enumerate(response["results"])]
    expected = [(i, p) for i, p in expected if p][:24]
    assert page["cards"] == len(expected) > 1                           # many cards, not one
    assert page["cardApiIndex"] == [i for i, _ in expected]             # backend order, only real results
    assert page["cardSrcs"] == [p for _, p in expected]                 # dataset poster, else verified TMDB poster
    assert any("image.tmdb.org" in s for s in page["cardSrcs"])         # the TMDB fallback is used
    assert not set(page["cardSrcs"]) & broken                           # never a poster that fails to load
    assert "skipped (no working poster)" in page["status"] and "posters from TMDB" in page["status"]


@requires_node
@requires_movie_data
def test_card_backs_show_storyline_rating_and_where_to_watch(movies_page):
    response, _, details, _, page = movies_page
    for idx, html in zip(page["cardApiIndex"], page["cardHtml"]):
        m = response["results"][idx]
        back = html.split('class="movie-face movie-back"')[1]
        if m["rating"] != "N/A":
            assert f"{m['rating']}<span class=\"out-of\"> / 10" in back
        if m["overview"]:
            overview = m["overview"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
            assert overview in back                                     # the full storyline, not the 200-char cut
        assert "Where to watch" in back and "India" in back
        d = details[str(m["movie_id"])]
        if d["providers"]["groups"]:
            assert "Synthetic Stream" in back and 'href="https://www.themoviedb.org/movie/' in back
            assert 'target="_blank" rel="noopener noreferrer"' in back
        elif d["status"] == "matched" and d["providers"]["verified"]:
            assert "No listed options for India." in back
        else:
            assert "Availability could not be verified." in back and "No listed options" not in back
    assert page["backHasInert"] is True
    assert page["attribution"] == "Streaming data from JustWatch via TMDB"


@requires_node
@requires_movie_data
def test_where_to_watch_shows_providers_or_says_it_could_not_verify(movies_page):
    *_, page = movies_page
    html = page["watchHtmlFor"]
    assert "Synthetic Stream A" in html["series"] and "Free with ads" in html["series"] and "Synthetic Stream B" in html["series"]
    assert 'href="https://www.themoviedb.org/tv/1668-x/watch?locale=IN"' in html["series"]
    assert "Rent" in html["movie"] and "Synthetic Rent" in html["movie"] and "Synthetic Buy" in html["movie"]
    assert "No listed options for India." in html["none"] and 'href="https://www.themoviedb.org/movie/9-x/' in html["none"]
    for name in ["unverified", "legacy", "error"]:
        assert "Availability could not be verified." in html[name], name
    for name in ["series", "movie", "unverified", "legacy", "error"]:
        assert "No listed options" not in html[name], name


@requires_node
@requires_movie_data
def test_missing_movie_fields_are_labelled_not_invented(movies_page):
    *_, page = movies_page
    f, html = page["missingFields"]["fields"], page["missingFields"]["html"]
    assert f["rating"] is None and f["year"] is None and f["genre"] is None and f["overview"] is None
    assert "Rating unavailable" in html and "No storyline available" in html and "Genre unavailable" in html
    assert "Availability could not be verified." in html
    assert "N/A" not in html and "No description available." not in html
    assert page["watchKinds"][:3] == ["unknown", "unmatched", "unverified"] and page["watchKinds"][4:] == ["error", "none"]
    assert page["watchKinds"][3]["kind"] == "providers" and page["watchKinds"][3]["link"] is None   # foreign link dropped


@requires_node
@requires_movie_data
def test_cards_flip_by_click_or_tap_and_by_their_buttons(movies_page):
    *_, page = movies_page
    assert page["flipByCardClick"] is True
    assert page["frontInertWhenFlipped"] is True and page["backInertWhenFlipped"] is False
    assert page["backAriaHiddenWhenFlipped"] == "false"
    assert page["unflippedByBackButton"] is True
    assert page["focusAfterUnflip"] == '[data-flip="open"]'           # focus never stranded on a hidden face
    assert page["flippedByFlipButton"] is True
    assert page["focusAfterFlip"] == '[data-flip="close"]'
    assert page["ariaExpanded"] == "true"
    assert page["unflippedByEscape"] is True


@requires_node
@requires_movie_data
def test_actions_and_provider_links_do_not_flip_cards(movies_page):
    *_, page = movies_page
    assert page["flippedAfterFavorite"] is True and page["favoritePressed"] == "true"
    assert page["flippedAfterProviderLink"] is True
    assert page["flippedAfterDetails"] is True and page["modalShown"] is True and page["modalInert"] is False
    assert page["modalTitle"] and "Where to watch" in page["modalWatch"]
    assert page["modalClosedByEscape"] is True


@requires_node
@requires_movie_data
def test_a_poster_that_fails_later_removes_its_card(movies_page):
    *_, page = movies_page
    before, after = page["cardsAfterLateError"]
    assert after == before - 1


@requires_node
@requires_movie_data
def test_results_stay_put_until_refresh_and_refresh_avoids_seen_movies(movies_page):
    response, refresh, _, _, page = movies_page
    assert page["requestsOnRevisit"] == 0 and page["cardsOnRevisit"] > 0
    body = page["refreshBody"]
    assert body["mood"] == "happy" and body["mode"] == "match" and body["type"] == "movies"
    assert set(body["exclude"]) == {m["movie_id"] for m in response["results"]}    # everything shown or checked
    assert page["cardsAfterRefresh"] and -1 not in page["cardsAfterRefresh"]      # cards now come from the refresh
    assert not {m["movie_id"] for m in refresh["results"]} & set(body["exclude"])


@requires_node
@requires_movie_data
def test_mood_and_mode_changes_request_and_show_new_results(movies_page):
    *_, page = movies_page
    assert page["changeRequests"] == [{"mood": "sad", "mode": "match", "type": "movies", "limit": 60},
                                      {"mood": "sad", "mode": "uplift", "type": "movies", "limit": 60}]
    assert page["activePills"] == ["sad"]
    assert page["titleAfterChanges"] == "Films to lift a sad mood"


@requires_node
@requires_movie_data
def test_errors_and_empty_results_show_a_message_not_stale_cards(movies_page):
    *_, page = movies_page
    assert page["cardsWhenDetailsFail"] > 0                             # dataset posters still work
    assert "Availability could not be verified." in page["watchWhenDetailsFail"]
    assert "No listed options" not in page["watchWhenDetailsFail"]
    assert page["attributionWhenDetailsFail"] == "Streaming data from JustWatch via TMDB"   # credit always shown
    assert page["errorCards"] == 0 and "Failed to fetch" in page["errorHtml"] and "retry-btn" in page["errorHtml"]
    assert page["errorStatus"] == "Recommendations could not be loaded"
    assert page["emptyCards"] == 0 and "No movies found" in page["emptyHtml"]
    assert page["noPosterCards"] == 0 and "No poster-backed picks" in page["noPosterHtml"]
    assert page["noPosterStatus"] == "No poster loads for any of the top 1 recommendation"


@requires_node
@requires_movie_data
def test_poster_display_url_resizes_only_the_same_cdn_image(movies_page):
    *_, page = movies_page
    assert page["posterUrls"] == [
        "https://m.media-amazon.com/images/M/MV5BNDVk@._V1_SX400.jpg",
        "https://m.media-amazon.com/images/M/MV5Bx@@._V1_SX300.jpg",   # already sized: unchanged
        "https://example.org/poster.jpg",                               # other hosts: unchanged
        None, None, None,
    ]


# ================================================================ music page flow (real music.js)

@pytest.fixture(scope="module")
def music_flow(tmp_path_factory):
    client = flask_app.app.test_client()
    body = client.post("/api/recommend", json={"mood": "happy", "type": "music", "limit": 30}).get_json()
    ids = [r["track_id"] for r in body["results"]]
    refresh = client.post("/api/recommend", json={"mood": "happy", "type": "music", "limit": 30, "exclude": ids}).get_json()
    tmp = tmp_path_factory.mktemp("music_flow")
    (tmp / "r.json").write_text(json.dumps(body), encoding="utf-8")
    (tmp / "d.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp / "f.json").write_text(json.dumps(refresh), encoding="utf-8")
    return ids, refresh, _run("music_page_harness.js", tmp / "r.json", tmp / "d.json", tmp / "f.json")


@requires_node
@requires_music_catalog
def test_music_requests_recommendations_then_titles_for_the_shown_tracks(music_flow):
    *_, page = music_flow
    assert page["firstRequests"][0] == {"path": "/api/recommend", "ids": 0,
                                        "body": {"mood": "happy", "mode": "match", "type": "music", "limit": 30}}
    assert page["firstRequests"][1]["path"] == "/api/track-details" and page["firstRequests"][1]["ids"] == 30


@requires_node
@requires_music_catalog
def test_music_stays_put_until_refresh_and_refresh_avoids_seen_tracks(music_flow):
    ids, refresh, page = music_flow
    assert page["requestsOnRevisit"] == 0
    assert page["refreshBody"]["mood"] == "happy" and page["refreshBody"]["mode"] == "match"
    assert set(page["refreshBody"]["exclude"]) == set(ids)
    assert page["rowsAfterRefresh"] == [r["track_id"] for r in refresh["results"]]
    assert not set(page["rowsAfterRefresh"]) & set(ids)


@requires_node
@requires_music_catalog
def test_music_mood_and_mode_changes_request_new_results(music_flow):
    *_, page = music_flow
    assert page["changeRequests"] == [{"mood": "sad", "mode": "match", "type": "music", "limit": 30},
                                      {"mood": "sad", "mode": "uplift", "type": "music", "limit": 30}]
    assert page["activePill"] == ["sad"] and page["titleAfterChanges"] == "Tracks to lift a sad mood"


@requires_node
@requires_music_catalog
def test_music_titles_settle_honestly_when_oembed_fails_and_errors_show(music_flow):
    *_, page = music_flow
    assert "Loading title…" not in page["headingsWhenDetailsFail"]
    assert not any(h and "Unknown" in h for h in page["headingsWhenDetailsFail"])
    assert page["errorRows"] == 0 and "Server exploded" in page["errorHtml"] and "retry-btn" in page["errorHtml"]
    assert page["refreshDisabledAfterError"] is False


# ================================================================ structure the behaviour relies on

def test_music_page_has_no_in_page_player():
    html = _read("templates", "music.html")
    js = _read("static", "js", "music.js")
    for marker in ["now-playing", "npPlay", "npNext", "npPrev", "npWaveform", "playAllBtn", "queueBadge", "<audio"]:
        assert marker not in html, marker
    assert "padding-bottom: 120px" not in html                          # no space reserved for a player
    for marker in ["function playTrack", "setInterval", "new Audio", ".play()", "autoplay"]:
        assert marker not in js, marker


def test_movies_page_has_no_fabricated_extras():
    js = _read("static", "js", "movies.js")
    html = _read("templates", "movies.html")
    assert "mulberry32" not in js and "estimate" not in js.lower()      # no seeded fake scores
    assert "Trailer" not in html and "trailer" not in js                 # no trailer data exists
    assert "hero-banner" not in html                                     # no single oversized top card
    for brand in ["Netflix", "Prime Video", "Disney", "Hotstar"]:       # providers only ever come from TMDB data
        assert brand not in js and brand not in html


def test_no_api_keys_in_frontend_files():
    for folder in ("static/js", "templates"):
        for name in os.listdir(os.path.join(BASE_DIR, folder)):
            text = _read(folder, name)
            assert "api_key" not in text and "TMDB_API_KEY" not in text and "api.themoviedb.org" not in text, name


def test_flip_card_css_hides_back_faces_and_pre_rotates_the_back():
    html = _read("templates", "movies.html")
    face = re.search(r"\.movie-face \{([^}]*)\}", html).group(1)
    assert "backface-visibility: hidden" in face and "position: absolute; inset: 0" in face and "overflow: hidden" in face
    assert re.search(r"\.movie-back \{ transform: rotateY\(180deg\); \}", html)       # reads normally once flipped
    assert re.search(r"\.movie-card\.is-flipped \.movie-card-inner \{ transform: rotateY\(180deg\); \}", html)
    assert re.search(r"\.movie-card \{[^}]*aspect-ratio: 2 / 3", html)                # fixed size: no grid jump
    assert re.search(r"\.back-scroll \{[^}]*overflow-y: auto", html)                  # long storylines scroll inside the card
    reduced = html[html.index("@media (prefers-reduced-motion: reduce)"):]
    assert ".movie-card-inner { transition: none; }" in reduced[:200]


def test_pages_stay_scrollable():
    """overflow: hidden on <body> propagates to the viewport and kills wheel/touch/keyboard scrolling."""
    css = _read("static", "css", "design-system.css")
    assert re.search(r"body\.ripple-host \{ overflow-x: hidden; overflow-y: visible; \}", css)
    for template in ["index.html", "music.html", "movies.html", "dashboard.html"]:
        html = _read("templates", template)
        assert not re.search(r"(html|body)\s*\{[^}]*overflow(-y)?:\s*hidden", html), template
        assert not re.search(r"(html|body|main)\s*\{[^}]*height:\s*100vh", html), template


def test_navbar_fits_phone_widths():
    css = _read("static", "css", "design-system.css")
    block = css[css.index("@media (max-width: 600px) {\n  .navbar {"):]
    block = block[:block.index("\n}\n")]
    assert "flex-wrap: wrap" in block and "width: 100%" in block and "position: relative" in block
    for template in ["index.html", "music.html", "movies.html", "dashboard.html"]:
        nav = _read("templates", template)
        assert all(f'href="{p}"' in nav for p in ["/", "/music", "/movies", "/dashboard"]), template
