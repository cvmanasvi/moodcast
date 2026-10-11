"""Frontend tests for the Popular Artists ON/OFF switch and its Refresh on the Music page.

tests/js/popular_mode_harness.js runs the real music.js against SYNTHETIC,
fully scripted /api/recommend and /api/popular responses (made-up ids, titles
and artist names; no random draws). Checked: exactly one switch whose
aria-checked, visual state and app state agree; OFF restores the normal list
without a request; selection changes while ON; Popular Artists Refresh (right
endpoint and body, exclusion of shown tracks, replacement only on success,
short and exhausted pools, server errors, unavailable data); separate caches
for the two modes and an unchanged normal Refresh; no category/badge text.
"""
import json
import os
import re
import shutil
import subprocess

import pytest

from src.config import BASE_DIR

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
CATEGORY_TEXT = re.compile(r"international|k-?pop|indian|preferred|category|artist-badge", re.IGNORECASE)
ON = {"checked": "true", "data": "on", "text": "On"}
OFF = {"checked": "false", "data": "off", "text": "Off"}


def _read(*parts):
    with open(os.path.join(BASE_DIR, *parts), encoding="utf-8") as f:
        return f.read()


def _tracks(prefix, n, artist=lambda i: f"Synthetic Artist {i}"):
    return [{"track_id": f"{prefix}{i:021d}", "title": f"Synthetic {prefix} {i}", "artist": artist(i),
             "artists": [], "has_metadata": True, "release_year": 2020, "year": 2020,
             "spotify_url": f"https://open.spotify.com/track/{prefix}{i:021d}", "duration_ms": 200000,
             "duration": "3:20", "mood": "happy", "mood_confidence": 0.8, "image": "",
             "audio_features": {"valence": 0.5, "energy": 0.5, "danceability": 0.5, "tempo": 120.0, "acousticness": 0.2}}
            for i in range(n)]


def _popular(results, **extra):
    return {"mood": "happy", "mode": "match", "available": True, "eligible": 300, "limited": False,
            "limited_by": None, "results": results, **extra}


FIXTURES = {
    "normal": {"mood": "happy", "mode": "match", "results": _tracks("N", 30)},
    "normalRefresh": {"mood": "happy", "mode": "match", "results": _tracks("M", 30),
                      "refresh": {"excluded": 30, "draws": 1, "returned": 30, "exhausted": False}},
    "popular": _popular(_tracks("P", 30, lambda i: "Synthetic Star, The Comma Band" if i == 0 else f"Synthetic Star {i}")),
    "popularRefreshes": [
        _popular(_tracks("R", 30), refresh={"excluded": 30, "returned": 30, "exhausted": False}),
        _popular(_tracks("S", 8), eligible=40, limited=True, limited_by="artist_cap",
                 refresh={"excluded": 60, "returned": 8, "exhausted": True}),
        _popular([], eligible=0, limited=True, limited_by="pool", refresh={"excluded": 68, "returned": 0, "exhausted": True}),
        {"status": 500, "error": "Server exploded"},
        {"mood": "happy", "mode": "match", "available": False, "results": [], "message": "Popular Artists needs its local dataset."},
    ],
}


def _ids(body):
    return [t["track_id"] for t in body["results"]]


NORMAL, POPULAR = _ids(FIXTURES["normal"]), _ids(FIXTURES["popular"])
R1, R2 = (_ids(r) for r in FIXTURES["popularRefreshes"][:2])


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    path = tmp_path_factory.mktemp("popular_flow") / "fixtures.json"
    path.write_text(json.dumps(FIXTURES), encoding="utf-8")
    out = subprocess.run([NODE, os.path.join("tests", "js", "popular_mode_harness.js"), BASE_DIR, str(path)],
                         cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _paths(step):
    return [r["path"] for r in step["requests"]]


# ---------------------------------------------------------------- switch ON / OFF

@requires_node
def test_page_starts_off_with_normal_recommendations(page):
    assert _paths(page["initial"]) == ["/api/recommend"] and page["initial"]["rows"] == NORMAL
    assert page["initial"]["switch"] == OFF


@requires_node
def test_switching_on_shows_popular_for_the_current_selection(page):
    on = page["on"]
    assert on["requests"] == [{"path": "/api/popular", "body": {"mood": "happy", "mode": "match", "limit": 30}}]
    assert on["rows"] == POPULAR and on["switch"] == ON
    assert "Popular Artists" in on["kicker"] and on["title"] == "Popular happy songs"
    assert "not chosen by people" in on["sourceNote"] and "mood predicted from audio features" in on["status"]
    assert on["artistLines"][0].startswith("Synthetic Star, The Comma Band")        # source string, unsplit
    assert on["spotifyLinks"] == [t["spotify_url"] for t in FIXTURES["popular"]["results"]]
    assert not CATEGORY_TEXT.search(on["html"]) and "<audio" not in on["html"]


@requires_node
def test_switching_off_restores_normal_without_fetching_and_on_again_reuses_popular(page):
    assert page["off"]["requests"] == [] and page["off"]["rows"] == NORMAL and page["off"]["switch"] == OFF
    assert "Popular Artists" not in page["off"]["kicker"]
    assert page["onAgain"]["requests"] == [] and page["onAgain"]["rows"] == POPULAR and page["onAgain"]["switch"] == ON


@requires_node
def test_mood_and_mode_changes_while_on_update_popular(page):
    assert page["moodChange"]["requests"] == [{"path": "/api/popular", "body": {"mood": "sad", "mode": "match", "limit": 30}}]
    assert page["modeChange"]["requests"] == [{"path": "/api/popular", "body": {"mood": "sad", "mode": "uplift", "limit": 30}}]
    assert page["moodChange"]["switch"] == ON and page["modeChange"]["switch"] == ON
    assert page["modeChange"]["title"] == "Popular songs to lift a sad mood"


# ---------------------------------------------------------------- Popular Artists Refresh

@requires_node
def test_refresh_requests_unseen_popular_tracks_and_replaces_the_list(page):
    r1 = page["refresh1"]
    assert _paths(r1) == ["/api/popular"]                                           # never /api/recommend
    body = r1["requests"][0]["body"]
    assert body["mood"] == "happy" and body["mode"] == "match" and body["limit"] == 30
    assert sorted(body["exclude"]) == sorted(POPULAR)
    assert r1["rows"] == R1 and r1["switch"] == ON and r1["status"].startswith("30 tracks")
    assert "Fresh picks from popular artists" in r1["status"]
    assert r1["popularCache"] == R1 and r1["normalCache"] == NORMAL


@requires_node
def test_exclusions_accumulate_and_a_short_list_says_why(page):
    r2 = page["refresh2"]
    assert sorted(r2["requests"][0]["body"]["exclude"]) == sorted(POPULAR + R1)
    assert r2["rows"] == R2 and r2["refreshDisabled"] is False
    assert "only 8 new songs could be picked without repeating an artist more than twice" in r2["status"]


@requires_node
def test_an_exhausted_pool_keeps_the_current_list(page):
    r3 = page["refresh3"]
    assert sorted(r3["requests"][0]["body"]["exclude"]) == sorted(POPULAR + R1 + R2)
    assert r3["rows"] == R2 and r3["switch"] == ON
    assert "No other tracks are available for this mood right now" in r3["status"]


@requires_node
def test_failed_or_unavailable_refresh_keeps_the_last_valid_list_and_stays_on(page):
    r4, r5 = page["refresh4"], page["refresh5"]
    assert r4["rows"] == R2 and r4["switch"] == ON and "Couldn't refresh (Server exploded)" in r4["status"]
    assert r5["rows"] == R2 and r5["switch"] == ON and "Couldn't refresh" in r5["status"] and "unavailable" in r5["status"]
    after = page["afterRefreshes"]
    assert after["popularCache"] == R2                                              # last successful refresh
    assert sorted(after["popularSeen"]) == sorted(POPULAR + R1 + R2)
    assert after["normalCache"] == NORMAL and sorted(after["normalSeen"]) == sorted(NORMAL)


@requires_node
def test_normal_refresh_is_unchanged_and_leaves_popular_alone(page):
    assert page["offAfterRefresh"]["requests"] == [] and page["offAfterRefresh"]["rows"] == NORMAL
    nr = page["normalRefresh"]
    assert nr["requests"] == [{"path": "/api/recommend", "body": {
        "mood": "happy", "mode": "match", "type": "music", "limit": 30, "exclude": NORMAL}}]
    assert nr["rows"] == _ids(FIXTURES["normalRefresh"]) and nr["switch"] == OFF
    assert nr["popularCache"] == R2 and nr["normalCache"] == _ids(FIXTURES["normalRefresh"])
    back = page["onAfterNormalRefresh"]
    assert back["requests"] == [] and back["rows"] == R2 and back["switch"] == ON


# ---------------------------------------------------------------- empty and unavailable

@requires_node
def test_empty_and_unavailable_states_are_honest_and_normal_mode_recovers(page):
    assert page["empty"]["rows"] == [] and "No popular songs fit this mood yet" in page["empty"]["list"]
    un = page["unavailable"]
    assert un["rows"] == [] and "isn't available yet" in un["list"] and "local dataset" in un["list"]
    assert "unavailable" in un["status"] and un["popularCache"] is None and un["switch"] == ON
    back = page["backToNormal"]
    assert _paths(back) == ["/api/recommend"] and back["rows"] == NORMAL and back["switch"] == OFF


# ---------------------------------------------------------------- structure

def test_template_has_exactly_one_accessible_switch_and_no_category_controls():
    html = _read("templates", "music.html")
    body = re.sub(r"<!--.*?-->", "", html.split("<body", 1)[1], flags=re.S)   # visible markup only
    assert html.count('id="popularSwitch"') == 1 and body.count("Popular Artists") == 1 and "popularBtn" not in html
    tag = re.search(r'<button[^>]*id="popularSwitch"[^>]*>', html).group(0)
    # a native button: focusable, and Enter/Space activate it (as a click) without extra key handling
    for attr in ['type="button"', 'role="switch"', 'aria-checked="false"', 'aria-labelledby="popularLabel"']:
        assert attr in tag, attr
    assert 'id="popularLabel">Popular Artists<' in html
    assert html.index('id="modeToggle"') < html.index('id="popularSwitch"') < html.index('id="trackList"')
    assert '.popular-switch[aria-checked="true"]' in html and ".switch-thumb" in html     # ON is visibly different
    assert not re.search(r"data-category|K-pop|Indian|International", body)


def test_music_js_drives_the_switch_from_one_state_and_never_renders_categories():
    js = _read("static", "js", "music.js")
    assert not re.search(r"category|artist-badge|priority_artist|preferred", js, re.IGNORECASE)
    assert "popularBtn" not in js                                                      # the old push button is gone
    assert js.count("setAttribute('aria-checked'") == 1                                # one renderer
    assert "/api/popular" in js and "'popular' : 'music'" in js                        # separate cache keys
