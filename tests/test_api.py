"""API contract tests for flask_app: valid requests, 4xx validation, and
error handling that doesn't swallow or leak failures."""
import pytest

import flask_app
from src import data_loader
from tests.local_data import requires_movie_data, requires_music_catalog


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=False)  # exercise the real error handlers
    return flask_app.app.test_client()


def recommend(client, body):
    return client.post('/api/recommend', json=body)


# ---- /api/recommend: happy paths ----

@pytest.mark.parametrize('media_type', [pytest.param('music', marks=requires_music_catalog),
                                        pytest.param('movies', marks=requires_movie_data)])
@pytest.mark.parametrize('mode', ['match', 'uplift'])
def test_recommend_returns_results_within_limit(client, media_type, mode):
    res = recommend(client, {'mood': 'sad', 'mode': mode, 'type': media_type, 'limit': 7})
    assert res.status_code == 200
    body = res.get_json()
    assert body['mood'] == 'sad' and body['mode'] == mode
    assert 1 <= len(body['results']) <= 7
    assert all('title' in r for r in body['results'])


@requires_movie_data
def test_recommend_defaults_mode_and_limit(client):
    res = recommend(client, {'mood': 'happy', 'type': 'movies'})
    assert res.status_code == 200
    body = res.get_json()
    assert body['mode'] == 'match'
    assert len(body['results']) <= flask_app.DEFAULT_LIMIT


# ---- /api/recommend: a dataset that isn't on this machine ----

@pytest.mark.parametrize('media_type,path_name,cache_name', [
    ('music', 'MUSIC_CATALOG_278K_PATH', '_music_catalog_df'),
    ('movies', 'MOVIES_FINAL_PATH', '_movies_df'),
])
def test_missing_dataset_is_an_honest_503(client, monkeypatch, tmp_path, media_type, path_name, cache_name):
    monkeypatch.setattr(data_loader, path_name, str(tmp_path / 'absent.csv'))
    monkeypatch.setattr(data_loader, cache_name, None)
    res = recommend(client, {'mood': 'sad', 'type': media_type})
    assert res.status_code == 503
    body = res.get_json()
    assert body['dataset_unavailable'] is True
    assert 'DATA.md' in body['error'] and 'unavailable' in body['error']


def test_search_without_movie_data_still_answers(client, monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader, 'MOVIES_FINAL_PATH', str(tmp_path / 'absent.csv'))
    monkeypatch.setattr(data_loader, '_movies_df', None)
    res = client.get('/api/search?q=love')
    assert res.status_code == 200
    body = res.get_json()
    assert body['movies'] == [] and body['genres'] == []


# ---- /api/recommend: validation ----

@pytest.mark.parametrize('body', [
    {'type': 'music'},                                   # missing mood
    {'mood': 'ecstatic', 'type': 'music'},               # unknown mood
    {'mood': 'happy'},                                   # missing type
    {'mood': 'happy', 'type': 'books'},                  # unknown type
    {'mood': 'happy', 'type': 'music', 'mode': 'x'},     # unknown mode
    {'mood': 'happy', 'type': 'music', 'limit': 'abc'},
    {'mood': 'happy', 'type': 'music', 'limit': 0},
    {'mood': 'happy', 'type': 'music', 'limit': -5},
    {'mood': 'happy', 'type': 'music', 'limit': flask_app.MAX_LIMIT + 1},
    {'mood': 'happy', 'type': 'music', 'limit': 2.5},
    {'mood': 'happy', 'type': 'music', 'limit': True},
    {'mood': ['happy'], 'type': 'music'},
])
def test_recommend_rejects_invalid_fields(client, body):
    res = recommend(client, body)
    assert res.status_code == 400
    assert 'error' in res.get_json()


@pytest.mark.parametrize('kwargs', [
    {'json': [1, 2]},
    {'json': 'happy'},
    {'data': 'not json', 'content_type': 'text/plain'},
    {'data': '{bad json', 'content_type': 'application/json'},
])
def test_recommend_rejects_non_object_bodies(client, kwargs):
    res = client.post('/api/recommend', **kwargs)
    assert res.status_code == 400
    assert 'error' in res.get_json()


def test_recommend_wrong_method_is_json_405(client):
    res = client.get('/api/recommend')
    assert res.status_code == 405
    assert 'error' in res.get_json()


def test_unknown_api_route_is_json_404(client):
    res = client.get('/api/does-not-exist')
    assert res.status_code == 404
    assert 'error' in res.get_json()


def test_internal_errors_surface_as_500_without_leaking_details(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError('secret internal detail')
    monkeypatch.setattr(flask_app, 'load_music_catalog', lambda: None)  # the failure under test is in the recommender
    monkeypatch.setattr(flask_app, 'recommend_music', boom)

    res = recommend(client, {'mood': 'happy', 'type': 'music'})
    assert res.status_code == 500
    assert res.get_json() == {'error': 'Internal server error'}
    assert b'secret internal detail' not in res.data


# ---- /api/detect-mood ----

def test_detect_mood_returns_a_known_mood(client):
    res = client.post('/api/detect-mood', json={'text': 'so stressed about my exam deadline'})
    assert res.status_code == 200
    body = res.get_json()
    assert body['mood'] in flask_app.VALID_MOODS
    assert 0.0 <= body['confidence'] <= 1.0


@pytest.mark.parametrize('body', [
    {},
    {'text': ''},
    {'text': '   '},
    {'text': 123},
    {'text': ['sad']},
    {'text': 'a' * (flask_app.MAX_TEXT_LENGTH + 1)},
])
def test_detect_mood_rejects_invalid_text(client, body):
    res = client.post('/api/detect-mood', json=body)
    assert res.status_code == 400
    assert 'error' in res.get_json()


# ---- /api/search ----

def test_search_short_query_returns_empty_groups(client):
    res = client.get('/api/search?q=a')
    assert res.status_code == 200
    assert res.get_json() == {'movies': [], 'songs': [], 'artists': [], 'genres': []}


def test_search_finds_matches(client):
    res = client.get('/api/search?q=love')
    assert res.status_code == 200
    body = res.get_json()
    assert set(body) == {'movies', 'songs', 'artists', 'genres'}


def test_search_rejects_overlong_query(client):
    res = client.get('/api/search?q=' + 'x' * (flask_app.MAX_QUERY_LENGTH + 1))
    assert res.status_code == 400


# ---- pages + config ----

@pytest.mark.parametrize('path', ['/', '/music', '/movies', '/dashboard'])
def test_pages_render(client, path):
    assert client.get(path).status_code == 200


def test_unknown_page_is_html_404(client):
    res = client.get('/no-such-page')
    assert res.status_code == 404
    assert res.mimetype == 'text/html'


@pytest.mark.parametrize('value,expected', [
    (None, False), ('0', False), ('false', False), ('', False),
    ('1', True), ('true', True), ('YES', True), ('on', True),
])
def test_debug_flag_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('FLASK_DEBUG', raising=False)
    else:
        monkeypatch.setenv('FLASK_DEBUG', value)
    assert flask_app._env_flag('FLASK_DEBUG') is expected
