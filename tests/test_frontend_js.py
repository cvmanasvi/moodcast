"""Frontend JavaScript checks, run under pytest so one command covers the
project: `node --check` on every static script, the Node unit tests for the
shared track-identity helper (tests/js, Node's built-in test runner), and
static checks that the music pages load and use that helper."""
import glob
import os
import re
import shutil
import subprocess

import pytest

from src.config import BASE_DIR

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
JS_FILES = sorted(glob.glob(os.path.join(BASE_DIR, "static", "js", "*.js")))


def _read(*parts):
    with open(os.path.join(BASE_DIR, *parts), encoding="utf-8") as f:
        return f.read()


@requires_node
@pytest.mark.parametrize("path", JS_FILES, ids=[os.path.basename(p) for p in JS_FILES])
def test_js_syntax(path):
    result = subprocess.run([NODE, "--check", path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


JS_TESTS = sorted(glob.glob(os.path.join(BASE_DIR, "tests", "js", "*.test.js")))


@requires_node
def test_track_identity_unit_tests():
    assert JS_TESTS, "no JS test files found"
    result = subprocess.run([NODE, "--test", *JS_TESTS], cwd=BASE_DIR, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    passed = re.search(r"# pass (\d+)", result.stdout)
    assert passed and int(passed.group(1)) > 0, result.stdout
    assert re.search(r"# fail 0\b", result.stdout), result.stdout


@pytest.mark.parametrize("template", ["music.html", "index.html", "movies.html", "dashboard.html"])
def test_identity_helper_loads_before_its_consumers(template):
    html = _read("templates", template)
    helper = html.find("js/track-identity.js")
    assert helper != -1
    for consumer in ["js/search.js", "js/music.js"]:
        if consumer in html:
            assert helper < html.find(consumer), f"{consumer} loads before track-identity.js"


def test_music_and_search_have_no_display_field_identity():
    """No title::artist key is built outside the helper's legacy fallback."""
    for name in ["music.js", "search.js"]:
        source = _read("static", "js", name)
        assert "'::'" not in source and '"::"' not in source, name
        assert not re.search(r"t\.title\s*\+.*t\.artist", source), name
        assert ".split(','" not in source and ".split(', '" not in source, name
