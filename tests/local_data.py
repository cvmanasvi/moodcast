"""Skip markers for tests that need files the repository does not distribute.

The 278k music catalog and classifier are built locally (python
train_music_278k.py), the movie dataset is not distributed at all, and a few
checks need a Git checkout or a specific commit. Each marker skips only when
its own requirement is missing, so with the files present every test runs.
See DATA.md for how to provide each file.
"""
import os
import shutil
import subprocess

import pytest

from src.config import BASE_DIR, MODELS_DIR, MOVIES_FINAL_PATH, MUSIC_278K_SOURCE_PATH, MUSIC_CATALOG_278K_PATH

MUSIC_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "music_mood_classifier_278k.pkl")


def _git_ok(*args):
    git = shutil.which("git")
    if git is None:
        return False
    result = subprocess.run([git, *args], cwd=BASE_DIR, capture_output=True, text=True)
    return result.returncode == 0


def git_has_commit(rev):
    return _git_ok("cat-file", "-e", f"{rev}^{{commit}}")


requires_music_catalog = pytest.mark.skipif(
    not os.path.exists(MUSIC_CATALOG_278K_PATH),
    reason="local-only: data/music_catalog_278k.csv not built (python train_music_278k.py; see DATA.md)")
requires_music_classifier = pytest.mark.skipif(
    not os.path.exists(MUSIC_CLASSIFIER_PATH),
    reason="local-only: models/music_mood_classifier_278k.pkl not built (python train_music_278k.py; see DATA.md)")
requires_music_source = pytest.mark.skipif(
    not os.path.exists(MUSIC_278K_SOURCE_PATH),
    reason="local-only: data/music_dataset2/278k_labelled_uri.csv not downloaded (see DATA.md)")
requires_movie_data = pytest.mark.skipif(
    not os.path.exists(MOVIES_FINAL_PATH),
    reason="local-only: data/movies_final.csv is not distributed (see DATA.md)")
requires_git_checkout = pytest.mark.skipif(
    not _git_ok("rev-parse", "--is-inside-work-tree"),
    reason="needs a Git checkout of the repository")

# prepare_music_listening_pilot*.py rebuild the pre-Phase-2K ranking from
# src/recommend.py at this commit, read from Git history.
PILOT_BASELINE_COMMIT = "9d34ac0dc1af484d5c3f9b50449535e98b6ba409"
requires_pilot_baseline = pytest.mark.skipif(
    not git_has_commit(PILOT_BASELINE_COMMIT),
    reason=f"needs Git history containing the listening-pilot baseline commit {PILOT_BASELINE_COMMIT[:7]}")


def skip_if_missing(*markers):
    """For fixtures (marks don't apply to them): skip the requesting test if any
    of these markers' requirements is missing."""
    for marker in markers:
        if marker.mark.args[0]:
            pytest.skip(marker.mark.kwargs["reason"])
