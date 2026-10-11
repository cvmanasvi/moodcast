"""Tests for the "Describe your day" text mood detector (POST /api/detect-mood).

Covers the reported bug ("i love my work" was predicted stressed at 89%
because the template training data used "work" almost only in stressed
sentences and never used "love"), contrasting positive/stressed phrases, the
label mapping to the six app moods and the recommender, confidence handling
(low-confidence answers come back as "uncertain" with their real
probability), the keyword fallback when the model file is missing, and that
evaluation sentences never leak into training. The probe phrases are a small
hand-written smoke test (text_mood_probe.py), not a benchmark: the floors
below catch regressions, they do not certify accuracy.
"""
import os
import re

import pandas as pd
import pytest

import flask_app
from src import ml_inference as mi
from src.config import BASE_DIR, MOODS
from text_mood_probe import BLIND, PROBE
from text_vocab import TEXT_HOLDOUT, TEXT_SHARED_CONTEXTS
from tests.local_data import requires_music_catalog

DATASET = os.path.join(BASE_DIR, "data", "text_mood_dataset.csv")


@pytest.fixture
def client():
    return flask_app.app.test_client()


@pytest.fixture(autouse=True)
def real_model():
    """Each test sees the trained model (some tests swap it out)."""
    mi._loaded = False
    mi._load()
    yield
    mi._loaded = False


def detect(client, text):
    r = client.post("/api/detect-mood", json={"text": text})
    assert r.status_code == 200
    return r.get_json()


# ---------------------------------------------------------------- the reported bug and contrasts

@pytest.mark.parametrize("text", ["i love my work", "I love my work.", "I really love my work"])
def test_loving_work_is_not_detected_as_stress(client, text):
    body = detect(client, text)
    assert body["mood"] == "happy" and body["source"] == "ml_classifier" and body["uncertain"] is False


@pytest.mark.parametrize("text,mood", [
    ("I love my work.", "happy"),
    ("I am proud of what I accomplished today.", "happy"),
    ("I feel exhausted and overwhelmed.", "stressed"),
    ("I am worried about my deadlines.", "stressed"),
    ("I feel lonely and sad.", "sad"),
    ("I am furious about what happened.", "angry"),
    ("I feel peaceful and relaxed.", "chill"),
    ("I am excited and full of energy.", "energetic"),
])
def test_contrasting_examples(client, text, mood):
    body = detect(client, text)
    assert body["mood"] == mood, body


def test_work_alone_is_no_longer_a_confident_stress_signal(client):
    body = detect(client, "Work.")
    assert body["confidence"] < 0.5                     # was 0.89 before the training-data fix


def test_training_data_uses_topics_in_every_mood():
    df = pd.read_csv(DATASET)
    for word in ["work", "job", "exam", "deadline", "weekend"]:
        moods = set(df[df.text.str.lower().str.contains(rf"\b{word}\b")].mood)
        assert moods == set(MOODS), (word, moods)
    assert (df.text.str.lower().str.contains(r"\blove\b")).sum() > 0
    assert len(TEXT_SHARED_CONTEXTS) >= 10


# ---------------------------------------------------------------- labels and the recommender

def test_model_labels_are_exactly_the_six_app_moods_in_training_order():
    art = mi._text_art
    # the trainer indexes labels as sorted(dataset moods); the artifact must carry that same order
    assert art["moods"] == sorted(pd.read_csv(DATASET).mood.unique()) == sorted(MOODS)
    assert art["model"].n_classes == len(art["moods"]) == 6
    # (that probability column i really is mood i is shown by the six "reachable" cases below,
    # which would fail under any shift of the mapping)


EVERY_MOOD = pytest.mark.parametrize("text,mood", [
    ("so happy and joyful today", "happy"), ("heartbroken and crying", "sad"), ("furious and enraged", "angry"),
    ("calm and relaxed", "chill"), ("pumped and full of energy", "energetic"), ("stressed and overwhelmed", "stressed"),
])


@EVERY_MOOD
def test_every_mood_is_reachable(client, text, mood):
    body = detect(client, text)
    assert body["mood"] == mood and body["mood"] in flask_app.VALID_MOODS


@requires_music_catalog
@EVERY_MOOD
def test_every_mood_is_reachable_and_feeds_the_recommender(client, text, mood):
    body = detect(client, text)
    assert body["mood"] == mood and body["mood"] in flask_app.VALID_MOODS
    r = client.post("/api/recommend", json={"mood": body["mood"], "type": "music", "limit": 5})
    assert r.status_code == 200 and len(r.get_json()["results"]) == 5


# ---------------------------------------------------------------- confidence and uncertainty

def test_threshold_is_above_the_six_class_chance_level():
    assert mi.CONFIDENCE_THRESHOLD > 1 / 6


def test_low_confidence_is_reported_as_uncertain_with_its_real_probability(client):
    body = detect(client, "It's Monday.")
    mood, conf, source = mi.predict_text_mood("It's Monday.")
    assert conf < mi.CONFIDENCE_THRESHOLD and source == "uncertain"
    assert body == {"mood": mood, "confidence": conf, "source": "uncertain", "uncertain": True}
    assert body["mood"] in flask_app.VALID_MOODS and 0 < body["confidence"] < 1


def test_text_with_no_known_word_is_uncertain():
    mood, conf, source = mi.predict_text_mood("zxqv blorp")
    assert source == "uncertain" and mood in MOODS


def test_threshold_decides_the_source():
    text = "I am worried about my deadlines."
    mood, conf, _ = mi.predict_text_mood(text)
    assert mi.predict_text_mood(text, confidence_threshold=conf - 1e-9)[2] == "ml_classifier"
    assert mi.predict_text_mood(text, confidence_threshold=conf + 1e-9) == (mood, conf, "uncertain")


def test_confidence_is_a_probability_and_flags_agree(client):
    for text, _, _ in PROBE[:20]:
        body = detect(client, text)
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["uncertain"] == (body["source"] != "ml_classifier")
        assert body["uncertain"] == (body["confidence"] < mi.CONFIDENCE_THRESHOLD or body["source"] == "uncertain")


# ---------------------------------------------------------------- model missing: keyword fallback

def test_keyword_fallback_only_when_the_model_is_missing(client, monkeypatch):
    monkeypatch.setattr(mi, "_text_art", None)
    monkeypatch.setattr(mi, "_loaded", True)
    assert mi.predict_text_mood("so stressed about the deadline") == ("stressed", 0.0, "keyword_fallback")
    assert mi.predict_text_mood("I cried all day, so lonely") == ("sad", 0.0, "keyword_fallback")
    assert mi.predict_text_mood("I had a sandwich") == ("happy", 0.0, "uncertain")     # placeholder, labelled
    body = detect(client, "I had a sandwich")
    assert body["uncertain"] is True and body["source"] == "uncertain"


def test_a_confident_model_answer_never_falls_back_to_keywords():
    # "relaxed" is a keyword for chill, but the model's confident answer wins
    mood, conf, source = mi.predict_text_mood("I feel exhausted and overwhelmed, I can't get relaxed")
    assert (mood, source) == ("stressed", "ml_classifier") and conf >= mi.CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------- evaluation hygiene

def _norm(t):
    return " ".join(re.sub(r"[^\w\s]", "", t.casefold()).split())


def test_evaluation_sentences_are_never_in_the_training_data():
    train = set(pd.read_csv(DATASET).text.map(_norm))
    for text in [t for t, _ in TEXT_HOLDOUT] + [t for t, _, _ in PROBE] + [t for t, _, _ in BLIND]:
        assert _norm(text) not in train, text


def test_probe_sets_have_no_confident_mistakes_and_a_floor_on_clear_phrases():
    import evaluate_text_mood as ev
    for phrases, floor in ((PROBE, 0.85), (BLIND, 0.75)):
        rows = ev.run(phrases)
        summary = ev.summarize(rows)
        assert summary["wrong_at_confidence_0.5_or_more"] == 0
        assert summary["clear_top_guess_accuracy"] >= floor
        assert summary["neutral"]["reported_uncertain"] >= summary["neutral"]["n"] - 1


# ---------------------------------------------------------------- frontend shows uncertainty

def test_pages_present_uncertain_answers_as_guesses():
    with open(os.path.join(BASE_DIR, "templates", "index.html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(BASE_DIR, "static", "js", "assistant.js"), encoding="utf-8") as f:
        js = f.read()
    assert 'id="analysisNote"' in html and "data.uncertain === true" in html and "Not sure — maybe" in html
    assert "data.uncertain === true" in js and "low confidence" in js
