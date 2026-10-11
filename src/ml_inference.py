"""
Runtime ML inference layer for the mood recommender.

Loads the trained text mood classifier once per process and exposes a small,
app-facing function used by src/mood_utils.py. Falls back gracefully to a
keyword heuristic if the model file is missing, so the app never hard-crashes
if `train_all_models.py` hasn't been run yet. (Music ranking needs no model
here: the 278k catalog carries precomputed classifier probabilities.)
"""
from __future__ import annotations
import os
import pickle
import numpy as np

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_text_art = None
_loaded = False


def _load():
    global _text_art, _loaded
    if _loaded:
        return
    _loaded = True
    try:
        with open(os.path.join(_MODEL_DIR, "text_mood_classifier.pkl"), "rb") as f:
            _text_art = pickle.load(f)
    except FileNotFoundError:
        _text_art = None


# Below this probability the classifier's answer is reported as "uncertain"
# (still its best guess, with its real probability). Chosen on 2026-10-10 from
# the development data (text_mood_probe.PROBE clear phrases + TEXT_HOLDOUT, 78
# sentences): answers at >= 0.30 were right 97% of the time, answers below it
# 70%. Six moods make 1/6 ~ 0.17 the "no idea" level, so the previous 0.20
# flagged almost nothing. A small, hand-written sample: a heuristic, not a
# calibration guarantee (see evaluate_text_mood.py).
CONFIDENCE_THRESHOLD = 0.30


# ---- fallback keyword rule, used only if the trained model file is missing ----
def _keyword_fallback(text: str):
    """A mood from obvious keywords, or None when no keyword matches."""
    if not text:
        return None
    t = text.lower()
    if any(w in t for w in ["sad", "down", "upset", "depressed", "bad day", "cry", "lonely"]):
        return "sad"
    if any(w in t for w in ["angry", "mad", "frustrated", "furious", "annoyed"]):
        return "angry"
    if any(w in t for w in ["stressed", "anxious", "overwhelmed", "deadline", "panic"]):
        return "stressed"
    if any(w in t for w in ["chill", "relaxed", "calm", "peaceful", "cozy"]):
        return "chill"
    if any(w in t for w in ["excited", "pumped", "hype", "energetic", "workout"]):
        return "energetic"
    return None


def predict_text_mood(text: str, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    """
    Returns (mood: str, confidence: float, source: str).

    source is
      "ml_classifier"    -- the TF-IDF + softmax-regression classifier's answer,
                            with probability >= confidence_threshold;
      "uncertain"        -- the classifier's best guess and its real probability,
                            when that is below the threshold or the text has no
                            word the model knows (callers should present it as a
                            tentative guess, not a detection);
      "keyword_fallback" -- the model file is missing and an obvious keyword
                            matched (confidence 0.0: no model probability).
    With no model and no keyword the result is ("happy", 0.0, "uncertain"):
    a placeholder, labelled as such.
    """
    _load()
    if not text:
        return "happy", 0.0, "default"

    if _text_art is not None:
        vec, model, moods = _text_art["vectorizer"], _text_art["model"], _text_art["moods"]
        X = vec.transform([text])
        proba = model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        conf = float(proba[idx])
        known = bool(np.any(X))      # an all-zero vector means no known word: the output is just the bias
        source = "ml_classifier" if known and conf >= confidence_threshold else "uncertain"
        return moods[idx], conf, source

    mood = _keyword_fallback(text)
    return (mood, 0.0, "keyword_fallback") if mood else ("happy", 0.0, "uncertain")
