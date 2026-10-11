from .ml_inference import predict_text_mood


def simple_text_to_mood(text: str) -> str:
    """
    Detects mood from free-text using a TF-IDF + softmax-regression classifier
    trained on ~1500 mostly template-generated examples across the 6 app
    moods (see models/text_mood_classifier_eval.json for held-out results and
    evaluate_text_mood.py for an audit; it is not validated on real user
    text). Low-confidence answers come back marked "uncertain"; a keyword
    heuristic is used only if the trained model file is missing.

    Kept as `simple_text_to_mood` (same name/signature as before) so the
    Streamlit and Flask apps don't need to change how they call this.
    """
    mood, confidence, source = predict_text_mood(text)
    return mood


def text_to_mood_with_confidence(text: str):
    """Same as simple_text_to_mood but also returns (mood, confidence, source)
    for callers (e.g. the Flask API) that want to show how sure the model is."""
    return predict_text_mood(text)
