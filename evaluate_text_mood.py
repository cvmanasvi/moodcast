"""
Audit the "Describe your day" text mood detector on the fixed probe phrases
in text_mood_probe.py, through the exact production path the website uses
(src.ml_inference.predict_text_mood, called by POST /api/detect-mood).

Run from the project root:
    python evaluate_text_mood.py                # print the table
    python evaluate_text_mood.py --json PATH    # also write the results as JSON

For each phrase it records the mood returned, the confidence, the source
("ml_classifier", "uncertain" or "keyword_fallback") and whether the mood is
one of the acceptable ones; the summary separates confident answers from
ones reported as uncertain. `--blind` runs the second set, which was written
before retraining and not used for any decision. Both sets are small and
hand-written: a smoke test of obvious behaviour, not a benchmark (see
text_mood_probe.py).
"""
import argparse
import json
import sys

from src import ml_inference
from text_mood_probe import PROBE


def run(phrases=PROBE):
    ml_inference._load()
    rows = []
    for text, acceptable, kind in phrases:
        mood, confidence, source = ml_inference.predict_text_mood(text)
        rows.append({"text": text, "kind": kind, "acceptable": sorted(acceptable), "mood": mood,
                     "confidence": round(float(confidence), 4), "source": source,
                     "uncertain": source != "ml_classifier",
                     # neutral phrases have no right mood: the honest answer is "uncertain"
                     "guess_ok": (mood in acceptable) if kind != "neutral" else None})
    return rows


def _outcomes(rows):
    """Answers split by whether they were confident and whether the mood is acceptable."""
    return {
        "n": len(rows),
        "confident_correct": sum(not r["uncertain"] and r["guess_ok"] for r in rows),
        "confident_wrong": sum(not r["uncertain"] and not r["guess_ok"] for r in rows),
        "uncertain_guess_correct": sum(r["uncertain"] and r["guess_ok"] for r in rows),
        "uncertain_guess_wrong": sum(r["uncertain"] and not r["guess_ok"] for r in rows),
    }


def summarize(rows):
    clear = [r for r in rows if r["kind"] == "clear"]
    neutral = [r for r in rows if r["kind"] == "neutral"]
    return {
        "clear": _outcomes(clear),
        "clear_top_guess_accuracy": round(sum(r["guess_ok"] for r in clear) / max(len(clear), 1), 4),
        "clear_by_mood": {m: _outcomes([r for r in clear if r["acceptable"] == [m]])
                          for m in ("happy", "sad", "angry", "chill", "energetic", "stressed")},
        "mixed": _outcomes([r for r in rows if r["kind"] == "mixed"]),
        "neutral": {"n": len(neutral), "reported_uncertain": sum(r["uncertain"] for r in neutral)},
        "wrong_at_confidence_0.5_or_more": sum(r["guess_ok"] is False and r["confidence"] >= 0.5 for r in rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", metavar="PATH")
    parser.add_argument("--blind", action="store_true", help="use text_mood_probe.BLIND instead of PROBE")
    args = parser.parse_args(argv)
    from text_mood_probe import BLIND
    rows = run(BLIND if args.blind else PROBE)
    for r in rows:
        mark = "-  " if r["guess_ok"] is None else ("ok " if r["guess_ok"] else "BAD")
        print(f"{mark} {r['kind']:7s} {r['mood']:9s} {r['confidence']:.2f} {r['source']:16s} "
              f"{r['text']}  (expected {', '.join(r['acceptable']) or 'uncertain'})")
    s = summarize(rows)
    print(json.dumps(s, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"summary": s, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
