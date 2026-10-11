"""
ANALYSIS WORKFLOW -- Phase 2L.3: analyse completed listener ratings from the
MoodCast listening pilot (P1-P4, Phase 2L.1) and, separately, the exploratory
supplementary calm pair (S1, Phase 2L.2). It implements the pre-written plans
(evaluation/listening_pilot_2l1/ANALYSIS_PLAN.md and
evaluation/listening_pilot_2l2_calm/ANALYSIS_PLAN_ADDENDUM.md) without changing
their outcomes, missing-data rules or tie handling; details the plans leave open
are fixed in IMPLEMENTATION_DETAILS and in
evaluation/listening_pilot_2l3/IMPLEMENTATION_NOTES.md.

Run from the project root, after the administrator has collected ratings:
    python analyze_music_listening_pilot.py [--ratings-dir DIR]

Reads   completed rating CSVs from --ratings-dir (default
        data/metadata/listening_results/, local and git-ignored):
          P1-P4: ratings_tracks.csv, ratings_lists.csv, ratings_pairs.csv
          S1:    calm_supplement_ratings_tracks.csv, _lists.csv, _pairs.csv
        and the administrator's local answer keys (the only source of the
        strategy mapping). Nothing else; no network access.
Writes  only when ratings exist: --out-json (aggregate JSON) and --out-md
        (readable Markdown). Neither contains track ids, titles, artists,
        links, listener ids or which list label belongs to which ranking.
        With no ratings it prints "No ratings available" and writes nothing.
It never infers, imputes or generates ratings.
"""
import argparse
import csv
import datetime
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
RATINGS_DIR = os.path.join(BASE, "data", "metadata", "listening_results")
OUT_JSON = os.path.join(BASE, "models", "music_listening_pilot_results.json")
OUT_MD = os.path.join(BASE, "evaluation", "listening_pilot_2l3", "listening_pilot_results.md")

STUDIES = {
    "main": {"title": "Primary pilot (P1-P4)", "exploratory_secondary": False,
             "files": {"tracks": "ratings_tracks.csv", "lists": "ratings_lists.csv", "pairs": "ratings_pairs.csv"},
             "key_dir": os.path.join(BASE, "data", "metadata", "listening_pilot_2l1", "answer_key"),
             "key_items": "answer_key_items.csv", "key_lists": "answer_key_lists.csv",
             "catalog_flag": ("catalog_speechiness_gt_0_66", "yesno")},
    "s1": {"title": "Supplementary calm pair (S1) -- EXPLORATORY, SECONDARY, speech-filtered", "exploratory_secondary": True,
           "files": {"tracks": "calm_supplement_ratings_tracks.csv", "lists": "calm_supplement_ratings_lists.csv",
                     "pairs": "calm_supplement_ratings_pairs.csv"},
           "key_dir": os.path.join(BASE, "data", "metadata", "listening_pilot_2l2_calm", "answer_key"),
           "key_items": "calm_supplement_answer_key_items.csv", "key_lists": "calm_supplement_answer_key_lists.csv",
           "catalog_flag": ("catalog_speechiness", "value")},
}
HEADERS = {
    "tracks": ["listener_id", "pair_id", "list_label", "item_id", "mood_fit_1_to_5", "enjoyment_1_to_5",
               "speech_like_yes_no", "notes"],
    "lists": ["listener_id", "pair_id", "list_label", "list_mood_fit_1_to_5", "perceived_variety_1_to_5",
              "list_enjoyment_1_to_5", "notes"],
    "pairs": ["listener_id", "pair_id", "request", "preferred_list_A_B_or_none", "notes"],
}
RATINGS = {
    "tracks": {"mood_fit_1_to_5": "scale", "enjoyment_1_to_5": "scale", "speech_like_yes_no": "yesno"},
    "lists": {"list_mood_fit_1_to_5": "scale", "perceived_variety_1_to_5": "scale", "list_enjoyment_1_to_5": "scale"},
    "pairs": {"preferred_list_A_B_or_none": "preference"},
}
RATINGS_FIELDS_ALL = [f for kind in RATINGS.values() for f in kind]
SPEECH_LIKE = 0.66
MIN_RATED_TRACKS = 6            # of the list's 8 (plan)
CLUSTER_MIN_LISTENERS = 5       # plan: resample listeners when there are at least 5
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 52000
ALPHA = 0.05

IMPLEMENTATION_DETAILS = [
    "Exact interval: Clopper-Pearson from beta quantiles (scipy.stats.beta), 95%, for the share of non-tie "
    "judgments preferring the current ranking; no interval when there are no non-tie judgments.",
    "Sign test: two-sided exact binomial test at 0.5 (scipy.stats.binomtest) on non-tie judgments; descriptive only.",
    "Tie sensitivity: (current wins + 0.5 x ties) / (wins + losses + ties), reported as a point estimate only.",
    "Bootstrap: 5,000 percentile resamples, seed 52000 (a fresh generator per outcome, so results do not depend on "
    "analysis order); listeners are resampled (all of a listener's judgments together) when at least 5 distinct "
    "listeners contribute to that outcome, otherwise judgments are resampled; the method used is reported per "
    "outcome. No interval is given for fewer than 2 paired differences.",
    "Track-level scores: a list's mean is used only when at least 6 of its 8 tracks have that rating (mood fit and "
    "enjoyment counted separately); the paired difference needs both lists of the pair.",
    "Speech-like content: per strategy (and per mood), listener 'yes' flags / flags given, next to the share of the "
    "same rated slots the catalog marks speechiness > 0.66, plus listener-by-catalog agreement counts.",
    "Missing responses: a listener is part of a study if any of their rows in that study's files has a rating; every "
    "expected judgment (pair, list or track) of such a listener without a value counts as missing.",
    "Input: ratings must be integers 1-5, speech flags yes/no, preferences A, B or none (case-insensitive); blank "
    "means missing. Rows with an empty listener id and no ratings are unused template rows and are ignored. A "
    "rating without a listener id, an unknown or inconsistent id, a duplicate judgment, a missing or unknown "
    "column, or a partial set of a study's three files is an error and nothing is written.",
    "S1 is analysed with the same rules but entirely separately; it is never pooled with P1-P4.",
]


class InputError(ValueError):
    pass


# ---- answer keys ----

def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def _strategy(name):
    if name.startswith("previous"):
        return "previous"
    if name.startswith("current"):
        return "current"
    raise InputError(f"Answer key has an unknown strategy name: {name!r}")


def load_key(study, key_dir=None):
    cfg = STUDIES[study]
    key_dir = key_dir or cfg["key_dir"]
    items_path = os.path.join(key_dir, cfg["key_items"])
    lists_path = os.path.join(key_dir, cfg["key_lists"])
    for p in (items_path, lists_path):
        if not os.path.exists(p):
            raise InputError(f"Answer key not found: {os.path.relpath(p, BASE)}")
    _, list_rows = _read_csv(lists_path)
    lists = {}
    for r in list_rows:
        key = (r["pair_id"], r["list_label"])
        if key in lists:
            raise InputError(f"Answer key lists: duplicate {key}")
        lists[key] = {"strategy": _strategy(r["strategy"]), "target": r["target"], "request": r["app_request"]}
    pairs = sorted({p for p, _ in lists})
    for p in pairs:
        if {lists.get((p, "A"), {}).get("strategy"), lists.get((p, "B"), {}).get("strategy")} != {"previous", "current"}:
            raise InputError(f"Answer key lists: pair {p} needs one previous and one current list labelled A/B")
    flag_col, flag_kind = cfg["catalog_flag"]
    _, item_rows = _read_csv(items_path)
    items = {}
    for r in item_rows:
        if r["item_id"] in items:
            raise InputError(f"Answer key items: duplicate {r['item_id']}")
        lst = lists.get((r["pair_id"], r["list_label"]))
        if lst is None or _strategy(r["strategy"]) != lst["strategy"] or r["target"] != lst["target"]:
            raise InputError(f"Answer key items: {r['item_id']} is inconsistent with the list key")
        catalog = (r[flag_col] == "yes") if flag_kind == "yesno" else float(r[flag_col]) > SPEECH_LIKE
        items[r["item_id"]] = {"pair_id": r["pair_id"], "list_label": r["list_label"],
                               "strategy": lst["strategy"], "target": lst["target"], "catalog_speech_like": catalog}
    for key in lists:
        if not any((i["pair_id"], i["list_label"]) == key for i in items.values()):
            raise InputError(f"Answer key items: no items for list {key}")
    return {"lists": lists, "items": items, "pairs": pairs}


# ---- ratings ----

def parse_value(kind, raw, where):
    v = (raw or "").strip()
    if v == "":
        return None
    if kind == "scale":
        if not re.fullmatch(r"[1-5]", v):
            raise InputError(f"{where}: rating must be an integer 1-5, got {raw!r}")
        return int(v)
    if kind == "yesno":
        if v.lower() not in ("yes", "no"):
            raise InputError(f"{where}: expected yes or no, got {raw!r}")
        return v.lower() == "yes"
    if v.lower() not in ("a", "b", "none"):
        raise InputError(f"{where}: preference must be A, B or none, got {raw!r}")
    return v.upper() if v.lower() != "none" else "none"


def read_ratings(path, kind, key, study):
    """Validated rows: dicts with listener, ids and parsed values. Raises InputError."""
    name = os.path.basename(path)
    fields, rows = _read_csv(path)
    fields = [f for f in fields if f is not None]
    missing = [c for c in HEADERS[kind] if c not in fields]
    unknown = [c for c in fields if c not in HEADERS[kind] and c.strip() != ""]
    if missing or unknown:
        raise InputError(f"{name}: missing columns {missing}, unknown columns {unknown}")
    out, seen = [], set()
    for line, r in enumerate(rows, start=2):
        where = f"{name} line {line}"
        if None in r and any((v or "").strip() for v in r[None]):
            raise InputError(f"{where}: more cells than columns")
        listener = (r.get("listener_id") or "").strip()
        values = {f: parse_value(k, r.get(f), f"{where} {f}") for f, k in RATINGS[kind].items()}
        if not listener:
            if any(v is not None for v in values.values()):
                raise InputError(f"{where}: ratings without a listener_id")
            continue                                         # unused template row
        pair, label = (r.get("pair_id") or "").strip(), (r.get("list_label") or "").strip()
        if pair not in key["pairs"]:
            raise InputError(f"{where}: unknown pair_id {pair!r} for this study")
        if kind == "tracks":
            item = (r.get("item_id") or "").strip()
            info = key["items"].get(item)
            if info is None or (info["pair_id"], info["list_label"]) != (pair, label):
                raise InputError(f"{where}: item_id {item!r} does not belong to {pair}/{label}")
            ident = (listener, item)
        elif kind == "lists":
            if (pair, label) not in key["lists"]:
                raise InputError(f"{where}: unknown list {pair}/{label!r}")
            ident = (listener, pair, label)
        else:
            request = (r.get("request") or "").strip()
            expected = {v["request"] for (p, _), v in key["lists"].items() if p == pair}
            if request and not any(request.startswith(e) for e in expected):
                raise InputError(f"{where}: request {request!r} does not match pair {pair}")
            ident = (listener, pair)
        if ident in seen:
            raise InputError(f"{where}: duplicate judgment for {ident[1:]} by the same listener")
        seen.add(ident)
        out.append({"listener": listener, "pair_id": pair, "list_label": label,
                    "item_id": (r.get("item_id") or "").strip(), **values})
    return out


def load_study_ratings(ratings_dir, study, key):
    files = STUDIES[study]["files"]
    present = {k: os.path.exists(os.path.join(ratings_dir, f)) for k, f in files.items()}
    if not any(present.values()):
        return None
    if not all(present.values()):
        raise InputError(f"{STUDIES[study]['title']}: incomplete file set, missing "
                         f"{[files[k] for k, p in present.items() if not p]}")
    return {k: read_ratings(os.path.join(ratings_dir, f), k, key, study) for k, f in files.items()}


def listeners_with_ratings(ratings):
    return sorted({r["listener"] for rows in ratings.values() for r in rows
                   if any(r[f] is not None for f in RATINGS_FIELDS_ALL if f in r)})


# ---- statistics ----

def clopper_pearson(x, n, alpha=ALPHA):
    if n == 0:
        return None
    lower = 0.0 if x == 0 else float(stats.beta.ppf(alpha / 2, x, n - x + 1))
    upper = 1.0 if x == n else float(stats.beta.ppf(1 - alpha / 2, x + 1, n - x))
    return [lower, upper]


def sign_test(x, n):
    return None if n == 0 else float(stats.binomtest(x, n, 0.5, alternative="two-sided").pvalue)


def bootstrap(diffs):
    """diffs: list of (listener, value). Percentile interval of the mean; method chosen per the plan."""
    values = np.array([v for _, v in diffs], dtype=float)
    out = {"n": int(len(values)), "mean": None, "median": None, "ci95": None, "resampled": None,
           "listeners": len({l for l, _ in diffs})}
    if len(values) == 0:
        return out
    out["mean"], out["median"] = round(float(values.mean()), 4), round(float(np.median(values)), 4)
    if len(values) < 2:
        out["resampled"] = "none (fewer than 2 paired differences)"
        return out
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    listeners = sorted({l for l, _ in diffs})
    if len(listeners) >= CLUSTER_MIN_LISTENERS:
        groups = [values[[i for i, (l, _) in enumerate(diffs) if l == who]] for who in listeners]
        means = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            pick = rng.randint(len(groups), size=len(groups))
            means.append(np.concatenate([groups[i] for i in pick]).mean())
        out["resampled"] = "listeners"
    else:
        means = values[rng.randint(len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))].mean(axis=1)
        out["resampled"] = "judgments (fewer than 5 listeners)"
    out["ci95"] = [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]
    return out


# ---- analysis ----

def preference_summary(judgments):
    """judgments: list of 'current' | 'previous' | 'tie' | 'missing'."""
    c, p = judgments.count("current"), judgments.count("previous")
    t, m = judgments.count("tie"), judgments.count("missing")
    n = c + p
    return {"current": c, "previous": p, "ties": t, "missing": m, "non_tie": n,
            "share_preferring_current": round(c / n, 4) if n else None,
            "exact_ci95": [round(v, 4) for v in clopper_pearson(c, n)] if n else None,
            "sign_test_p_two_sided_descriptive": round(sign_test(c, n), 4) if n else None,
            "sensitivity_ties_as_half": round((c + 0.5 * t) / (n + t), 4) if (n + t) else None}


def analyze_study(ratings, key):
    lists, items = key["lists"], key["items"]
    listeners = listeners_with_ratings(ratings)
    pair_target = {p: lists[(p, "A")]["target"] for p in key["pairs"]}
    label_of = {(p, lists[(p, lb)]["strategy"]): lb for (p, lb) in lists}

    # primary: preference
    prefs = {(r["listener"], r["pair_id"]): r["preferred_list_A_B_or_none"] for r in ratings["pairs"]}
    judged = {}
    for who in listeners:
        for p in key["pairs"]:
            v = prefs.get((who, p))
            judged[(who, p)] = ("missing" if v is None else "tie" if v == "none"
                                else lists[(p, v)]["strategy"])
    primary = {"overall": preference_summary(list(judged.values())),
               "by_mood": {pair_target[p]: preference_summary([v for (w, q), v in judged.items() if q == p])
                           for p in key["pairs"]}}

    # secondary: list-level paired differences
    list_vals = {(r["listener"], r["pair_id"], r["list_label"]): r for r in ratings["lists"]}
    secondary_lists, missing_lists = {}, {}
    for field in RATINGS["lists"]:
        diffs, by_mood, miss = [], {}, 0
        for who in listeners:
            for p in key["pairs"]:
                a = list_vals.get((who, p, label_of[(p, "current")]), {}).get(field)
                b = list_vals.get((who, p, label_of[(p, "previous")]), {}).get(field)
                miss += (a is None) + (b is None)
                if a is not None and b is not None:
                    diffs.append((who, a - b))
                    by_mood.setdefault(pair_target[p], []).append((who, a - b))
        secondary_lists[field] = {"overall": bootstrap(diffs),
                                  "by_mood": {m: bootstrap(d) for m, d in sorted(by_mood.items())}}
        missing_lists[field] = miss

    # secondary: track-level means per list (>= 6 of 8), then paired
    track_vals = {(r["listener"], r["item_id"]): r for r in ratings["tracks"]}
    list_items = {}
    for item, info in items.items():
        list_items.setdefault((info["pair_id"], info["list_label"]), []).append(item)
    secondary_tracks, excluded_lists = {}, {}
    for field in ("mood_fit_1_to_5", "enjoyment_1_to_5"):
        diffs, by_mood, excluded = [], {}, 0
        for who in listeners:
            for p in key["pairs"]:
                means = {}
                for strategy in ("current", "previous"):
                    vals = [track_vals.get((who, i), {}).get(field) for i in list_items[(p, label_of[(p, strategy)])]]
                    vals = [v for v in vals if v is not None]
                    means[strategy] = float(np.mean(vals)) if len(vals) >= MIN_RATED_TRACKS else None
                    excluded += means[strategy] is None
                if means["current"] is not None and means["previous"] is not None:
                    d = means["current"] - means["previous"]
                    diffs.append((who, d))
                    by_mood.setdefault(pair_target[p], []).append((who, d))
        secondary_tracks[field] = {"overall": bootstrap(diffs),
                                   "by_mood": {m: bootstrap(d) for m, d in sorted(by_mood.items())}}
        excluded_lists[field] = excluded

    # speech-like flags vs the catalog indicator
    speech = {}
    for strategy in ("previous", "current"):
        for mood in sorted(set(pair_target.values())) + ["all"]:
            flags = [(r["speech_like_yes_no"], items[r["item_id"]]["catalog_speech_like"]) for r in ratings["tracks"]
                     if r["speech_like_yes_no"] is not None and items[r["item_id"]]["strategy"] == strategy
                     and (mood == "all" or items[r["item_id"]]["target"] == mood)]
            n = len(flags)
            speech.setdefault(strategy, {})[mood] = {
                "flags_given": n,
                "listener_yes_share": round(sum(f for f, _ in flags) / n, 4) if n else None,
                "catalog_speech_like_share_of_same_slots": round(sum(c for _, c in flags) / n, 4) if n else None,
                "agreement": {"listener_yes_catalog_yes": sum(f and c for f, c in flags),
                              "listener_yes_catalog_no": sum(f and not c for f, c in flags),
                              "listener_no_catalog_yes": sum((not f) and c for f, c in flags),
                              "listener_no_catalog_no": sum((not f) and (not c) for f, c in flags)}}

    expected_tracks = len(listeners) * len(items)
    track_missing = {f: expected_tracks - sum(1 for r in ratings["tracks"] if r[f] is not None and r["listener"] in listeners)
                     for f in RATINGS["tracks"]}
    return {
        "listeners": len(listeners),
        "primary_preference": primary,
        "secondary_list_level_current_minus_previous": secondary_lists,
        "secondary_track_level_current_minus_previous": secondary_tracks,
        "speech_like_flags": speech,
        "missing": {"pair_preferences": primary["overall"]["missing"], "list_ratings": missing_lists,
                    "track_ratings": track_missing,
                    "lists_excluded_by_min_6_of_8_rule": excluded_lists},
    }


# ---- reports ----

def _ci(ci):
    return "n/a" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def markdown(report):
    out = ["# MoodCast listening pilot -- results", "",
           f"Generated {report['generated_on']}. Aggregates only: no track, listener or label-to-ranking "
           "information. Plans: `evaluation/listening_pilot_2l1/ANALYSIS_PLAN.md`, "
           "`evaluation/listening_pilot_2l2_calm/ANALYSIS_PLAN_ADDENDUM.md`.", ""]
    for study, res in report["studies"].items():
        cfg = STUDIES[study]
        out += [f"## {cfg['title']}", ""]
        if cfg["exploratory_secondary"]:
            out += ["> **Exploratory secondary comparison.** Speech-like tracks (catalog speechiness > 0.66) were "
                    "removed before sampling; it does not replace the original calm pair and is never pooled with "
                    "P1-P4.", ""]
        if res.get("status") == "no ratings":
            out += ["No ratings available.", ""]
            continue
        pp = res["primary_preference"]
        out += [f"Listeners: {res['listeners']}", "", "### Preference (current vs previous ranking)", "",
                "| | current | previous | ties | missing | share current | exact 95% CI | sign test p | ties as half |",
                "|---|---|---|---|---|---|---|---|---|"]
        for name, s in [("overall", pp["overall"])] + list(pp["by_mood"].items()):
            out.append(f"| {name} | {s['current']} | {s['previous']} | {s['ties']} | {s['missing']} | "
                       f"{s['share_preferring_current'] if s['share_preferring_current'] is not None else 'n/a'} | "
                       f"{_ci(s['exact_ci95'])} | {s['sign_test_p_two_sided_descriptive'] if s['sign_test_p_two_sided_descriptive'] is not None else 'n/a'} | "
                       f"{s['sensitivity_ties_as_half'] if s['sensitivity_ties_as_half'] is not None else 'n/a'} |")
        out += ["", "### Paired differences (current minus previous)", "",
                "| outcome | n | mean | median | 95% CI | resampled |", "|---|---|---|---|---|---|"]
        for section in ("secondary_list_level_current_minus_previous", "secondary_track_level_current_minus_previous"):
            for field, s in res[section].items():
                o = s["overall"]
                tag = "list" if "list_level" in section else "track mean"
                out.append(f"| {tag}: {field} | {o['n']} | {o['mean']} | {o['median']} | {_ci(o['ci95'])} | "
                           f"{o['resampled']} |")
        out += ["", "### Speech-like flags", "", "| strategy | flags | listener yes | catalog speech-like (same slots) |",
                "|---|---|---|---|"]
        for strategy, moods in res["speech_like_flags"].items():
            s = moods["all"]
            out.append(f"| {strategy} | {s['flags_given']} | {s['listener_yes_share']} | "
                       f"{s['catalog_speech_like_share_of_same_slots']} |")
        out += ["", f"Missing: {json.dumps(res['missing'])}", ""]
    out += ["## Implementation details", ""] + [f"- {d}" for d in IMPLEMENTATION_DETAILS] + [
        "", "## Interpretation limits", "",
        "- Exploratory pilot: one list pair per mood, 8 tracks per list; no population-level conclusions.",
        "- Listeners can see song identity on Spotify; repeated judgments by one listener are not independent.", ""]
    return "\n".join(out)


def run(ratings_dir, out_json, out_md, key_dirs=None):
    """Returns the report dict, or None when no ratings exist (nothing is written)."""
    loaded = {}
    for study in STUDIES:
        key = load_key(study, (key_dirs or {}).get(study))
        ratings = load_study_ratings(ratings_dir, study, key)
        loaded[study] = (key, ratings)
    available = {s: r is not None and listeners_with_ratings(r) for s, (_, r) in loaded.items()}
    if not any(available.values()):
        print("No ratings available -- no outcome estimates produced.")
        return None
    report = {"generated_on": datetime.date.today().isoformat(), "ratings_simulated": False,
              "implementation_details": IMPLEMENTATION_DETAILS, "studies": {}}
    for study, (key, ratings) in loaded.items():
        res = {"exploratory_secondary": STUDIES[study]["exploratory_secondary"]}
        if available[study]:
            res.update(analyze_study(ratings, key))
        else:
            res["status"] = "no ratings"
        report["studies"][study] = res
    for path, text in ((out_json, json.dumps(report, indent=2) + "\n"), (out_md, markdown(report))):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(f"Wrote {os.path.relpath(out_json, BASE)} and {os.path.relpath(out_md, BASE)}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyse completed listening-pilot ratings")
    parser.add_argument("--ratings-dir", default=RATINGS_DIR)
    parser.add_argument("--out-json", default=OUT_JSON)
    parser.add_argument("--out-md", default=OUT_MD)
    args = parser.parse_args()
    try:
        run(args.ratings_dir, args.out_json, args.out_md)
    except InputError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
