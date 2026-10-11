"""Tests for the listening-pilot analysis workflow (analyze_music_listening_pilot.py).

All rating values below are SYNTHETIC TEST FIXTURES written to temporary directories to check
validation and arithmetic. They are not listener judgments and are never written to the real
ratings directory."""
import csv
import json
import os
import re

import numpy as np
import pytest

import analyze_music_listening_pilot as al

FAKE_TARGETS = {"P1": "sad", "P2": "happy", "P3": "calm", "P4": "energetic"}


def _real_outputs_state():
    """Bytes of every file in the real ratings directory and of the real result reports (None if absent)."""
    state = {}
    if os.path.isdir(al.RATINGS_DIR):
        for root, _, files in os.walk(al.RATINGS_DIR):
            for name in files:
                path = os.path.join(root, name)
                with open(path, "rb") as f:
                    state[path] = f.read()
    for path in (al.OUT_JSON, al.OUT_MD):
        if os.path.exists(path):
            with open(path, "rb") as f:
                state[path] = f.read()
    return state


REAL_OUTPUTS_BEFORE = _real_outputs_state()                # taken at collection, before any test runs


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _fake_id(n):
    return f"X{n:021d}"                                   # 22 chars, like a track id, but fake


def make_keys(base):
    """Fake answer keys: in each pair, A is previous for P1/P3/S1 and current for P2/P4."""
    keys = {}
    for study, pairs, items_name, lists_name, flag_col in (
            ("main", FAKE_TARGETS, "answer_key_items.csv", "answer_key_lists.csv", "catalog_speechiness_gt_0_66"),
            ("s1", {"S1": "calm"}, "calm_supplement_answer_key_items.csv", "calm_supplement_answer_key_lists.csv",
             "catalog_speechiness")):
        d = base / f"key_{study}"
        list_rows, item_rows, n = [], [], 0
        for p, target in pairs.items():
            prev_label = "A" if p in ("P1", "P3", "S1") else "B"
            request = f"Match my mood: {'chill' if target == 'calm' else target}"
            for label in ("A", "B"):
                name = "previous ranking (x)" if label == prev_label else "current feature-cell ranking (x)"
                list_rows.append([p, label, name, target, request, 1, 30])
                for k in range(1, 9):
                    n += 1
                    speech = ("yes" if (label == prev_label and k <= 2) else "no") if study == "main" \
                        else ("0.9000" if (label == prev_label and k == 1) else "0.1000")
                    item_rows.append([f"{p}-{label}{k:02d}", p, label, k, name, target, request, 1, k,
                                      _fake_id(n), f"https://open.spotify.com/track/{_fake_id(n)}", speech])
        _write(d / lists_name, ["pair_id", "list_label", "strategy", "target", "app_request", "list_seed",
                                "sampled_from_list_of"], list_rows)
        _write(d / items_name, ["item_id", "pair_id", "list_label", "display_order", "strategy", "target",
                                "app_request", "list_seed", "position_in_list_of_30", "track_id", "listen_url",
                                flag_col], item_rows)
        keys[study] = str(d)
    return keys


def ratings_files(d, study="main", tracks=(), lists=(), pairs=(), template=True):
    pre = "" if study == "main" else "calm_supplement_"
    pids = list(FAKE_TARGETS) if study == "main" else ["S1"]
    t_rows = [["", p, lb, f"{p}-{lb}{k:02d}", "", "", "", ""] for p in pids for lb in "AB" for k in range(1, 9)] \
        if template else []
    l_rows = [["", p, lb, "", "", "", ""] for p in pids for lb in "AB"] if template else []
    p_rows = [["", p, "", "", ""] for p in pids] if template else []
    _write(d / f"{pre}ratings_tracks.csv", al.HEADERS["tracks"], t_rows + [list(r) for r in tracks])
    _write(d / f"{pre}ratings_lists.csv", al.HEADERS["lists"], l_rows + [list(r) for r in lists])
    _write(d / f"{pre}ratings_pairs.csv", al.HEADERS["pairs"], p_rows + [list(r) for r in pairs])


def full_listener(who, prefs, list_scores=(4, 3), track_score=4, study="main"):
    """Rows for one listener: prefs {pair: 'A'|'B'|'none'|''}; list_scores (A, B) for every list field."""
    pids = list(FAKE_TARGETS) if study == "main" else ["S1"]
    tracks = [[who, p, lb, f"{p}-{lb}{k:02d}", track_score, track_score, "no", ""]
              for p in pids for lb in "AB" for k in range(1, 9)]
    lists = [[who, p, lb, *([list_scores[0] if lb == "A" else list_scores[1]] * 3), ""] for p in pids for lb in "AB"]
    pairs = [[who, p, "", prefs.get(p, ""), ""] for p in pids]
    return tracks, lists, pairs


def run(tmp_path, keys, out="out"):
    return al.run(str(tmp_path / "ratings"), str(tmp_path / out / "r.json"), str(tmp_path / out / "r.md"), keys)


@pytest.fixture
def keys(tmp_path):
    return make_keys(tmp_path)


# ---- statistics ----

def test_exact_interval_matches_known_clopper_pearson_values():
    assert al.clopper_pearson(7, 10) == pytest.approx([0.3475, 0.9333], abs=1e-4)
    assert al.clopper_pearson(0, 5) == pytest.approx([0.0, 0.5218], abs=1e-4)
    assert al.clopper_pearson(5, 5) == pytest.approx([0.4782, 1.0], abs=1e-4)
    assert al.clopper_pearson(0, 0) is None
    assert al.sign_test(9, 10) == pytest.approx(0.02148, abs=1e-4)


def test_preference_summary_ties_missing_and_sensitivity():
    s = al.preference_summary(["current"] * 3 + ["previous"] + ["tie"] * 2 + ["missing"])
    assert (s["current"], s["previous"], s["ties"], s["missing"], s["non_tie"]) == (3, 1, 2, 1, 4)
    assert s["share_preferring_current"] == 0.75
    assert s["sensitivity_ties_as_half"] == round((3 + 1) / 6, 4)
    none = al.preference_summary(["tie", "missing"])
    assert none["share_preferring_current"] is None and none["exact_ci95"] is None


def test_bootstrap_is_deterministic_and_follows_the_listener_rule():
    few = [("a", 1.0), ("a", 2.0), ("b", 0.0), ("b", 3.0)]
    one, two = al.bootstrap(few), al.bootstrap(few)
    assert one == two and one["mean"] == 1.5 and one["median"] == 1.5
    assert one["resampled"].startswith("judgments") and one["ci95"][0] <= 1.5 <= one["ci95"][1]
    many = [(f"l{i}", float(i % 3)) for i in range(10)]
    assert al.bootstrap(many)["resampled"] == "listeners"
    single = al.bootstrap([("a", 2.0)])
    assert single["ci95"] is None and single["mean"] == 2.0
    assert al.bootstrap([])["n"] == 0


# ---- end-to-end on valid synthetic input ----

def test_valid_input_is_analysed_with_the_answer_key_mapping(tmp_path, keys):
    # A is previous in P1/P3 and current in P2/P4 (fake key)
    t1, l1, p1 = full_listener("L-one", {"P1": "B", "P2": "A", "P3": "A", "P4": "none"}, list_scores=(2, 5))
    t2, l2, p2 = full_listener("L-two", {"P1": "B", "P2": "B", "P3": ""}, list_scores=(2, 5))
    ratings_files(tmp_path / "ratings", tracks=t1 + t2, lists=l1 + l2, pairs=p1 + p2)
    report = run(tmp_path, keys)
    main = report["studies"]["main"]
    o = main["primary_preference"]["overall"]
    # L-one: P1 B=current, P2 A=current, P3 A=previous, P4 tie; L-two: P1 current, P2 B=previous, P3/P4 missing
    assert (o["current"], o["previous"], o["ties"], o["missing"]) == (3, 2, 1, 2)
    assert o["exact_ci95"] == [round(v, 4) for v in al.clopper_pearson(3, 5)]
    assert main["primary_preference"]["by_mood"]["sad"]["current"] == 2
    variety = main["secondary_list_level_current_minus_previous"]["perceived_variety_1_to_5"]["overall"]
    # list A=2, B=5: current-previous is +3 where B is current (P1, P3) and -3 where A is current (P2, P4)
    assert variety["n"] == 8 and variety["mean"] == 0.0
    assert variety["resampled"].startswith("judgments")
    assert report["studies"]["s1"]["status"] == "no ratings"
    assert (tmp_path / "out" / "r.json").exists() and (tmp_path / "out" / "r.md").exists()


def test_track_level_min_six_of_eight_and_complete_case(tmp_path, keys):
    t, l, p = full_listener("L-one", {"P1": "A"})
    # P1/A: blank 3 mood-fit ratings (5 of 8 left -> excluded); P1/B: blank 2 (6 of 8 -> kept)
    for row in t:
        if row[1] == "P1" and ((row[2] == "A" and row[3][-2:] in ("01", "02", "03"))
                               or (row[2] == "B" and row[3][-2:] in ("01", "02"))):
            row[4] = ""
    ratings_files(tmp_path / "ratings", tracks=t, lists=l, pairs=p)
    main = run(tmp_path, keys)["studies"]["main"]
    fit = main["secondary_track_level_current_minus_previous"]["mood_fit_1_to_5"]["overall"]
    enjoy = main["secondary_track_level_current_minus_previous"]["enjoyment_1_to_5"]["overall"]
    assert fit["n"] == 3 and enjoy["n"] == 4                  # P1 dropped for mood fit only
    assert main["missing"]["lists_excluded_by_min_6_of_8_rule"]["mood_fit_1_to_5"] == 1
    assert main["missing"]["track_ratings"]["mood_fit_1_to_5"] == 5


def test_speech_flags_are_compared_with_the_catalog_indicator(tmp_path, keys):
    t, l, p = full_listener("L-one", {})
    for row in t:
        row[6] = "yes" if row[3].endswith("01") else "no"
    ratings_files(tmp_path / "ratings", tracks=t, lists=l, pairs=p)
    sp = run(tmp_path, keys)["studies"]["main"]["speech_like_flags"]
    prev = sp["previous"]["all"]
    assert prev["flags_given"] == 32 and prev["catalog_speech_like_share_of_same_slots"] == 0.25
    assert prev["agreement"]["listener_yes_catalog_yes"] == 4 and prev["agreement"]["listener_no_catalog_yes"] == 4
    assert sp["current"]["all"]["catalog_speech_like_share_of_same_slots"] == 0.0


def test_s1_is_analysed_separately_and_never_pooled(tmp_path, keys):
    t, l, p = full_listener("L-one", {"P1": "B", "P2": "A", "P3": "B", "P4": "A"})
    ratings_files(tmp_path / "ratings", tracks=t, lists=l, pairs=p)
    alone = run(tmp_path, keys, "a")["studies"]["main"]
    ts, ls, ps = full_listener("L-one", {"S1": "A"}, study="s1")          # S1: A is previous
    ratings_files(tmp_path / "ratings", study="s1", tracks=ts, lists=ls, pairs=ps)
    both = run(tmp_path, keys, "b")["studies"]
    assert both["main"] == alone
    s1 = both["s1"]
    assert s1["exploratory_secondary"] is True and s1["primary_preference"]["overall"]["previous"] == 1
    assert set(s1["primary_preference"]["by_mood"]) == {"calm"}
    assert "EXPLORATORY" in (tmp_path / "b" / "r.md").read_text(encoding="utf-8")


# ---- malformed input ----

@pytest.mark.parametrize("mutate, message", [
    (lambda t, l, p: t[0].__setitem__(4, 6), "integer 1-5"),
    (lambda t, l, p: t[0].__setitem__(4, 0), "integer 1-5"),
    (lambda t, l, p: t[0].__setitem__(4, "2.5"), "integer 1-5"),
    (lambda t, l, p: t[0].__setitem__(6, "maybe"), "yes or no"),
    (lambda t, l, p: p[0].__setitem__(3, "C"), "A, B or none"),
    (lambda t, l, p: t.append(list(t[0])), "duplicate"),
    (lambda t, l, p: p.append(list(p[0])), "duplicate"),
    (lambda t, l, p: t[0].__setitem__(1, "P9"), "unknown pair_id"),
    (lambda t, l, p: t[0].__setitem__(3, "P2-A01"), "does not belong"),
    (lambda t, l, p: l[0].__setitem__(2, "C"), "unknown list"),
    (lambda t, l, p: p[0].__setitem__(2, "Match my mood: angry"), "does not match"),
    (lambda t, l, p: t[0].__setitem__(0, ""), "without a listener_id"),
])
def test_malformed_values_and_ids_fail_and_write_nothing(tmp_path, keys, mutate, message):
    t, l, p = full_listener("L-one", {"P1": "A"})
    mutate(t, l, p)
    ratings_files(tmp_path / "ratings", tracks=t, lists=l, pairs=p)
    with pytest.raises(al.InputError, match=message):
        run(tmp_path, keys)
    assert not (tmp_path / "out").exists()


def test_missing_or_unknown_columns_and_partial_file_sets_fail(tmp_path, keys):
    t, l, p = full_listener("L-one", {"P1": "A"})
    ratings_files(tmp_path / "ratings", tracks=t, lists=l, pairs=p)
    path = tmp_path / "ratings" / "ratings_lists.csv"
    _write(path, al.HEADERS["lists"][:-2] + ["notes"], [])
    with pytest.raises(al.InputError, match="missing columns"):
        run(tmp_path, keys)
    _write(path, al.HEADERS["lists"] + ["extra"], [])
    with pytest.raises(al.InputError, match="unknown columns"):
        run(tmp_path, keys)
    os.remove(path)
    with pytest.raises(al.InputError, match="incomplete file set"):
        run(tmp_path, keys)


def test_answer_key_integrity_is_enforced(tmp_path, keys):
    path = os.path.join(keys["main"], "answer_key_lists.csv")
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(text.replace("current feature-cell ranking (x)", "previous ranking (x)", 1))
    with pytest.raises(al.InputError, match="needs one previous and one current"):
        al.load_key("main", keys["main"])


def test_real_answer_keys_match_the_rating_templates_when_present():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(al.STUDIES["main"]["key_dir"], "answer_key_items.csv")):
        pytest.skip("local answer keys are not on this machine")
    for study, share in (("main", "listening_pilot_2l1"), ("s1", "listening_pilot_2l2_calm")):
        key = al.load_key(study)
        template = os.path.join(base, "evaluation", share, al.STUDIES[study]["files"]["tracks"])
        ids = [r["item_id"] for r in al._read_csv(template)[1]]
        assert sorted(ids) == sorted(key["items"])
        per_list = {}
        for info in key["items"].values():
            per_list[(info["pair_id"], info["list_label"])] = per_list.get((info["pair_id"], info["list_label"]), 0) + 1
        assert set(per_list) == set(key["lists"]) and set(per_list.values()) == {8}
        # the unmodified templates are valid input that contains no ratings
        rows = al.read_ratings(template, "tracks", key, study)
        assert rows == []


# ---- no ratings ----

def test_no_ratings_produces_no_estimates_and_writes_nothing(tmp_path, keys, capsys):
    assert run(tmp_path, keys) is None                         # no files at all
    ratings_files(tmp_path / "ratings")                        # untouched templates only
    ratings_files(tmp_path / "ratings", study="s1")
    assert run(tmp_path, keys) is None
    assert "No ratings available" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


# ---- privacy ----

def test_reports_contain_no_track_listener_or_mapping_information(tmp_path, keys):
    t, l, p = full_listener("listener-zeta", {"P1": "A", "P2": "B", "P3": "none", "P4": "A"})
    ratings_files(tmp_path / "ratings", tracks=t, lists=l, pairs=p)
    run(tmp_path, keys)
    for name in ("r.json", "r.md"):
        text = (tmp_path / "out" / name).read_text(encoding="utf-8")
        assert not re.search(r"X\d{21}", text) and "open.spotify.com" not in text
        assert "listener-zeta" not in text
        assert not re.search(r"P[1-4]-[AB]\d\d", text)          # no item ids
        assert "list_label" not in text and "previous ranking (x)" not in text
        assert not re.search(r'"(A|B)"\s*:', text)               # no label-keyed results


def test_production_and_pilot_materials_are_not_written_by_the_tests():
    # The real ratings directory may legitimately hold collected ratings; the tests must not create,
    # change or delete anything in it, nor in the real result reports.
    assert _real_outputs_state() == REAL_OUTPUTS_BEFORE
