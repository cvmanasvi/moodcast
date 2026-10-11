"""
STUDY PREPARATION ONLY -- Phase 2L.1: build a reproducible, offline pack for a
blinded human listening pilot comparing the previous music ranking (before
Phase 2K) with the current feature-cell ranking. Nothing here changes
production, and no ratings are generated.

Run from the project root:
    python prepare_music_listening_pilot.py

Reads   data/music_catalog_278k.csv, the optional local display metadata
        (data/metadata/music_metadata.csv), and the pre-Phase-2K
        src/recommend.py from Git history (exact blob, verified by hash).
        No network or API access.
Writes  only:
  data/metadata/listening_pilot_2l1/listening_sheet.csv           LOCAL ONLY (track links + local metadata)
  data/metadata/listening_pilot_2l1/answer_key/answer_key_items.csv  LOCAL ONLY, keep from listeners
  data/metadata/listening_pilot_2l1/answer_key/answer_key_lists.csv  LOCAL ONLY, keep from listeners
  evaluation/listening_pilot_2l1/ratings_tracks.csv, ratings_lists.csv, ratings_pairs.csv  (blank templates)
  evaluation/listening_pilot_2l1/PROTOCOL.md, ANALYSIS_PLAN.md
  models/music_listening_pilot_pack.json                          (aggregate only)
data/metadata/ is git-ignored (local-only metadata, see METADATA.md), so the
files that carry track IDs and titles stay out of Git.

Design: for each pilot mood (happy, sad, calm via chill, energetic; "Match my
mood"), both strategies produce the API's default 30-track list from the same
catalog and the same seed. 8 positions are sampled uniformly at random from
each list (not its top 8). Pair order, which strategy is list A or B, and the
display order inside each list are randomised with fixed seeds.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from src import recommend as current_recommend
from src.config import MODELS_DIR, MUSIC_CATALOG_278K_PATH
from src.data_loader import load_music_metadata
from src.music_metadata import spotify_track_url
from src import music_pipeline as mp

BASELINE_COMMIT = "9d34ac0dc1af484d5c3f9b50449535e98b6ba409"     # last commit before Phase 2K
BASELINE_BLOB = "91656a82d8cc92d1cf5f8cb412a32462449c93c4"       # src/recommend.py at that commit
PILOT = (("happy", "happy"), ("sad", "sad"), ("calm", "chill"), ("energetic", "energetic"))  # (target, app mood)
MODE = "Match my mood"
LIST_SIZE = 30                       # the API's default list length
PER_LIST = 8
LIST_SEEDS = {"happy": 48001, "sad": 48002, "calm": 48003, "energetic": 48004}
SAMPLE_SEED = 48100
BLIND_SEED = 48200
STRATEGIES = ("previous", "current")
STRATEGY_NAMES = {"previous": f"previous ranking (pre-Phase 2K, {BASELINE_COMMIT[:7]})",
                  "current": "current feature-cell ranking (Phase 2K/2K.1)"}
SPEECH_LIKE = 0.66
LOCAL_DIR = os.path.join(BASE, "data", "metadata", "listening_pilot_2l1")
SHARE_DIR = os.path.join(BASE, "evaluation", "listening_pilot_2l1")
REPORT_PATH = os.path.join(MODELS_DIR, "music_listening_pilot_pack.json")


# ---- the exact previous ranking, from Git history ----

def git_blob_sha(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def baseline_source() -> bytes:
    """src/recommend.py as of BASELINE_COMMIT, verified against BASELINE_BLOB."""
    resolved = subprocess.run(["git", "rev-parse", f"{BASELINE_COMMIT}:src/recommend.py"], cwd=BASE,
                              capture_output=True, text=True, check=True).stdout.strip()
    if resolved != BASELINE_BLOB:
        raise SystemExit(f"Baseline blob mismatch: {resolved} != {BASELINE_BLOB}")
    content = subprocess.run(["git", "cat-file", "blob", BASELINE_BLOB], cwd=BASE,
                             capture_output=True, check=True).stdout
    if git_blob_sha(content) != BASELINE_BLOB:
        raise SystemExit("Baseline source does not hash to the recorded blob")
    return content


def load_baseline_module():
    """The pre-Phase-2K recommend module, loaded in isolation (production files are not touched)."""
    spec = importlib.util.spec_from_loader("src._recommend_pre_phase_2k", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "src"
    exec(compile(baseline_source(), f"<git {BASELINE_COMMIT[:7]}:src/recommend.py>", "exec"), module.__dict__)
    return module


# ---- sample ----

def generate_lists(catalog, baseline):
    """{(target, strategy): row positions of the 30-track list}, same catalog and seed for both strategies."""
    modules = {"previous": baseline, "current": current_recommend}
    lists = {}
    for target, app in PILOT:
        for strategy in STRATEGIES:
            np.random.seed(LIST_SEEDS[target])
            recs = modules[strategy].recommend_music(catalog, app, MODE, top_n=LIST_SIZE)
            lists[(target, strategy)] = catalog.index.get_indexer(recs.index)
    return lists


def build_items(catalog, lists, metadata):
    """One record per sampled slot (64), with anonymous ids and the hidden mapping."""
    sample_rng = np.random.RandomState(SAMPLE_SEED)
    positions = {key: np.sort(sample_rng.choice(LIST_SIZE, PER_LIST, replace=False))
                 for key in [(t, s) for t, _ in PILOT for s in STRATEGIES]}
    blind = np.random.RandomState(BLIND_SEED)
    pair_order = blind.permutation(len(PILOT))
    items, list_rows = [], []
    for pair_no, p in enumerate(pair_order, start=1):
        target, app = PILOT[p]
        labels = ("A", "B") if blind.rand() < 0.5 else ("B", "A")      # labels for (previous, current)
        for strategy, label in sorted(zip(STRATEGIES, labels), key=lambda x: x[1]):
            pos = positions[(target, strategy)]
            order = blind.permutation(PER_LIST)
            list_rows.append({"pair_id": f"P{pair_no}", "list_label": label, "strategy": STRATEGY_NAMES[strategy],
                              "strategy_key": strategy, "target": target, "app_request": f"{MODE}: {app}",
                              "list_seed": LIST_SEEDS[target], "list_length": LIST_SIZE})
            for display, k in enumerate(order, start=1):
                row = int(lists[(target, strategy)][pos[k]])
                track = catalog.iloc[row]
                tid = str(track["track_id"])
                shown = metadata.display(tid) if metadata is not None else None
                items.append({
                    "item_id": f"P{pair_no}-{label}{display:02d}", "pair_id": f"P{pair_no}", "list_label": label,
                    "display_order": display, "strategy": STRATEGY_NAMES[strategy], "strategy_key": strategy,
                    "target": target, "app_request": f"{MODE}: {app}", "list_seed": LIST_SEEDS[target],
                    "position_in_list": int(pos[k]) + 1, "track_id": tid, "listen_url": spotify_track_url(tid),
                    "title": shown["title"] if shown and shown["has_metadata"] else "",
                    "artist": shown["artist"] if shown and shown["has_metadata"] else "",
                    "has_metadata": bool(shown and shown["has_metadata"]),
                    "speech_like": bool(float(track["speechiness"]) > SPEECH_LIKE),
                    "label_matches_target": track["mood"] == target,
                })
    return items, list_rows


# ---- writers ----

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:     # BOM so spreadsheets read UTF-8 titles
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_materials(items, list_rows, local_dir, share_dir):
    by_pair = sorted(items, key=lambda r: (int(r["pair_id"][1:]), r["list_label"], r["display_order"]))
    unavailable = "unavailable (no local metadata for this track)"
    write_csv(os.path.join(local_dir, "listening_sheet.csv"),
              ["pair_id", "request", "list_label", "item_id", "listen_url", "title", "artist"],
              [[r["pair_id"], r["app_request"], r["list_label"], r["item_id"], r["listen_url"],
                r["title"] or unavailable, r["artist"] or unavailable] for r in by_pair])
    write_csv(os.path.join(local_dir, "answer_key", "answer_key_items.csv"),
              ["item_id", "pair_id", "list_label", "display_order", "strategy", "target", "app_request",
               "list_seed", "position_in_list_of_30", "track_id", "listen_url", "catalog_speechiness_gt_0_66"],
              [[r["item_id"], r["pair_id"], r["list_label"], r["display_order"], r["strategy"], r["target"],
                r["app_request"], r["list_seed"], r["position_in_list"], r["track_id"], r["listen_url"],
                "yes" if r["speech_like"] else "no"] for r in by_pair])
    write_csv(os.path.join(local_dir, "answer_key", "answer_key_lists.csv"),
              ["pair_id", "list_label", "strategy", "target", "app_request", "list_seed", "sampled_from_list_of"],
              [[r["pair_id"], r["list_label"], r["strategy"], r["target"], r["app_request"], r["list_seed"],
                r["list_length"]] for r in sorted(list_rows, key=lambda r: (int(r["pair_id"][1:]), r["list_label"]))])
    write_csv(os.path.join(share_dir, "ratings_tracks.csv"),
              ["listener_id", "pair_id", "list_label", "item_id", "mood_fit_1_to_5", "enjoyment_1_to_5",
               "speech_like_yes_no", "notes"],
              [["", r["pair_id"], r["list_label"], r["item_id"], "", "", "", ""] for r in by_pair])
    pairs = sorted({(r["pair_id"], r["app_request"]) for r in items}, key=lambda x: int(x[0][1:]))
    write_csv(os.path.join(share_dir, "ratings_lists.csv"),
              ["listener_id", "pair_id", "list_label", "list_mood_fit_1_to_5", "perceived_variety_1_to_5",
               "list_enjoyment_1_to_5", "notes"],
              [["", p, label, "", "", "", ""] for p, _ in pairs for label in ("A", "B")])
    write_csv(os.path.join(share_dir, "ratings_pairs.csv"),
              ["listener_id", "pair_id", "request", "preferred_list_A_B_or_none", "notes"],
              [["", p, req, "", ""] for p, req in pairs])
    with open(os.path.join(share_dir, "PROTOCOL.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(PROTOCOL)
    with open(os.path.join(share_dir, "ANALYSIS_PLAN.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(ANALYSIS_PLAN)


PROTOCOL = """# MoodCast listening pilot (Phase 2L.1) -- protocol

Pilot comparing two music rankings for four "Match my mood" requests (happy,
sad, chill, energetic). Each request is a *pair* (P1-P4) of two 8-track lists,
labelled A and B. The labels and orders are random; nothing on the listening
sheet says which ranking produced a list.

## Roles
- **Administrator** prepares the sheet, collects ratings and alone holds
  `data/metadata/listening_pilot_2l1/answer_key/`. The administrator must not
  tell listeners anything about how lists were made, and must not open the
  answer key until every rating is in.
- **Listeners** receive only `listening_sheet.csv` and the three rating
  templates (`ratings_tracks.csv`, `ratings_lists.csv`, `ratings_pairs.csv`).

## Blinding: what it does and does not cover
- The sheet shows no strategy, seed, rank or score, and track order inside each
  list is shuffled.
- It is **not** fully blind: listening on Spotify shows each track's title,
  artist and artwork, and a listener may recognise songs. If the administrator
  is also a listener, or has seen the answer key or the generating code, the
  study is not blind for that person -- record this in the notes.
- The current ranking removes speech-like tracks from calm lists, so content
  differences (e.g. spoken-word tracks) can themselves hint at a list's origin.
  That is part of what is being measured, not a leak in the materials.

## Listening procedure (per listener)
1. Use the same device, headphones/speakers and a comfortable fixed volume for
   the whole session. Close other audio.
2. Work through pairs in sheet order (P1 to P4). Within a pair, listen to list
   A then list B, tracks in sheet order.
3. For each track, play about 30 seconds from the start (skip a silent intro if
   needed; use the same rule throughout). Do not read lyrics or reviews.
4. Rate each track right after listening, in `ratings_tracks.csv`:
   mood fit to the request (1 = not at all, 5 = perfectly), enjoyment (1-5),
   and whether it is mostly speech rather than music (yes/no).
5. After each list, rate it in `ratings_lists.csv`: overall mood fit, perceived
   variety (1 = all sounds the same, 5 = very varied while still fitting), and
   enjoyment (1-5).
6. After both lists of a pair, record the preferred list (A, B or none) in
   `ratings_pairs.csv`. "None" means no real preference.
7. Put your listener id (any code, no names) in the first column of every row
   you fill. Leave a cell blank if you could not rate it (e.g. a link that does
   not play) and say why in notes.
8. Do not change earlier ratings after hearing later lists.

## Links and metadata
Links are Spotify track pages built from track ids (no API calls). Titles and
artists come only from the optional local metadata file; where a track has none,
the sheet says "unavailable" -- nothing is guessed.
"""

ANALYSIS_PLAN = """# MoodCast listening pilot (Phase 2L.1) -- analysis plan

Fixed before any rating is collected. No ratings are simulated or generated.

## Units
- Pair judgment: one listener x one request pair (4 per listener).
- List rating: one listener x one list (8 per listener).
- Track rating: one listener x one track slot (64 per listener).
Strategy labels are joined from the answer key only after all ratings are in.

## Primary outcome
Overall preference per pair judgment: current ranking, previous ranking, or no
preference. Summary: the share of non-tie judgments preferring the current
ranking, with an exact (Clopper-Pearson) 95% interval, overall and per mood;
counts of ties reported alongside. A two-sided exact sign test is reported as
descriptive only.

## Secondary outcomes
- List-level perceived variety, list mood fit and list enjoyment: paired
  difference (current minus previous) per pair judgment; mean and median with a
  95% bootstrap interval over pair judgments (resampling listeners when there
  are at least 5, otherwise judgments, stated explicitly).
- Track-level mood fit and enjoyment: averaged per list first, then compared as
  paired list differences as above.
- Speech-like content: the share of tracks flagged by listeners per strategy and
  per mood, shown next to the catalog indicator (speechiness > 0.66) from the
  answer key.

## Missing data and ties
- Analyses are complete-case per outcome; no imputation.
- A pair judgment without a preference is excluded from the primary outcome.
- A track-averaged list score needs at least 6 of 8 rated tracks; otherwise that
  list is excluded for that outcome (and its pair for the paired difference).
- "No preference" is a tie: excluded from the preference share, always reported
  as a count, and in a sensitivity analysis counted as half a preference for
  each side.

## Interpretation limits
- This is an exploratory pilot: 4 requests, 8 tracks per list, one sample per
  mood, and whatever listeners are available. It cannot establish broad,
  population-level conclusions, and results for one mood rest on one pair of
  lists.
- Listeners are not blind to song identity (see PROTOCOL.md), and repeated
  judgments by the same listener are not independent.
- Feature-space diversity from earlier phases is not validated by this pilot
  unless listener variety ratings agree with it; any disagreement is reported.
"""


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def aggregate_report(items, lists, local_dir, share_dir, metadata, catalog_hash, current_head):
    per = {}
    for target, _ in PILOT:
        a = set(lists[(target, "previous")].tolist())
        b = set(lists[(target, "current")].tolist())
        sa = {r["track_id"] for r in items if r["target"] == target and r["strategy_key"] == "previous"}
        sb = {r["track_id"] for r in items if r["target"] == target and r["strategy_key"] == "current"}
        per[target] = {
            "tracks_shared_by_the_two_30_track_lists": len(a & b),
            "tracks_shared_by_the_two_8_track_samples": len(sa & sb),
            "sampled_speech_like": {s: sum(r["speech_like"] for r in items
                                           if r["target"] == target and r["strategy_key"] == s) for s in STRATEGIES},
            "sampled_with_local_metadata": {s: sum(r["has_metadata"] for r in items
                                                   if r["target"] == target and r["strategy_key"] == s)
                                            for s in STRATEGIES},
            "sampled_label_matches_target": all(r["label_matches_target"] for r in items if r["target"] == target),
        }
    files = {os.path.relpath(p, BASE).replace(os.sep, "/"): sha256(p) for p in sorted(
        [os.path.join(d, f) for d in (local_dir, os.path.join(local_dir, "answer_key"), share_dir)
         for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])}
    return {
        "study_preparation_only": True,
        "ratings_collected": 0,
        "ratings_simulated": False,
        "generated_on": datetime.date.today().isoformat(),
        "current_head": current_head,
        "current_src_recommend_sha256": sha256(current_recommend.__file__),
        "baseline": {"commit": BASELINE_COMMIT, "git_blob_src_recommend": BASELINE_BLOB,
                     "verified_from_git_history": True,
                     "method": "src/recommend.py loaded from the baseline blob into an isolated module"},
        "catalog_sha256": catalog_hash,
        "design": {"moods": [{"target": t, "app_request": f"{MODE}: {a}"} for t, a in PILOT],
                   "strategies": STRATEGY_NAMES, "list_length_sampled_from": LIST_SIZE,
                   "tracks_per_list": PER_LIST, "slots": len(items),
                   "unique_tracks": len({r["track_id"] for r in items}),
                   "sampling": "8 positions uniformly at random (without replacement) from each 30-track list",
                   "seeds": {"lists": LIST_SEEDS, "sampling": SAMPLE_SEED, "blinding": BLIND_SEED}},
        "per_mood": per,
        "local_metadata_available": metadata is not None,
        "files_sha256": files,
        "local_only_files_contain_track_identity": [f for f in files if f.startswith("data/metadata/")],
        "notes": [
            "This report holds aggregates only; track ids, titles, artists and links are only in the local, "
            "git-ignored listening sheet and answer key.",
            "Blinding is partial: the listening sheet hides the strategy, but song identity is visible when "
            "listening (see PROTOCOL.md).",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare the Phase 2L.1 listening pilot (no ratings)")
    parser.add_argument("--local-dir", default=LOCAL_DIR)
    parser.add_argument("--share-dir", default=SHARE_DIR)
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args()

    catalog_hash = mp.file_sha256(MUSIC_CATALOG_278K_PATH)
    current_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True,
                                  check=True).stdout.strip()
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    baseline = load_baseline_module()
    metadata = load_music_metadata()
    lists = generate_lists(catalog, baseline)
    items, list_rows = build_items(catalog, lists, metadata)
    write_materials(items, list_rows, args.local_dir, args.share_dir)
    report = aggregate_report(items, lists, args.local_dir, args.share_dir, metadata, catalog_hash, current_head)
    if mp.file_sha256(MUSIC_CATALOG_278K_PATH) != catalog_hash:
        raise SystemExit("The catalog changed during preparation -- this must never happen.")
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Prepared {report['design']['slots']} slots ({report['design']['unique_tracks']} unique tracks) "
          f"in {len(report['files_sha256'])} files; report {os.path.relpath(args.report, BASE)}")


if __name__ == "__main__":
    main()
