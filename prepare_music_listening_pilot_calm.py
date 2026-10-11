"""
STUDY PREPARATION ONLY -- Phase 2L.2: a supplementary, exploratory calm
comparison for the listening pilot that removes the speech-like-content
confound of the original calm pair. The Phase 2L.1 pack is not touched (it is
read only to report sample overlap), production is unchanged, and no ratings
are generated.

Run from the project root:
    python prepare_music_listening_pilot_calm.py

Writes only:
  data/metadata/listening_pilot_2l2_calm/calm_supplement_listening_sheet.csv            LOCAL ONLY
  data/metadata/listening_pilot_2l2_calm/answer_key/calm_supplement_answer_key_items.csv LOCAL ONLY
  data/metadata/listening_pilot_2l2_calm/answer_key/calm_supplement_answer_key_lists.csv LOCAL ONLY
  evaluation/listening_pilot_2l2_calm/calm_supplement_ratings_tracks.csv, _lists.csv, _pairs.csv (blank)
  evaluation/listening_pilot_2l2_calm/README.md, ANALYSIS_PLAN_ADDENDUM.md
  models/music_listening_pilot_calm_supplement.json                                    (aggregate only)
(data/metadata/ is git-ignored local-only data, see METADATA.md.)

Design: the request "Match my mood: chill" (target calm), the same catalog and
the API's default 30-track list from each ranking with one fresh seed. From each
list, the tracks the catalog marks speech-like (speechiness > 0.66) are removed
and 8 of the remaining positions are drawn uniformly at random. If a ranking has
fewer than 8 such tracks the script stops without writing anything. List labels
and display order are randomised with fixed seeds.
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import prepare_music_listening_pilot as lp
from src import recommend as current_recommend
from src.config import MODELS_DIR, MUSIC_CATALOG_278K_PATH
from src.data_loader import load_music_metadata
from src.music_metadata import spotify_track_url
from src import music_pipeline as mp

TARGET, APP_MOOD = "calm", "chill"
REQUEST = f"{lp.MODE}: {APP_MOOD}"
PAIR_ID = "S1"
LIST_SEED = 51001
SAMPLE_SEED = 51100
BLIND_SEED = 51200
USED_SEEDS = {"tests and cold-start probes": (0, 49), "Phase 2H.6 API test": (2024, 2024),
              "exploratory (2H.5-2I)": (2026, 2075), "held-out (2H.7-2H.8)": (7000, 7049),
              "third set (2H.8.1)": (12000, 12049), "fourth set (2H.9)": (15000, 15049),
              "held-out (2I)": (18000, 18049), "confirmation (2I.1)": (21000, 21049),
              "primary (2J)": (24000, 24049), "replication (2J)": (27000, 27049),
              "primary (2J.1)": (30000, 30049), "replication (2J.1)": (33000, 33049),
              "primary (2K)": (36000, 36049), "replication (2K)": (39000, 39049),
              "primary (2K.1)": (42000, 42049), "replication (2K.1)": (45000, 45049),
              "pilot lists (2L.1)": (min(lp.LIST_SEEDS.values()), max(lp.LIST_SEEDS.values())),
              "pilot sampling (2L.1)": (lp.SAMPLE_SEED, lp.SAMPLE_SEED),
              "pilot blinding (2L.1)": (lp.BLIND_SEED, lp.BLIND_SEED)}
LOCAL_DIR = os.path.join(BASE, "data", "metadata", "listening_pilot_2l2_calm")
SHARE_DIR = os.path.join(BASE, "evaluation", "listening_pilot_2l2_calm")
REPORT_PATH = os.path.join(MODELS_DIR, "music_listening_pilot_calm_supplement.json")
PILOT_1_KEY = os.path.join(lp.LOCAL_DIR, "answer_key", "answer_key_items.csv")


class ShortageError(SystemExit):
    pass


def seeds_disjoint():
    return all(not (a <= s <= b) for s in (LIST_SEED, SAMPLE_SEED, BLIND_SEED) for a, b in USED_SEEDS.values())


def generate_lists(catalog, baseline):
    modules = {"previous": baseline, "current": current_recommend}
    lists = {}
    for strategy in lp.STRATEGIES:
        np.random.seed(LIST_SEED)
        recs = modules[strategy].recommend_music(catalog, APP_MOOD, lp.MODE, top_n=lp.LIST_SIZE)
        lists[strategy] = catalog.index.get_indexer(recs.index)
    return lists


def eligible_positions(catalog, rows):
    """0-based positions in a list whose track is not speech-like by the catalog indicator."""
    speech = catalog["speechiness"].to_numpy(dtype=float)[rows] > lp.SPEECH_LIKE
    return np.flatnonzero(~speech)


def sample(catalog, lists):
    rng = np.random.RandomState(SAMPLE_SEED)
    chosen = {}
    for strategy in lp.STRATEGIES:
        positions = eligible_positions(catalog, lists[strategy])
        if len(positions) < lp.PER_LIST:
            raise ShortageError(f"Shortage: the {strategy} ranking has only {len(positions)} non-speech-like tracks "
                                f"in its {lp.LIST_SIZE}-track calm list (needs {lp.PER_LIST}); nothing was written.")
        chosen[strategy] = np.sort(rng.choice(positions, lp.PER_LIST, replace=False))
    return chosen


def build_items(catalog, lists, chosen, metadata):
    blind = np.random.RandomState(BLIND_SEED)
    labels = ("A", "B") if blind.rand() < 0.5 else ("B", "A")          # labels for (previous, current)
    items, list_rows = [], []
    for strategy, label in sorted(zip(lp.STRATEGIES, labels), key=lambda x: x[1]):
        order = blind.permutation(lp.PER_LIST)
        list_rows.append({"pair_id": PAIR_ID, "list_label": label, "strategy": lp.STRATEGY_NAMES[strategy],
                          "strategy_key": strategy, "list_seed": LIST_SEED, "list_length": lp.LIST_SIZE})
        for display, k in enumerate(order, start=1):
            row = int(lists[strategy][chosen[strategy][k]])
            track = catalog.iloc[row]
            tid = str(track["track_id"])
            shown = metadata.display(tid) if metadata is not None else None
            items.append({
                "item_id": f"{PAIR_ID}-{label}{display:02d}", "list_label": label, "display_order": display,
                "strategy": lp.STRATEGY_NAMES[strategy], "strategy_key": strategy,
                "position_in_list": int(chosen[strategy][k]) + 1, "track_id": tid,
                "listen_url": spotify_track_url(tid),
                "title": shown["title"] if shown and shown["has_metadata"] else "",
                "artist": shown["artist"] if shown and shown["has_metadata"] else "",
                "has_metadata": bool(shown and shown["has_metadata"]),
                "speechiness": float(track["speechiness"]), "label": track["mood"],
            })
    return items, list_rows


def write_materials(items, list_rows, local_dir, share_dir):
    rows = sorted(items, key=lambda r: (r["list_label"], r["display_order"]))
    unavailable = "unavailable (no local metadata for this track)"
    request = f"{REQUEST} (supplementary pair)"
    lp.write_csv(os.path.join(local_dir, "calm_supplement_listening_sheet.csv"),
                 ["pair_id", "request", "list_label", "item_id", "listen_url", "title", "artist"],
                 [[PAIR_ID, request, r["list_label"], r["item_id"], r["listen_url"], r["title"] or unavailable,
                   r["artist"] or unavailable] for r in rows])
    lp.write_csv(os.path.join(local_dir, "answer_key", "calm_supplement_answer_key_items.csv"),
                 ["item_id", "pair_id", "list_label", "display_order", "strategy", "target", "app_request",
                  "list_seed", "position_in_list_of_30", "track_id", "listen_url", "catalog_speechiness"],
                 [[r["item_id"], PAIR_ID, r["list_label"], r["display_order"], r["strategy"], TARGET, REQUEST,
                   LIST_SEED, r["position_in_list"], r["track_id"], r["listen_url"], f"{r['speechiness']:.4f}"]
                  for r in rows])
    lp.write_csv(os.path.join(local_dir, "answer_key", "calm_supplement_answer_key_lists.csv"),
                 ["pair_id", "list_label", "strategy", "target", "app_request", "list_seed", "sampled_from_list_of",
                  "excluded_from_sampling"],
                 [[r["pair_id"], r["list_label"], r["strategy"], TARGET, REQUEST, r["list_seed"], r["list_length"],
                   f"catalog speechiness > {lp.SPEECH_LIKE}"] for r in sorted(list_rows, key=lambda r: r["list_label"])])
    lp.write_csv(os.path.join(share_dir, "calm_supplement_ratings_tracks.csv"),
                 ["listener_id", "pair_id", "list_label", "item_id", "mood_fit_1_to_5", "enjoyment_1_to_5",
                  "speech_like_yes_no", "notes"],
                 [["", PAIR_ID, r["list_label"], r["item_id"], "", "", "", ""] for r in rows])
    lp.write_csv(os.path.join(share_dir, "calm_supplement_ratings_lists.csv"),
                 ["listener_id", "pair_id", "list_label", "list_mood_fit_1_to_5", "perceived_variety_1_to_5",
                  "list_enjoyment_1_to_5", "notes"],
                 [["", PAIR_ID, label, "", "", "", ""] for label in ("A", "B")])
    lp.write_csv(os.path.join(share_dir, "calm_supplement_ratings_pairs.csv"),
                 ["listener_id", "pair_id", "request", "preferred_list_A_B_or_none", "notes"],
                 [["", PAIR_ID, request, "", ""]])
    for name, text in (("README.md", README), ("ANALYSIS_PLAN_ADDENDUM.md", ADDENDUM)):
        with open(os.path.join(share_dir, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


README = """# Supplementary calm pair (Phase 2L.2) -- exploratory, secondary

One extra pair, **S1** ("Match my mood: chill"), with lists A and B of 8 tracks
each. It is an exploratory secondary comparison added to the Phase 2L.1 pilot;
the original pairs P1-P4 and their files are unchanged and remain the primary
comparison.

- Listeners receive `data/metadata/listening_pilot_2l2_calm/calm_supplement_listening_sheet.csv`
  (local only) and the three `calm_supplement_ratings_*.csv` templates here.
- Follow `../listening_pilot_2l1/PROTOCOL.md` exactly (same device, volume,
  ~30 seconds per track, rate each track, then each list, then the pair).
- Rate S1 in a separate block after P1-P4 (or in a separate session), so its
  ratings are never mixed with the original pairs.
- The administrator alone holds
  `data/metadata/listening_pilot_2l2_calm/answer_key/` and opens it only after all
  ratings are in. Do not tell listeners how S1's tracks were chosen.
- Analysis: see `ANALYSIS_PLAN_ADDENDUM.md`.
"""

ADDENDUM = """# Analysis plan addendum -- supplementary calm pair S1 (Phase 2L.2)

Fixed before any rating is collected; it extends, and does not change,
`../listening_pilot_2l1/ANALYSIS_PLAN.md` (left byte-for-byte unchanged).

## Why S1 exists
In the original calm pair the two rankings differ in speech-like content (on
the pilot seeds the previous ranking's calm sample had 5 of 8 tracks the catalog
marks speech-like, the current one none). Listener differences there may reflect
that content rather than musical choice.

## What S1 conditions on
Both S1 lists are drawn from each ranking's own 30-track calm list (same
catalog, request and list length, a fresh seed) **after removing every track
whose catalog speechiness is above 0.66**. S1 therefore compares the two
rankings *given* the catalog's speech indicator.

## Interpretation
- S1 does **not** replace the original calm pair, which remains the real-world
  comparison of what each ranking actually serves (speech-like tracks included).
- S1 is exploratory and secondary: one pair, 8 tracks per list. Report it
  separately from P1-P4 and never pool it into the primary outcome.
- The catalog indicator is imperfect: tracks under the threshold may still sound
  speech-heavy, and listener speech-like flags are reported as a check.

## Outcomes and handling
Same definitions, missing-data rules and tie handling as the original plan,
applied to S1 alone: preference (current vs previous vs no preference) as
counts with an exact interval, and paired list-level differences for mood fit,
perceived variety and enjoyment. No pooling with P1-P4, no imputation.
"""


def aggregate_report(items, lists, chosen, catalog, local_dir, share_dir, metadata, catalog_hash, current_head):
    files = {os.path.relpath(p, BASE).replace(os.sep, "/"): lp.sha256(p) for p in sorted(
        [os.path.join(d, f) for d in (local_dir, os.path.join(local_dir, "answer_key"), share_dir)
         for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])}
    pilot_1 = set()
    if os.path.exists(PILOT_1_KEY):
        with open(PILOT_1_KEY, encoding="utf-8-sig", newline="") as f:
            pilot_1 = {r["track_id"] for r in csv.DictReader(f) if r["target"] == TARGET}
    speech = catalog["speechiness"].to_numpy(dtype=float)
    return {
        "study_preparation_only": True,
        "exploratory_secondary_comparison": True,
        "ratings_collected": 0,
        "ratings_simulated": False,
        "generated_on": datetime.date.today().isoformat(),
        "current_head": current_head,
        "current_src_recommend_sha256": lp.sha256(current_recommend.__file__),
        "baseline": {"commit": lp.BASELINE_COMMIT, "git_blob_src_recommend": lp.BASELINE_BLOB,
                     "verified_from_git_history": True},
        "catalog_sha256": catalog_hash,
        "design": {"pair_id": PAIR_ID, "target": TARGET, "app_request": REQUEST, "strategies": lp.STRATEGY_NAMES,
                   "list_length_sampled_from": lp.LIST_SIZE, "tracks_per_list": lp.PER_LIST,
                   "excluded": f"catalog speechiness > {lp.SPEECH_LIKE}", "slots": len(items),
                   "seeds": {"list": LIST_SEED, "sampling": SAMPLE_SEED, "blinding": BLIND_SEED},
                   "seeds_disjoint_from_every_earlier_set": seeds_disjoint()},
        "per_strategy": {s: {"speech_like_in_30_track_list": int((speech[lists[s]] > lp.SPEECH_LIKE).sum()),
                             "eligible_positions": int(len(eligible_positions(catalog, lists[s]))),
                             "sampled": int(len(chosen[s])),
                             "sampled_speech_like": sum(r["speechiness"] > lp.SPEECH_LIKE for r in items
                                                        if r["strategy_key"] == s),
                             "sampled_with_local_metadata": sum(r["has_metadata"] for r in items
                                                                if r["strategy_key"] == s)}
                         for s in lp.STRATEGIES},
        "tracks_shared_by_the_two_30_track_lists": len(set(lists["previous"].tolist()) & set(lists["current"].tolist())),
        "tracks_shared_by_the_two_samples": len({r["track_id"] for r in items if r["strategy_key"] == "previous"}
                                                & {r["track_id"] for r in items if r["strategy_key"] == "current"}),
        "tracks_also_in_the_phase_2l1_calm_pair": len({r["track_id"] for r in items} & pilot_1),
        "local_metadata_available": metadata is not None,
        "files_sha256": files,
        "local_only_files_contain_track_identity": [f for f in files if f.startswith("data/metadata/")],
        "notes": ["Aggregates only; track ids, titles and links are only in the local, git-ignored sheet and key.",
                  "Conditions on the catalog speechiness indicator; does not replace the original calm pair."],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare the Phase 2L.2 supplementary calm pair (no ratings)")
    parser.add_argument("--local-dir", default=LOCAL_DIR)
    parser.add_argument("--share-dir", default=SHARE_DIR)
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args()
    if not seeds_disjoint():
        raise SystemExit("Supplementary seeds overlap an earlier seed set")

    catalog_hash = mp.file_sha256(MUSIC_CATALOG_278K_PATH)
    current_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True,
                                  check=True).stdout.strip()
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    lists = generate_lists(catalog, lp.load_baseline_module())
    chosen = sample(catalog, lists)                # stops here, before writing, on a shortage
    metadata = load_music_metadata()
    items, list_rows = build_items(catalog, lists, chosen, metadata)
    write_materials(items, list_rows, args.local_dir, args.share_dir)
    report = aggregate_report(items, lists, chosen, catalog, args.local_dir, args.share_dir, metadata,
                              catalog_hash, current_head)
    if mp.file_sha256(MUSIC_CATALOG_278K_PATH) != catalog_hash:
        raise SystemExit("The catalog changed during preparation -- this must never happen.")
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Prepared supplementary pair {PAIR_ID}: {report['design']['slots']} slots in "
          f"{len(report['files_sha256'])} files; report {os.path.relpath(args.report, BASE)}")


if __name__ == "__main__":
    main()
