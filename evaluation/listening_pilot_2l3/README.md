# Listening-pilot results workflow (Phase 2L.3)

For the administrator, after ratings are collected. Nothing here collects,
simulates or infers ratings.

## 1. Gather completed rating files (local only)
Put the listeners' completed copies of the rating templates in
`data/metadata/listening_results/` (git-ignored, never committed):

| Study | Files |
|---|---|
| Primary pilot P1-P4 | `ratings_tracks.csv`, `ratings_lists.csv`, `ratings_pairs.csv` |
| Supplementary calm pair S1 (exploratory) | `calm_supplement_ratings_tracks.csv`, `calm_supplement_ratings_lists.csv`, `calm_supplement_ratings_pairs.csv` |

- Keep the template columns and ids unchanged; add one row per listener per
  item/list/pair (copy the template rows for each listener and fill in
  `listener_id`, a code, never a name).
- Supply all three files of a study or none of them. S1 is optional.
- Leave a cell blank when it was not rated; never fill in guesses.

## 2. Run
```
python analyze_music_listening_pilot.py
```
- Any malformed or inconsistent input stops the run with an `Input error`
  message and writes nothing.
- With no ratings it prints `No ratings available` and writes nothing.
- Otherwise it writes `models/music_listening_pilot_results.json` and
  `evaluation/listening_pilot_2l3/listening_pilot_results.md`: aggregates only,
  with no track, listener or label-to-ranking information.

The strategy behind each A/B list is joined only from the local answer keys in
`data/metadata/listening_pilot_2l1/answer_key/` and
`data/metadata/listening_pilot_2l2_calm/answer_key/`. Do not open them, or the
results, until every rating is in.

## 3. Read
Methods follow `../listening_pilot_2l1/ANALYSIS_PLAN.md` and
`../listening_pilot_2l2_calm/ANALYSIS_PLAN_ADDENDUM.md`; details those plans leave
open are in `IMPLEMENTATION_NOTES.md`. S1 is reported separately and is never
pooled with P1-P4.
