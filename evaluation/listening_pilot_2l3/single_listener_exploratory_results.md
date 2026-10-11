# Single-listener exploratory results -- listening pilot

**Status: exploratory and descriptive only. One listener, 20 tracks. This is not a
multi-listener validation and does not establish that either ranking is better.**

## What was collected
- One listener (the project author) rated 20 individual tracks from the Phase 2L.1
  pilot: mood fit (1-5), enjoyment (1-5) and a speech-like flag.
- Playlist (list-level) ratings and A/B pair preferences were **not** collected.
- The recorded ratings are kept unchanged in the git-ignored
  `data/metadata/listening_results/ratings_tracks.csv`; the shareable template was
  restored to its original blank state.
- Each rated item was joined to its ranking (previous = pre-Phase 2K ranking at
  9d34ac0; current = Phase 2K/2K.1 feature-cell ranking) only through the local,
  private answer key. All 20 items matched the key's pair and list. The key and the
  A/B-to-ranking mapping are not reproduced here.
- These numbers were computed directly from the 20 recorded ratings, not by
  `analyze_music_listening_pilot.py`: that script implements the pre-written
  multi-listener plan, whose primary outcome needs pair preferences, and the
  recorded file has no listener id.

## Results (means of the recorded ratings; no intervals or tests)
| Mood | Ranking | Tracks rated | Mood fit, mean | Enjoyment, mean |
|---|---|---|---|---|
| Energetic | current | 8 of 8 | 4.75 | 4.38 |
| Energetic | previous | 8 of 8 | 4.75 | 4.00 |
| Sad | previous | 4 of 8 | 4.00 | 3.75 |
| Sad | current | 0 of 8 | -- | -- |
| Happy | both | 0 | -- | -- |
| Chill | both | 0 | -- | -- |

- **Energetic** is the only mood where the two rankings can be compared. Mood fit was
  the same (4.75 each). Enjoyment was 0.375 points higher for the current ranking
  (4.38 vs 4.00). This difference is descriptive only: it comes from one listener,
  8 tracks per ranking, and is well within what individual taste and track-to-track
  variation could produce. It does not show that the current method is better.
- **Sad**: all 4 rated tracks come from the previous ranking, so there is no valid
  between-method comparison.
- **Happy and chill**: no ratings.

## Excluded from interpretation
- **Speech-like flag**: every rated track was marked "yes", while the catalog's
  speechiness indicator marks none of them as speech-like; the notes indicate the
  question was read as "has vocals/shouting" rather than "spoken word". The field is
  not interpreted.

## What cannot be estimated
- The planned primary outcome (share of pair preferences favoring the current
  ranking, `../listening_pilot_2l1/ANALYSIS_PLAN.md`) cannot be estimated: no pair
  preferences were recorded.
- The planned list-level outcomes (list mood fit, perceived variety, list enjoyment)
  cannot be estimated: no playlist ratings were recorded.
- The supplementary calm pair S1 was not rated.
- No confidence intervals or significance tests are reported: with one listener the
  plan's listener-level resampling is not possible, and none would support a
  general conclusion.

The method decision for the production recommender (Phase 2K) rests on the earlier
offline evaluations, not on this listening check.
