# MoodCast listening pilot (Phase 2L.1) -- analysis plan

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
