# Implementation notes -- listening-pilot analysis (Phase 2L.3)

Written before any rating was collected. These are implementation details the
pre-written plans (`../listening_pilot_2l1/ANALYSIS_PLAN.md`,
`../listening_pilot_2l2_calm/ANALYSIS_PLAN_ADDENDUM.md`) leave open. They do not
change any outcome, missing-data rule or tie rule in those plans. The same list
is embedded in `analyze_music_listening_pilot.py` (`IMPLEMENTATION_DETAILS`) and
in every results report.

## Unchanged from the plans
- Primary outcome: share of non-tie pair judgments preferring the current
  ranking, exact 95% interval, overall and per mood; ties counted separately.
- Secondary: paired current-minus-previous differences (list mood fit, perceived
  variety, list enjoyment; track mood fit and enjoyment averaged per list) with
  bootstrap intervals; speech-like flags next to the catalog indicator.
- Complete-case per outcome, no imputation; a pair without a preference is
  excluded from the primary outcome; a track-averaged list needs at least 6 of 8
  rated tracks; ties excluded from the share, reported as counts, and counted as
  half in a sensitivity analysis.
- S1 analysed separately with the same rules and never pooled with P1-P4.

## Details fixed here
1. **Exact interval**: Clopper-Pearson, from beta quantiles (`scipy.stats.beta`).
   No interval when there are no non-tie judgments.
2. **Sign test**: two-sided exact binomial test at 0.5 (`scipy.stats.binomtest`)
   on non-tie judgments; descriptive only, as the plan says.
3. **Tie sensitivity**: (current wins + 0.5 x ties) / (wins + losses + ties),
   reported as a point estimate without an interval.
4. **Bootstrap**: 5,000 percentile resamples of the mean, seed 52000, with a fresh
   generator for every outcome so results do not depend on analysis order.
   "At least 5 listeners" is counted per outcome (listeners contributing a
   paired difference to it); then whole listeners are resampled with all their
   judgments, otherwise individual judgments are resampled. The method used is
   reported per outcome. No interval for fewer than 2 paired differences.
5. **Track-level rule**: the 6-of-8 minimum is applied separately to mood fit and
   to enjoyment.
6. **Speech-like comparison**: per strategy and per mood, listener "yes" flags /
   flags given, next to the share of the same rated slots the catalog marks
   speechiness > 0.66, plus listener-by-catalog agreement counts.
7. **Who counts as a listener**: anyone with at least one rating in a study's
   files. For such a listener, every expected pair, list or track judgment
   without a value is a missing response and is counted.
8. **Input rules**: ratings are integers 1-5; speech flags yes/no; preferences A,
   B or none (case-insensitive); blank means missing. Template rows with an empty
   listener id and no ratings are ignored. Any of these stops the run with an
   error and writes nothing: a rating without a listener id, an unknown pair,
   list or item id, an item listed under the wrong pair/list, a request that
   does not match its pair, a duplicate judgment by the same listener, a missing
   or unknown column, extra cells, or a study with only some of its three files.
9. **Privacy**: reports contain aggregates only -- no track ids, titles,
   artists, links, listener ids, or which label (A/B) belongs to which ranking.
