"""
ANALYSIS ONLY -- Phase 2J: balanced recommendation-quality analysis across
"Match my mood" and "Cheer me up". Nothing here changes production: the app
still calls src.recommend.recommend_music (pool_multiplier=20).

Run from the project root:
    python evaluate_music_balanced_quality.py [--output PATH]

Reads   data/music_catalog_278k.csv (read-only)
Writes  models/music_balanced_quality_analysis.json (aggregate metrics only: no
        track IDs, titles, artists, names or URIs). No earlier report is touched.

Hypothesis (from Phases 2H.7-2I.1): ranking a target's eligible tracks by
confidence selects that mood's feature-space extreme. Single-target lists are
therefore narrow; two-target uplift lists look diverse largely because they
join two moods' extremes (between-target separation), not because of variety
within each mood. Pools that move picks toward each mood's centre (2I) widen
single-target lists but shrink that separation. H1: choosing the most confident
tracks *within fixed feature cells spanning each target's whole eligible
population* raises within-target diversity in both families without an
uplift regression, while keeping label consistency and classifier agreement.

Strategies (all keep eligibility, exact shares and interleaving; none expands
the top-N pool or applies a confidence threshold):
  P   production                      (replica, verified identical)
  G   production + calm speech guard  (calm candidates exclude speechiness > 0.66)
  S   feature-cell selection + guard  PRIMARY. Per target, its eligible tracks are
      split once into 30 equal-count cells: 6 quantile bins on the first principal
      component of their z-scored audio features, then 5 quantile bins on the second
      within each. A share of n draws n distinct cells (np.random), and from each
      cell one production-weighted pick among its CELL_TOP most confident available
      tracks. Cells are fixed per catalog (deterministic, no per-request clustering).
  SM  S for single-target requests, G for multi-target requests (secondary, scoped)
PREREGISTRATION and GATES were fixed, and the tests written, before the first
run on the evaluation seeds; the script records its own SHA-256.
"""
import argparse
import datetime
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_music_calm_guard_analysis as guard
import evaluate_music_diversity_analysis as div
import evaluate_music_recommender_analysis as sweep
import evaluate_music_stratified_prototype as proto
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend

OUTPUT_PATH = os.path.join(MODELS_DIR, "music_balanced_quality_analysis.json")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "music_mood_classifier_278k.pkl")
RUNS = 50
PRIMARY_SEED = 24000
REPLICATION_SEED = 27000
USED_SEEDS = {"tests and cold-start probes": (0, 49), "Phase 2H.6 API test": (2024, 2024),
              "exploratory (2H.5-2I)": (2026, 2075), "held-out (2H.7-2H.8)": (7000, 7049),
              "third set (2H.8.1)": (12000, 12049), "fourth set (2H.9)": (15000, 15049),
              "held-out (2I)": (18000, 18049), "confirmation (2I.1)": (21000, 21049)}
PC1_BINS, PC2_BINS = 6, 5            # 30 cells = list size
CELL_TOP = 20                        # confident candidates per cell (production keeps 20 per slot)
LABEL_TARGETS = tuple(recommend._LABEL_BY_NAME)
BOOTSTRAP = 2000
COLD_START_PROCESSES = 5

P, G, S, SM = ("P: production", "G: production + calm guard", "S: feature cells + calm guard (primary)",
               "SM: feature cells for single-target, G for multi-target")
VARIANTS = (P, G, S, SM)
REFERENCE = G        # identical to P outside calm; speech-like calm spread is not quality (2H.8.1)

PREREGISTRATION = {
    "families": "Unique effective configurations by (target, share): single-target family = 6 (happy, sad, "
                "angry, calm, energetic, stressed; chill/match and stressed/uplift are the same calm:30 and count "
                "once); multi-target family = 5 two-target uplift mixes. Families are scored separately.",
    "reference": "Paired comparisons use G (production + calm guard) as the reference. G is identical to production "
                 "on every configuration without calm (same seeds, same lists); on calm configurations it removes "
                 "speech-like tracks, whose feature spread Phase 2H.8.1 established is not a quality gain. "
                 "Unguarded production is reported alongside.",
    "primary_and_secondary": "S is the primary candidate. SM is reported as a scoped secondary design and is the "
                             "proposal candidate only if S fails a multi-target gate while SM passes every gate.",
    "diversity_decomposition": "list diversity = mean pairwise distance over the whole list (as in all earlier "
                               "phases); within-target diversity = mean pairwise distance among slots of the same "
                               "target (pairs pooled); between-target separation = mean distance between slots "
                               "of different targets (multi-target only).",
    "relevance": "Measured: label/rule consistency and exact shares (circular by construction). Proxy: classifier "
                 "agreement (classifier argmax == dataset label) and mean classifier probability on base-label "
                 "slots. The earlier mean-probability >= 0.99 gate is reported as a legacy reference, not as a "
                 "gate: Phase 2I showed it confines selection to the confident core, where feature concentration "
                 "is intrinsic, so with any diversity gate it decides the outcome in advance.",
    "seeds": "No parameter is tuned: S's 30 cells and CELL_TOP=20 are fixed from the list size and production's "
             "20-per-slot pool. Gates are applied on the primary seeds 24000-24049; replication seeds 27000-27049 "
             "are reported and must not reverse a gate decision.",
    "uncertainty": "Paired per-seed differences against the reference, 2,000-sample percentile bootstrap over the 50 "
                   "runs; family means bootstrap the per-run mean over the family's configurations. Intervals "
                   "describe sampling variability on one catalog, not listener-level significance.",
}

GATES = {
    "integrity": "100% target consistency, exact shares in every list, same-seed determinism on all 12 requests",
    "single_family_mean_gain_lower_bound_gt": 0.0,
    "single_family_configs_with_lower_bound_gt_0_min": 4,      # of 6
    "single_config_regression_upper_bound_lt": -0.10,          # any config whose 95% upper bound is below this fails
    "multi_family_mean_change_lower_bound_gt": -0.10,          # non-inferiority on list diversity
    "multi_config_change_lower_bound_gt": -0.25,               # per configuration
    "min_classifier_agreement_base_label_slots": 0.95,
    "min_mean_classifier_probability_base_label_slots": 0.95,
    "min_catalog_unique_tracks": 2000,
    "max_mean_repeat_rate": 0.75,
    "max_latency_p95_ms": 25.0,
    "max_calm_speech_like_share": 0.20,
    "calm_speech_like_not_worse_than_production": True,
}


def seeds_disjoint():
    sets = [(PRIMARY_SEED, PRIMARY_SEED + RUNS - 1), (REPLICATION_SEED, REPLICATION_SEED + RUNS - 1)]
    ok = all(hi < a or lo > b for lo, hi in sets for a, b in USED_SEEDS.values())
    return ok and (sets[0][1] < sets[1][0])


# ---- configurations ----

def configuration(mood, mode):
    targets = ev.request_targets(mood, mode)
    shares = recommend._allocate_shares([w for _, w in targets], ev.LIST_SIZE)
    return tuple((t, int(s)) for (t, _), s in zip(targets, shares))


def families():
    single, multi = {}, {}
    for mood in MOODS:
        for mode in ev.MODES:
            config = configuration(mood, mode)
            (single if len(config) == 1 else multi).setdefault(config, []).append(f"{mood} / {mode}")
    return single, multi


def config_name(config):
    return " + ".join(f"{t}:{s}" for t, s in config)


# ---- S: fixed feature cells ----

def build_cells(catalog, z, guarded):
    """Per target: (cell id per catalog row, -1 if ineligible) and each cell's rows sorted by signal desc."""
    candidates = guard.guarded_candidates if guarded else recommend._target_candidates
    out = {}
    for target in list(LABEL_TARGETS) + ["angry", "stressed"]:
        member, signal = candidates(catalog, target)
        rows = np.flatnonzero(member)
        f = z[rows] - z[rows].mean(axis=0)
        _, _, vt = np.linalg.svd(f, full_matrices=False)
        pc = f @ vt[:2].T
        cell = np.empty(len(rows), dtype=int)
        b1 = np.empty(len(rows), dtype=int)
        b1[np.argsort(pc[:, 0], kind="stable")] = np.arange(len(rows)) * PC1_BINS // len(rows)
        for b in range(PC1_BINS):
            sel = np.flatnonzero(b1 == b)
            b2 = np.empty(len(sel), dtype=int)
            b2[np.argsort(pc[sel, 1], kind="stable")] = np.arange(len(sel)) * PC2_BINS // len(sel)
            cell[sel] = b * PC2_BINS + b2
        labels = np.full(len(catalog), -1, dtype=int)
        labels[rows] = cell
        by_cell = []
        for c in range(PC1_BINS * PC2_BINS):
            r = rows[cell == c]
            by_cell.append(r[np.argsort(-signal[r], kind="stable")])
        out[target] = {"labels": labels, "by_cell": by_cell}
    return out


def cell_select(target, signal, available, n, cells):
    """n picks from n distinct cells (random order); within a cell, a production-weighted pick among its
    CELL_TOP most confident available tracks. Cells with nothing available are skipped. Best-first."""
    if n <= 0:
        return np.array([], dtype=int)
    picks = []
    for c in np.random.permutation(len(cells[target]["by_cell"])):
        rows = cells[target]["by_cell"][c]
        top = rows[available[rows]][:CELL_TOP]
        if len(top) == 0:
            continue
        w = signal[top] - signal[top].min() + 1e-6
        picks.append(int(np.random.choice(top, p=w / w.sum())))
        if len(picks) == n:
            break
    return div._best_first(np.array(picks, dtype=int), signal)


def recommend_rows(catalog, targets, variant, cells, counters=None):
    """recommend._coverage_top with the variant's candidates and within-target selection. Row positions."""
    guarded = variant != P
    use_cells = variant == S or (variant == SM and len(targets) == 1)
    candidates = guard.guarded_candidates if guarded else recommend._target_candidates
    shares = recommend._allocate_shares([w for _, w in targets], ev.LIST_SIZE)
    taken = np.zeros(len(catalog), dtype=bool)
    picks = []

    def select(target, signal, available, n):
        if use_cells:
            return cell_select(target, signal, available, n, cells)
        return recommend._sample_best(signal, available, n, div.POOL_MULTIPLIER)

    for (target, _), share in zip(targets, shares):
        member, signal = candidates(catalog, target)
        chosen = select(target, signal, member & ~taken, share)
        taken[chosen] = True
        if counters is not None and len(chosen) < share:
            counters[target] = counters.get(target, 0) + share - len(chosen)
        picks.append([target, chosen, member, signal])
    missing = ev.LIST_SIZE - int(taken.sum())
    for pick in picks:
        if missing <= 0:
            break
        target, chosen, member, signal = pick
        extra = select(target, signal, member & ~taken, missing)
        taken[extra] = True
        missing -= len(extra)
        pick[1] = div._best_first(np.concatenate([chosen, extra]), signal)
    slots = []
    for t_i, (target, chosen, _, signal) in enumerate(picks):
        for k, row in enumerate(chosen):
            slots.append(((k + 0.5) / len(chosen), t_i, row))
    slots.sort(key=lambda slot: slot[:2])
    return np.array([slot[2] for slot in slots], dtype=int)


# ---- metrics ----

def decompose(rows, z, membership):
    """(within-target diversity, between-target separation or None) for one list."""
    groups = [rows[m[rows]] for m in membership]
    d = np.sqrt(((z[rows][:, None, :] - z[rows][None, :, :]) ** 2).sum(axis=2))
    pos = {r: i for i, r in enumerate(rows.tolist())}
    within, between = [], []
    for gi, g in enumerate(groups):
        idx = [pos[r] for r in g.tolist()]
        for a, b in itertools.combinations(idx, 2):
            within.append(d[a, b])
        for gj in range(gi + 1, len(groups)):
            jdx = [pos[r] for r in groups[gj].tolist()]
            between.extend(d[np.ix_(idx, jdx)].ravel().tolist())
    return float(np.mean(within)), (float(np.mean(between)) if between else None)


def relevance(catalog, lists_by_request):
    """Base-label slots only: classifier agreement and probability."""
    labels = catalog["mood"].to_numpy()
    predicted = catalog["predicted_mood"].to_numpy()
    prob, agree = [], []
    by_target = {}
    for target in LABEL_TARGETS:
        _, proba = recommend._target_candidates(catalog, target)
        vals, agr = [], []
        for name, lists in lists_by_request.items():
            mood, mode = name.split(" / ")
            if target in [t for t, _ in ev.request_targets(mood, mode)]:
                rows = np.concatenate(lists)
                rows = rows[labels[rows] == target]
                vals.append(proba[rows])
                agr.append(predicted[rows] == target)
        v, a = np.concatenate(vals), np.concatenate(agr)
        prob.append(v)
        agree.append(a)
        by_target[target] = {"slots": int(len(v)), "mean_probability": round(float(v.mean()), 4),
                             "p5_probability": round(float(np.percentile(v, 5)), 4),
                             "min_probability": round(float(v.min()), 4),
                             "classifier_agreement": round(float(a.mean()), 4)}
    p, a = np.concatenate(prob), np.concatenate(agree)
    return {"by_target": by_target, "mean_probability": round(float(p.mean()), 4),
            "classifier_agreement": round(float(a.mean()), 4),
            "legacy_mean_probability_>=0.99": bool(p.mean() >= 0.99)}


def typicality(catalog, z, lists, target):
    """Mean distance of served tracks of `target` to its eligible centroid, relative to the eligible mean."""
    member, _ = recommend._target_candidates(catalog, target)
    centre = z[member].mean(axis=0)
    eligible = np.sqrt(((z[member] - centre) ** 2).sum(axis=1)).mean()
    rows = np.concatenate(lists)
    rows = rows[member[rows]]
    return round(float(np.sqrt(((z[rows] - centre) ** 2).sum(axis=1)).mean() / eligible), 4)


def run_variant(catalog, z, variant, cells, seed):
    record, latencies, counters = {}, [], {}

    def recommender(targets, s):
        t0 = time.perf_counter()
        rows = recommend_rows(catalog, targets, variant, cells, counters)
        latencies.append((time.perf_counter() - t0) * 1000)
        record.setdefault((tuple(t for t, _ in targets), s), rows)
        return rows

    cfg = sweep.run_config(catalog, z, recommender, div.POOL_MULTIPLIER, RUNS, seed)
    lists = {}
    for name in cfg["requests"]:
        mood, mode = name.split(" / ")
        key = tuple(t for t, _ in ev.request_targets(mood, mode))
        lists[name] = [record[(key, seed + r)] for r in range(RUNS)]
    return cfg, lists, latencies, counters


def per_run_metrics(catalog, z, lists):
    """Per request: per-run list diversity, within-target diversity, between-target separation."""
    out = {}
    for name, runs in lists.items():
        mood, mode = name.split(" / ")
        membership = [recommend._target_candidates(catalog, t)[0] for t, _ in ev.request_targets(mood, mode)]
        dec = [decompose(rows, z, membership) for rows in runs]
        out[name] = {"list": [ev.intra_list_diversity(z[rows]) for rows in runs],
                     "within": [d[0] for d in dec],
                     "between": [d[1] for d in dec] if dec[0][1] is not None else None}
    return out


def boot(diff, rng):
    diff = np.asarray(diff)
    b = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(BOOTSTRAP)]
    return {"mean": round(float(diff.mean()), 4),
            "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}


def evaluate(catalog, z, seed, cells):
    single, multi = families()
    reference = sweep.random_eligible_reference(catalog, z, RUNS, seed)
    results, metrics = {}, {}
    for variant in VARIANTS:
        cfg, lists, lat, shortfall = run_variant(catalog, z, variant, cells, seed)
        summ = div.summarize(cfg, reference, list(cfg["requests"]))["summary"]
        metrics[variant] = per_run_metrics(catalog, z, lists)
        res = {"summary": summ, "latency": proto.latency_summary(lat),
               "speech_like_share": proto.speech_like(catalog, lists),
               "relevance": relevance(catalog, lists), "shortfall_slots": shortfall, "configurations": {}}
        for fam_name, fam in (("single", single), ("multi", multi)):
            for config, names in fam.items():
                n = names[0]
                req = cfg["requests"][n]
                m = metrics[variant][n]
                res["configurations"][config_name(config)] = {
                    "family": fam_name, "requests": names,
                    "aliases_identical": all(np.array_equal(a, b) for a, b in zip(lists[names[0]], lists[names[-1]])),
                    "list_diversity": round(float(np.mean(m["list"])), 4),
                    "within_target_diversity": round(float(np.mean(m["within"])), 4),
                    "between_target_separation": (round(float(np.mean(m["between"])), 4) if m["between"] else None),
                    "random_eligible_list_diversity": reference["requests"][n]["diversity"],
                    "typicality_by_target": {t: typicality(catalog, z, lists[n], t) for t, _ in config},
                    "unique_tracks_over_runs": req["unique_tracks_over_runs"], "repeat_rate": req["repeat_rate"],
                    "target_consistency": req["target_consistency"],
                    "exact_target_shares_fraction": req["exact_target_shares_fraction"],
                    "mean_ranking_signal_of_served": req["mean_ranking_signal_of_served"],
                }
        results[variant] = res
    rng = np.random.RandomState(0)
    for variant in VARIANTS:
        if variant == REFERENCE:
            continue
        paired = {}
        for fam_name, fam in (("single", single), ("multi", multi)):
            for config, names in fam.items():
                n = names[0]
                a, b = metrics[REFERENCE][n], metrics[variant][n]
                paired[config_name(config)] = {
                    "list_diversity": boot(np.subtract(b["list"], a["list"]), rng),
                    "within_target_diversity": boot(np.subtract(b["within"], a["within"]), rng)}
        fam_mean = {}
        for fam_name, fam, key in (("single", single, "within"), ("multi", multi, "list")):
            per_run = np.mean([np.subtract(metrics[variant][ns[0]][key], metrics[REFERENCE][ns[0]][key])
                               for ns in fam.values()], axis=0)
            fam_mean[fam_name] = boot(per_run, rng)
        results[variant]["paired_vs_reference"] = paired
        results[variant]["family_mean_change_vs_reference"] = fam_mean
    prod_speech = results[P]["speech_like_share"]["calm_requests"]
    for variant in VARIANTS:
        results[variant]["gates"] = gate_status(results[variant], prod_speech, single, multi) \
            if variant != REFERENCE else None
    return results


def gate_status(res, prod_speech, single, multi):
    g, s = GATES, res["summary"]
    paired = res.get("paired_vs_reference")
    single_names = [config_name(c) for c in single]
    multi_names = [config_name(c) for c in multi]
    st = {
        "integrity": s["target_consistency"] == 1.0 and s["exact_target_shares_all_lists"]
                     and s["all_same_seed_identical"],
        "single_family_within_gain_lb_>0":
            res["family_mean_change_vs_reference"]["single"]["ci95"][0] > g["single_family_mean_gain_lower_bound_gt"],
        ">=4_of_6_single_configs_within_gain_lb_>0":
            sum(paired[n]["within_target_diversity"]["ci95"][0] > 0 for n in single_names)
            >= g["single_family_configs_with_lower_bound_gt_0_min"],
        "no_single_config_regression":
            all(paired[n]["within_target_diversity"]["ci95"][1] >= g["single_config_regression_upper_bound_lt"]
                for n in single_names),
        "multi_family_list_diversity_noninferior_lb_>-0.10":
            res["family_mean_change_vs_reference"]["multi"]["ci95"][0] > g["multi_family_mean_change_lower_bound_gt"],
        "each_multi_config_list_diversity_lb_>-0.25":
            all(paired[n]["list_diversity"]["ci95"][0] > g["multi_config_change_lower_bound_gt"] for n in multi_names),
        "classifier_agreement_>=0.95":
            res["relevance"]["classifier_agreement"] >= g["min_classifier_agreement_base_label_slots"],
        "mean_classifier_probability_>=0.95":
            res["relevance"]["mean_probability"] >= g["min_mean_classifier_probability_base_label_slots"],
        "coverage_>=2000": s["catalog_unique_tracks"] >= g["min_catalog_unique_tracks"],
        "repeat_rate_<=0.75": s["mean_repeat_rate"] <= g["max_mean_repeat_rate"],
        "p95_latency_<=25ms": s["latency_ms"]["p95"] <= g["max_latency_p95_ms"],
        "calm_speech_like_<=0.20_and_<=production":
            res["speech_like_share"]["calm_requests"] <= g["max_calm_speech_like_share"]
            and res["speech_like_share"]["calm_requests"] <= prod_speech,
    }
    return {k: "PASS" if v else "FAIL" for k, v in st.items()}


# ---- measurement validity (static facts, plus figures filled from this run) ----

MEASUREMENT_VALIDITY = {
    "measured_facts": [
        "Which tracks are served, their dataset label / derived-rule membership, exact target shares, repeat "
        "rate, unique-track coverage, determinism and latency in this harness.",
        "Audio-feature values (Spotify-derived) and the classifier's probabilities for served tracks.",
    ],
    "proxies": [
        "Feature-space diversity (mean pairwise distance in 9 z-scored audio features) is a proxy for variety; "
        "it is not listener-perceived variety and is inflated by outliers such as speech-like tracks.",
        "Classifier probability and agreement are proxies for mood relevance; the classifier was trained on the "
        "same model-generated labels and the ranking optimises its probability, so they are not independent.",
        "Target consistency is circular: candidates are filtered by the same label/rule that defines relevance.",
        "Typicality (distance to the target's eligible centroid) describes where in the mood's feature "
        "distribution picks come from; whether central or extreme tracks are 'better' is not measured.",
    ],
    "assumptions": [
        "The Moodify labels (model-generated, CC BY 4.0) are an acceptable definition of each base mood.",
        "Angry and stressed percentile rules define those moods (stressed is experimental).",
        "More within-mood variety at constant relevance is desirable; no user data confirms this.",
    ],
    "not_measured": "No human relevance judgments, listening data, skips, likes or A/B outcomes exist for this "
                    "catalog; nothing here measures listener satisfaction.",
}


# ---- cold start ----

def cold_start_probe(variant):
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    t0 = time.perf_counter()
    cells = build_cells(catalog, z, guarded=True) if variant in (S, SM) else None
    build_ms = (time.perf_counter() - t0) * 1000
    targets = ev.request_targets("sad", "match")
    times = []
    for i in range(10):
        np.random.seed(i)
        t1 = time.perf_counter()
        recommend_rows(catalog, targets, variant, cells)
        times.append((time.perf_counter() - t1) * 1000)
    print(json.dumps({"cell_build_ms": build_ms, "first_ms": times[0], "second_ms": times[1], "tenth_ms": times[9]}))


def cold_start():
    out = {}
    for variant in VARIANTS:
        rows = []
        for _ in range(COLD_START_PROCESSES):
            res = subprocess.run([sys.executable, os.path.abspath(__file__), "--cold-start-probe", variant],
                                 capture_output=True, text=True, check=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__)))
            rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
        out[variant] = {k: round(float(np.median([r[k] for r in rows])), 2)
                        for k in ("cell_build_ms", "first_ms", "second_ms", "tenth_ms")}
    return out


def main():
    parser = argparse.ArgumentParser(description="Balanced recommendation-quality analysis (analysis only)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--cold-start-probe", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold_start_probe:
        cold_start_probe(args.cold_start_probe)
        return
    if not seeds_disjoint():
        raise SystemExit("Seed sets overlap seeds used earlier")

    t0 = time.perf_counter()
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    hashes_before = {"catalog": mp.file_sha256(MUSIC_CATALOG_278K_PATH), "classifier": mp.file_sha256(CLASSIFIER_PATH)}
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(PRIMARY_SEED)
        a = recommend._coverage_top(catalog, targets, ev.LIST_SIZE).index.to_numpy()
        np.random.seed(PRIMARY_SEED)
        if not np.array_equal(a, recommend_rows(catalog, targets, P, None)):
            raise SystemExit(f"Production replica diverges for {mood}/{mode}")
    cells = build_cells(catalog, z, guarded=True)

    primary = evaluate(catalog, z, PRIMARY_SEED, cells)
    replication = evaluate(catalog, z, REPLICATION_SEED, cells)
    for tag, res in (("primary", primary), ("replication", replication)):
        for v, r in res.items():
            s = r["summary"]
            fm = r.get("family_mean_change_vs_reference") or {}
            print(f"[{tag}] {v:>55}: cov {s['catalog_unique_tracks']} rep {s['mean_repeat_rate']:.3f} "
                  f"agree {r['relevance']['classifier_agreement']} prob {r['relevance']['mean_probability']} "
                  f"p95 {s['latency_ms']['p95']} speech {r['speech_like_share']['calm_requests']:.3f} "
                  f"single-within {fm.get('single')} multi-list {fm.get('multi')} "
                  f"fails {[k for k, x in (r['gates'] or {}).items() if x != 'PASS']}")
    cold = cold_start()
    hashes_after = {"catalog": mp.file_sha256(MUSIC_CATALOG_278K_PATH), "classifier": mp.file_sha256(CLASSIFIER_PATH)}
    if hashes_after != hashes_before:
        raise SystemExit("A source artifact changed during the analysis -- this must never happen.")

    single, multi = families()
    report = {
        "analysis_only": True,
        "evaluation_date": datetime.date.today().isoformat(),
        "script_sha256": script_sha,
        "source_artifact_sha256": hashes_before,
        "production_configuration": {"pool_multiplier": div.POOL_MULTIPLIER, "weighting": "current"},
        "hypothesis": __doc__.split("Hypothesis")[1].split("Strategies")[0].strip(),
        "preregistration": PREREGISTRATION,
        "gates": GATES,
        "measurement_validity": MEASUREMENT_VALIDITY,
        "seeds": {"primary": f"{PRIMARY_SEED}..{PRIMARY_SEED + RUNS - 1}",
                  "replication": f"{REPLICATION_SEED}..{REPLICATION_SEED + RUNS - 1}",
                  "previously_used": {k: f"{a}..{b}" for k, (a, b) in USED_SEEDS.items()},
                  "disjoint": seeds_disjoint(), "runs_per_request": RUNS, "bootstrap_rng": 0},
        "configurations": {"single_target": {config_name(c): n for c, n in single.items()},
                           "multi_target": {config_name(c): n for c, n in multi.items()}},
        "cell_design": {"cells_per_target": PC1_BINS * PC2_BINS, "pc1_bins": PC1_BINS, "pc2_bins": PC2_BINS,
                        "cell_top": CELL_TOP,
                        "eligible_per_target": {t: int((v["labels"] >= 0).sum()) for t, v in cells.items()},
                        "cell_size_range": {t: [int(min(len(c) for c in v["by_cell"])),
                                                int(max(len(c) for c in v["by_cell"]))] for t, v in cells.items()}},
        "primary": primary,
        "replication": replication,
        "cold_start_median_of_5_processes": cold,
        "notes": [
            "Diversity is feature-space diversity, not listener-perceived variety.",
            "Bootstrap intervals describe sampling variability over 50 seeded runs on one catalog only.",
            "Latency is machine-dependent; variants ran sequentially in one process. S's cells are built once per "
            "catalog (cell_build_ms in cold start), not per request.",
        ],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
