"""
ANALYSIS ONLY -- why single-target music lists are less diverse than random
eligible tracks, and whether a different within-pool selection helps. Nothing
here changes production: the app still calls src.recommend.recommend_music
(pool_multiplier=20, classifier-signal weighting).

Run from the project root:
    python evaluate_music_diversity_analysis.py [--runs N] [--output PATH]

Reads   data/music_catalog_278k.csv and models/music_mood_classifier_278k.pkl
        (both read-only; the classifier is only re-applied to recover the
        unrounded probabilities for the tie counterfactual)
Writes  models/music_diversity_analysis.json (aggregate metrics only: no track
        IDs, titles, artists or catalog rows). The recommender eval and sweep
        artifacts are not touched.

Part 1, root cause: per single-target target, measures signal resolution and
ties, how concentrated the confidence-ranked candidate pool is in the
standardized audio-feature space, how much the sampling weights add, and a
no-ties counterfactual using unrounded classifier probabilities.

Part 2, strategies: every strategy keeps production's eligibility, exact
per-target shares, candidate pool (top pool_multiplier x share by signal) and
interleaving; only the choice of `share` tracks inside the pool differs.
  A  production       recommend._sample_best (weights = s - min(pool s) + 1e-6)
  B  uniform          uniform without replacement from the same pool
  C  mmr(lambda)      Maximal Marginal Relevance in the pool (see mmr_select)
  C' mmr_sub(lambda)  MMR on a production-weighted random sub-pool of 3 x share
  D  stratified       k-means the pool into `share` clusters, one production-
                      weighted pick per cluster
The selection rule and acceptance gates (SELECTION_RULE) are fixed in code
before any result is seen. They are applied to the exploratory seeds; the
chosen strategy is then re-run, with production, on disjoint held-out seeds.
"""
import argparse
import datetime
import inspect
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_music_recommender_analysis as sweep
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend

OUTPUT_PATH = os.path.join(MODELS_DIR, "music_diversity_analysis.json")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "music_mood_classifier_278k.pkl")
POOL_MULTIPLIER = inspect.signature(recommend._coverage_top).parameters["pool_multiplier"].default
EXPLORATORY_SEED = 2026    # same seeds as evaluate_music_recommender.py (2026 .. 2026 + runs - 1)
HELD_OUT_SEED = 7000       # disjoint confirmation seeds (7000 .. 7000 + runs - 1)
MMR_LAMBDAS = (0.3, 0.5, 0.7, 0.9)
MMR_SUBPOOL = 3
TOP_K_CURVE = (90, 600, 2400, 9600)
BOOTSTRAP = 2000

# Fixed before results. Gates reuse the Phase 2H.6 acceptance thresholds.
SELECTION_RULE = {
    "gates": {"target_consistency": 1.0, "exact_target_shares_all_lists": True, "all_same_seed_identical": True,
              "min_mean_ranking_signal": 0.99, "max_mean_repeat_rate": 0.75, "max_latency_p95_ms": 25.0},
    "objective": "maximize mean diversity over the single-target requests (exploratory seeds)",
    "tie_break": "higher mean ranking signal",
    "candidates": "every strategy except production",
}


# ---- replica of production with a pluggable within-pool selector ----

def candidate_pool(signal, available, n, pool_multiplier):
    """Exactly _sample_best's pool: the best n * pool_multiplier available rows by signal."""
    idx = np.flatnonzero(available)
    if n <= 0 or len(idx) == 0:
        return np.array([], dtype=int)
    pool_size = min(len(idx), n * pool_multiplier)
    if pool_size < len(idx):
        idx = idx[np.argpartition(-signal[idx], pool_size - 1)[:pool_size]]
    return idx


def _best_first(rows, signal):
    return rows[np.argsort(-signal[rows], kind="stable")]


def _production_weights(s):
    w = s - s.min() + 1e-6
    return w / w.sum()


def select_production(signal, available, n, pool_multiplier, z):
    return recommend._sample_best(signal, available, n, pool_multiplier)


def select_uniform(signal, available, n, pool_multiplier, z):
    idx = candidate_pool(signal, available, n, pool_multiplier)
    if len(idx) > n:
        idx = np.random.choice(idx, size=n, replace=False)
    return _best_first(idx, signal)


def mmr_select(signal, available, n, pool_multiplier, z, lam, subpool=None):
    """
    Greedy MMR inside the production pool.
      relevance  r_i = (s_i - min s) / (max s - min s) over the pool (1 if flat):
                 the same "linear above the pool minimum" shape as production's weights.
      redundancy max_{j chosen} sim_ij, sim_ij = 1 - d_ij / d_ref, d = Euclidean distance in the
                 catalog-standardized 9 audio features (same space as the diversity metric),
                 d_ref = 2 * max distance from the pool centroid (an upper bound on any pairwise
                 distance in the pool, so sim is in [0, 1]); distances are computed only from
                 chosen tracks, n x pool instead of pool x pool.
      next pick  argmax  lam * r_i - (1 - lam) * redundancy_i  (lowest position on ties).
    The first pick is drawn with production's weights, so lists vary across seeds and are
    reproducible for a fixed np.random seed. subpool=k first draws k * n pool candidates with
    production's weights and runs MMR on those (more variety across runs).
    """
    idx = candidate_pool(signal, available, n, pool_multiplier)
    if len(idx) <= n:
        return _best_first(idx, signal)
    if subpool and len(idx) > subpool * n:
        idx = np.random.choice(idx, size=subpool * n, replace=False, p=_production_weights(signal[idx]))
    s = signal[idx]
    span = s.max() - s.min()
    rel = (s - s.min()) / span if span > 0 else np.ones(len(idx))
    f = z[idx]
    d_ref = 2 * np.sqrt(((f - f.mean(axis=0)) ** 2).sum(axis=1)).max()

    def sim_to(j):
        return 1 - np.sqrt(((f - f[j]) ** 2).sum(axis=1)) / d_ref if d_ref > 0 else np.ones(len(f))

    first = np.random.choice(len(idx), p=_production_weights(s))
    chosen, redundancy = [first], sim_to(first)
    for _ in range(n - 1):
        score = lam * rel - (1 - lam) * redundancy
        score[chosen] = -np.inf
        j = int(np.argmax(score))
        chosen.append(j)
        redundancy = np.maximum(redundancy, sim_to(j))
    return _best_first(idx[chosen], signal)


def stratified_select(signal, available, n, pool_multiplier, z):
    """k-means (k = n) on the pool's standardized features; one production-weighted pick per cluster."""
    idx = candidate_pool(signal, available, n, pool_multiplier)
    if len(idx) <= n:
        return _best_first(idx, signal)
    labels = KMeans(n_clusters=n, n_init=1, random_state=np.random.randint(2 ** 31 - 1)).fit_predict(z[idx])
    picks = []
    for k in range(n):
        members = idx[labels == k]
        picks.append(np.random.choice(members, p=_production_weights(signal[members])))
    return _best_first(np.array(picks), signal)


def strategies():
    out = {"A: production": select_production, "B: uniform": select_uniform}
    for lam in MMR_LAMBDAS:
        out[f"C: mmr lambda={lam}"] = lambda *a, lam=lam: mmr_select(*a, lam=lam)
    for lam in MMR_LAMBDAS:
        out[f"C': mmr_sub{MMR_SUBPOOL} lambda={lam}"] = lambda *a, lam=lam: mmr_select(*a, lam=lam, subpool=MMR_SUBPOOL)
    out["D: stratified k-means"] = stratified_select
    return out


def coverage_top_selector(df, targets, top_n, pool_multiplier, select, z):
    """recommend._coverage_top with the within-pool selection replaced by `select`.
    Returns row positions in list order."""
    shares = recommend._allocate_shares([w for _, w in targets], top_n)
    taken = np.zeros(len(df), dtype=bool)
    picks = []
    for (target, _), share in zip(targets, shares):
        member, signal = recommend._target_candidates(df, target)
        chosen = select(signal, member & ~taken, share, pool_multiplier, z)
        taken[chosen] = True
        picks.append([target, chosen, member, signal])
    missing = top_n - int(taken.sum())
    for pick in picks:
        if missing <= 0:
            break
        target, chosen, member, signal = pick
        extra = select(signal, member & ~taken, missing, pool_multiplier, z)
        taken[extra] = True
        missing -= len(extra)
        pick[1] = _best_first(np.concatenate([chosen, extra]), signal)
    slots = []
    for t_i, (target, chosen, _, signal) in enumerate(picks):
        for k, row in enumerate(chosen):
            slots.append(((k + 0.5) / len(chosen), t_i, row))
    slots.sort(key=lambda slot: slot[:2])
    return np.array([slot[2] for slot in slots], dtype=int)


# ---- part 1: root cause ----

def single_target_requests():
    out = {}
    for mood in MOODS:
        for mode in ev.MODES:
            targets = ev.request_targets(mood, mode)
            if len(targets) == 1:
                out[f"{mood} / {mode}"] = targets[0][0]
    return out


def _uniform_diversity(rows, z, k, draws, rng):
    return float(np.mean([ev.intra_list_diversity(z[rng.choice(rows, size=k, replace=False)]) for _ in range(draws)]))


def root_cause(catalog, z, unrounded, runs, seed):
    rng = np.random.RandomState(0)
    n, k = ev.LIST_SIZE, ev.LIST_SIZE
    out = {}
    for target in sorted(set(single_target_requests().values())):
        member, signal = recommend._target_candidates(catalog, target)
        idx = np.flatnonzero(member)
        s = signal[idx]
        ranked = idx[np.argsort(-s, kind="stable")]
        desc = np.sort(s)[::-1]
        pool = candidate_pool(signal, member, n, POOL_MULTIPLIER)
        boundary = signal[pool].min()

        served = []
        for r in range(runs):
            np.random.seed(seed + r)
            served.append(recommend._sample_best(signal, member, n, POOL_MULTIPLIER))
        served_rows = np.concatenate(served)

        entry = {
            "eligible_tracks": int(len(idx)),
            "signal": {
                "kind": "classifier probability, stored rounded to 4 dp" if target in recommend._LABEL_BY_NAME
                        else "derived percentile score, stored rounded to 6 dp",
                "distinct_values_among_eligible": int(len(np.unique(s))),
                "max": float(desc[0]),
                "tracks_at_max": int((s == desc[0]).sum()),
                "pool_size": int(len(pool)),
                "pool_signal_range": [float(boundary), float(signal[pool].max())],
                "eligible_tracks_tied_at_pool_boundary": int((s == boundary).sum()),
                "pool_members_at_boundary_value": int((signal[pool] == boundary).sum()),
                "eligible_tracks_at_or_above_boundary": int((s >= boundary).sum()),
            },
            "feature_concentration": {
                "uniform30_diversity_eligible": round(_uniform_diversity(idx, z, k, 200, rng), 4),
                "uniform30_diversity_pool": round(_uniform_diversity(pool, z, k, 200, rng), 4),
                "production_list_diversity": round(float(np.mean([ev.intra_list_diversity(z[l]) for l in served])), 4),
                "pool_std_over_eligible_std": {f: round(float(v), 3) for f, v in
                                               zip(mp.AUDIO_FEATURES, z[pool].std(0) / z[idx].std(0))},
                "served_std_over_eligible_std": {f: round(float(v), 3) for f, v in
                                                 zip(mp.AUDIO_FEATURES, z[served_rows].std(0) / z[idx].std(0))},
                "uniform30_diversity_by_top_k": {
                    str(kk): {"diversity": round(_uniform_diversity(ranked[:kk], z, k, 200, rng), 4),
                              "min_signal": float(desc[kk - 1])}
                    for kk in TOP_K_CURVE if kk <= len(idx)},
            },
            "sampling_weights": {
                "effective_candidates_in_pool": round(float(1 / np.sum(_production_weights(signal[pool]) ** 2)), 1),
                "weight_share_of_best_10pct_of_pool": round(float(np.sort(_production_weights(signal[pool]))[::-1]
                                                                  [:len(pool) // 10].sum()), 4),
            },
        }
        if target in recommend._LABEL_BY_NAME:
            col = mp.PROBA_COLUMNS.index(f"proba_{target}")
            exact = unrounded[:, col]
            pool_u = candidate_pool(exact, member, n, POOL_MULTIPLIER)
            served_u = []
            for r in range(runs):
                np.random.seed(seed + r)
                served_u.append(recommend._sample_best(exact, member, n, POOL_MULTIPLIER))
            entry["no_ties_counterfactual"] = {
                "distinct_unrounded_values_among_eligible": int(len(np.unique(exact[idx]))),
                "pool_overlap_with_rounded_pool": round(len(set(pool_u) & set(pool)) / len(pool), 4),
                "uniform30_diversity_pool": round(_uniform_diversity(pool_u, z, k, 200, rng), 4),
                "production_selection_diversity": round(float(np.mean([ev.intra_list_diversity(z[l])
                                                                       for l in served_u])), 4),
            }
        out[target] = entry
    return out


# ---- part 2: strategy comparison ----

def run_strategy(catalog, z, select, runs, seed):
    record = {}

    def recommender(targets, s):
        rows = coverage_top_selector(catalog, targets, ev.LIST_SIZE, POOL_MULTIPLIER, select, z)
        record.setdefault((tuple(t for t, _ in targets), s), rows)
        return rows

    cfg = sweep.run_config(catalog, z, recommender, POOL_MULTIPLIER, runs, seed)
    per_run = {}
    for name in cfg["requests"]:
        mood, mode = name.split(" / ")
        key = tuple(t for t, _ in ev.request_targets(mood, mode))
        per_run[name] = [ev.intra_list_diversity(z[record[(key, seed + r)]]) for r in range(runs)]
    return cfg, per_run


def summarize(cfg, reference, single):
    reqs = cfg["requests"]
    s = cfg["summary"]
    per = {name: {"diversity": r["metrics"]["diversity"],
                  "random_eligible_diversity": reference["requests"][name]["diversity"],
                  "diversity_minus_random_eligible": round(r["metrics"]["diversity"]
                                                           - reference["requests"][name]["diversity"], 4),
                  "unique_tracks_over_runs": r["unique_tracks_over_runs"],
                  "repeat_rate": r["repeat_rate"],
                  "mean_pairwise_jaccard": r["mean_pairwise_jaccard"],
                  "mean_ranking_signal_of_served": r["mean_ranking_signal_of_served"],
                  "target_consistency": r["target_consistency"],
                  "exact_target_shares_fraction": r["exact_target_shares_fraction"],
                  "same_seed_identical": r["same_seed_identical"]}
           for name, r in reqs.items()}
    return {
        "summary": {
            "target_consistency": s["target_consistency"],
            "exact_target_shares_all_lists": s["exact_target_shares_all_lists"],
            "all_same_seed_identical": s["all_same_seed_identical"],
            "catalog_unique_tracks": s["catalog_unique_tracks"],
            "mean_repeat_rate": s["mean_repeat_rate"],
            "mean_diversity": s["mean_diversity"],
            "single_target_mean_diversity": round(float(np.mean([per[n]["diversity"] for n in single])), 4),
            "single_target_requests_at_least_random_eligible": sum(per[n]["diversity_minus_random_eligible"] >= 0
                                                                   for n in single),
            "requests_at_least_random_eligible": sum(p["diversity_minus_random_eligible"] >= 0 for p in per.values()),
            "mean_ranking_signal_of_served": s["mean_ranking_signal_of_served"],
            "single_target_mean_ranking_signal": round(float(np.mean([per[n]["mean_ranking_signal_of_served"]
                                                                      for n in single])), 4),
            "latency_ms": s["latency_ms"],
        },
        "requests": per,
    }


def paired(per_run_a, per_run_b, names, rng):
    """Per request: mean of (b - a) over runs that share a seed, with a percentile bootstrap 95% interval."""
    out = {}
    for name in names:
        diff = np.asarray(per_run_b[name]) - np.asarray(per_run_a[name])
        boots = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(BOOTSTRAP)]
        out[name] = {"mean_diff": round(float(diff.mean()), 4),
                     "bootstrap_95": [round(float(np.percentile(boots, 2.5)), 4),
                                      round(float(np.percentile(boots, 97.5)), 4)],
                     "runs_where_b_more_diverse": int((diff > 0).sum())}
    return out


def passes_gates(summary):
    g = SELECTION_RULE["gates"]
    checks = {
        "target_consistency": summary["target_consistency"] >= g["target_consistency"],
        "exact_target_shares_all_lists": summary["exact_target_shares_all_lists"] is True,
        "all_same_seed_identical": summary["all_same_seed_identical"] is True,
        "min_mean_ranking_signal": summary["mean_ranking_signal_of_served"] >= g["min_mean_ranking_signal"],
        "max_mean_repeat_rate": summary["mean_repeat_rate"] <= g["max_mean_repeat_rate"],
        "max_latency_p95_ms": summary["latency_ms"]["p95"] <= g["max_latency_p95_ms"],
    }
    return all(checks.values()), checks


def choose(results):
    eligible = []
    for name, res in results.items():
        if name == "A: production":
            continue
        ok, _ = passes_gates(res["summary"])
        if ok:
            eligible.append((res["summary"]["single_target_mean_diversity"],
                             res["summary"]["mean_ranking_signal_of_served"], name))
    return max(eligible)[2] if eligible else None


def main():
    parser = argparse.ArgumentParser(description="Single-target diversity analysis (analysis only)")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    t0 = time.perf_counter()
    catalog_hash = mp.file_sha256(MUSIC_CATALOG_278K_PATH)
    classifier_hash = mp.file_sha256(CLASSIFIER_PATH)
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    with open(CLASSIFIER_PATH, "rb") as f:
        bundle = pickle.load(f)
    unrounded = mp.predict_label_proba(catalog, bundle["scaler"], bundle["model"])
    if not all(np.array_equal(np.round(unrounded[:, i], 4), catalog[c].to_numpy())
               for i, c in enumerate(mp.PROBA_COLUMNS)):
        raise SystemExit("Re-applied classifier does not reproduce the catalog probabilities")

    # the replica with production selection must reproduce production lists exactly
    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        for s in range(3):
            np.random.seed(EXPLORATORY_SEED + s)
            a = recommend._coverage_top(catalog, targets, ev.LIST_SIZE).index.to_numpy()
            np.random.seed(EXPLORATORY_SEED + s)
            b = coverage_top_selector(catalog, targets, ev.LIST_SIZE, POOL_MULTIPLIER, select_production, z)
            if not np.array_equal(a, b):
                raise SystemExit(f"Replica diverges from production for {mood}/{mode}")
    print("Replica fidelity check passed")

    single = list(single_target_requests())
    causes = root_cause(catalog, z, unrounded, args.runs, EXPLORATORY_SEED)
    for t, e in causes.items():
        fc = e["feature_concentration"]
        print(f"{t:>9}: eligible div {fc['uniform30_diversity_eligible']:.3f} | pool uniform "
              f"{fc['uniform30_diversity_pool']:.3f} | production {fc['production_list_diversity']:.3f} | "
              f"ties@boundary {e['signal']['eligible_tracks_tied_at_pool_boundary']}"
              + (f" | no-ties production {e['no_ties_counterfactual']['production_selection_diversity']:.3f}"
                 if "no_ties_counterfactual" in e else ""))

    rng = np.random.RandomState(0)
    reference = sweep.random_eligible_reference(catalog, z, args.runs, EXPLORATORY_SEED)
    exploratory, per_run = {}, {}
    for name, select in strategies().items():
        cfg, runs_div = run_strategy(catalog, z, select, args.runs, EXPLORATORY_SEED)
        exploratory[name] = summarize(cfg, reference, single)
        per_run[name] = runs_div
        s = exploratory[name]["summary"]
        print(f"{name:>28}: single-target div {s['single_target_mean_diversity']:.3f} "
              f"(>=RE {s['single_target_requests_at_least_random_eligible']}/{len(single)}) mean div "
              f"{s['mean_diversity']:.3f} cov {s['catalog_unique_tracks']} repeat {s['mean_repeat_rate']:.3f} "
              f"signal {s['mean_ranking_signal_of_served']:.4f} p95 {s['latency_ms']['p95']}ms "
              f"gates {passes_gates(s)[0]}")
    for name in exploratory:
        exploratory[name]["gates"] = passes_gates(exploratory[name]["summary"])[1]
        if name != "A: production":
            exploratory[name]["paired_vs_production"] = paired(per_run["A: production"], per_run[name], single, rng)

    chosen = choose(exploratory)
    print(f"Chosen by the pre-registered rule: {chosen}")
    held_out, runs_h = None, {}
    if chosen:
        ref_h = sweep.random_eligible_reference(catalog, z, args.runs, HELD_OUT_SEED)
        held_out = {"seed": HELD_OUT_SEED, "strategies": {}}
        for name in ["A: production", chosen]:
            cfg, runs_h[name] = run_strategy(catalog, z, strategies()[name], args.runs, HELD_OUT_SEED)
            held_out["strategies"][name] = summarize(cfg, ref_h, single)
            held_out["strategies"][name]["gates"] = passes_gates(held_out["strategies"][name]["summary"])[1]
        held_out["paired_vs_production"] = paired(runs_h["A: production"], runs_h[chosen], single, rng)
        s = held_out["strategies"][chosen]["summary"]
        print(f"held-out {chosen}: single-target div {s['single_target_mean_diversity']:.3f} vs production "
              f"{held_out['strategies']['A: production']['summary']['single_target_mean_diversity']:.3f}")

    # Post hoc, NOT used for selection: held-out runs for every strategy that beat production's
    # single-target diversity on the exploratory seeds, gates or not, to show whether gains replicate.
    base = exploratory["A: production"]["summary"]["single_target_mean_diversity"]
    post_hoc = {"note": "post hoc robustness check on the held-out seeds; not part of the selection rule",
                "strategies": {}}
    ref_h = sweep.random_eligible_reference(catalog, z, args.runs, HELD_OUT_SEED)
    prod_h = held_out["strategies"]["A: production"] if held_out else None
    if prod_h is None:
        cfg, prod_runs_h = run_strategy(catalog, z, select_production, args.runs, HELD_OUT_SEED)
        prod_h = summarize(cfg, ref_h, single)
    else:
        prod_runs_h = runs_h["A: production"]
    post_hoc["strategies"]["A: production"] = prod_h
    for name, res in exploratory.items():
        if name == "A: production" or res["summary"]["single_target_mean_diversity"] <= base:
            continue
        cfg, runs_n = run_strategy(catalog, z, strategies()[name], args.runs, HELD_OUT_SEED)
        post_hoc["strategies"][name] = summarize(cfg, ref_h, single)
        post_hoc["strategies"][name]["gates"] = passes_gates(post_hoc["strategies"][name]["summary"])[1]
        post_hoc["strategies"][name]["paired_vs_production"] = paired(prod_runs_h, runs_n, single, rng)
        s = post_hoc["strategies"][name]["summary"]
        print(f"post-hoc held-out {name:>28}: single-target div {s['single_target_mean_diversity']:.3f} "
              f"repeat {s['mean_repeat_rate']:.3f} p95 {s['latency_ms']['p95']}ms")

    if mp.file_sha256(MUSIC_CATALOG_278K_PATH) != catalog_hash or mp.file_sha256(CLASSIFIER_PATH) != classifier_hash:
        raise SystemExit("The catalog or classifier changed during the analysis -- this must never happen.")
    report = {
        "analysis_only": True,
        "evaluation_date": datetime.date.today().isoformat(),
        "production_configuration": {"pool_multiplier": POOL_MULTIPLIER, "weighting": "current"},
        "catalog": {"path": os.path.relpath(MUSIC_CATALOG_278K_PATH, os.path.dirname(MODELS_DIR)),
                    "sha256": catalog_hash, "rows": int(len(catalog))},
        "classifier_sha256": classifier_hash,
        "runs_per_request": args.runs, "list_size": ev.LIST_SIZE,
        "seeds": {"exploratory": f"{EXPLORATORY_SEED}..{EXPLORATORY_SEED + args.runs - 1}",
                  "held_out": f"{HELD_OUT_SEED}..{HELD_OUT_SEED + args.runs - 1}"},
        "single_target_requests": single_target_requests(),
        "diversity_definition": "mean pairwise Euclidean distance between a list's tracks in the 9 audio features, "
                                "each z-scored over the catalog (feature-space diversity, not perceived diversity)",
        "strategy_definitions": {
            "A: production": "recommend._sample_best: weighted sample, w = s - min(pool s) + 1e-6",
            "B: uniform": "uniform sample without replacement from the same pool",
            "C: mmr": "greedy MMR in the pool; relevance = min-max signal over the pool; redundancy = max over chosen "
                      "of 1 - d / d_ref, d_ref = 2 x max distance to the pool centroid; first pick production-weighted",
            "C': mmr_sub": f"MMR on a production-weighted random sub-pool of {MMR_SUBPOOL} x share",
            "D: stratified k-means": "k-means (k = share, n_init=1, seeded) on the pool; one production-weighted "
                                     "pick per cluster",
            "shared": "all strategies keep eligibility, exact per-target shares, the top pool_multiplier x share "
                      "pool and production's interleaving",
        },
        "selection_rule": SELECTION_RULE,
        "root_cause": causes,
        "random_eligible_reference": reference,
        "exploratory": exploratory,
        "chosen_on_exploratory": chosen,
        "held_out_confirmation": held_out,
        "post_hoc_held_out": post_hoc,
        "notes": [
            "Precision/NDCG are omitted: relevance is the eligibility definition itself, so they are 1.0 for every "
            "strategy here and cannot rank strategies.",
            "mean_ranking_signal_of_served is the classifier probability (label targets) or derived score "
            "(angry/stressed): the ranking's own signal, not an independent quality measure.",
            "Bootstrap intervals resample the runs of one request; runs share the same candidate pool, so they "
            "describe sampling variability only, not variation across users or catalogs. No significance is claimed.",
            "Latency is the whole replica call in this harness on this machine.",
        ],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
