"""
ANALYSIS ONLY -- Phase 2I: does redesigning candidate-pool formation fix the
happy/match and sad/match diversity problem? Nothing here changes production:
the app still calls src.recommend.recommend_music (top 20 x share pool).

Run from the project root:
    python evaluate_music_pool_redesign.py [--output PATH]

Reads   data/music_catalog_278k.csv (read-only)
Writes  models/music_pool_redesign_analysis.json (aggregate metrics only: no
        track IDs, titles, artists or catalog rows). No earlier report is touched.

Part 1 (root cause, catalog only, no evaluation seeds): per label target, the
signal distribution, diversity of uniform random 30-track lists from top-K
pools, from same-size random subsets, from confidence thresholds and from
confidence deciles.

Part 2 (pool strategies): only pool formation changes. Selection inside the
pool is production's (weights s - min(pool s) + 1e-6, sampled without
replacement), shares, eligibility, interleaving and metrics are unchanged.
  A   top 20 x share                 (production, the control; verified identical)
  B1  top 40 x share                 (1,200 for a 30-slot target)
  B2  top 80 x share                 (2,400)
  C   threshold tau in THRESHOLDS    label targets (sad/happy/energetic/calm): every
                                     available candidate with signal >= tau; if that is
                                     smaller than the production pool, the production pool
                                     is used instead (counted as a fallback), so a threshold
                                     can only widen the pool. Derived targets (angry,
                                     stressed) keep the production pool.
  D   hybrid tau=0.95, cap 160 x share (4,800): C's pool cut to its best cap by signal,
                                     bounding work for very large threshold pools.
  +G  calm speech guard (speechiness > 0.66 excluded from calm candidates only), as a
      separate constraint on A and on the strategy chosen from the exploratory seeds.
The strategies, thresholds, gates, selection rule and seed sets below were
fixed from the catalog-only Part 1 numbers before any evaluation seed was run.
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
import evaluate_music_phase_2h9 as h9
import evaluate_music_stratified_prototype as proto
from src.config import MODELS_DIR, MOODS, MUSIC_CATALOG_278K_PATH
from src import music_evaluation as ev
from src import music_pipeline as mp
from src import recommend

OUTPUT_PATH = os.path.join(MODELS_DIR, "music_pool_redesign_analysis.json")
RUNS = 50
EXPLORATORY_SEED = 2026          # parameter selection (as in earlier phases)
HELD_OUT_SEED = 18000            # new; never used for selection
USED_ELSEWHERE = {"tests and cold-start probes": (0, 49), "Phase 2H.6 API test": (2024, 2024),
                  "2H.7-2H.8 held-out": (7000, 7049), "2H.8.1 third set": (12000, 12049),
                  "2H.9 fourth set": (15000, 15049)}
PRODUCTION_MULTIPLIER = div.POOL_MULTIPLIER      # 20
THRESHOLDS = (0.99, 0.95, 0.90)
LABEL_TARGETS = tuple(recommend._LABEL_BY_NAME)  # sad, happy, energetic, calm
TOP_K_CURVE = (90, 600, 1200, 2400, 4800, 9600, 19200)
CURVE_THRESHOLDS = (0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8)
COLD_START_PROCESSES = 5

STRATEGIES = {
    "A: top 20x (production)": {"kind": "topn", "multiplier": 20},
    "B1: top 40x": {"kind": "topn", "multiplier": 40},
    "B2: top 80x": {"kind": "topn", "multiplier": 80},
    **{f"C: threshold {t}": {"kind": "threshold", "tau": t, "cap_multiplier": None} for t in THRESHOLDS},
    "D: threshold 0.95, cap 160x": {"kind": "threshold", "tau": 0.95, "cap_multiplier": 160},
}
CONTROL = "A: top 20x (production)"
SELECTION_RULE = ("Among B1, B2, C and D: the strategy passing every gate on the exploratory seeds, highest "
                  "unique-configuration mean diversity first; if none passes every gate, none is chosen and the "
                  "best by unique-configuration mean diversity among those passing the signal, repeat-rate and "
                  "p95 gates is carried to held-out as the guarded candidate. All strategies are also re-run on "
                  "the held-out seeds for transparency; held-out results never feed back into selection.")
GATES = h9.GATES                 # the Phase 2H.9 gates, unchanged


def held_out_disjoint():
    lo, hi = HELD_OUT_SEED, HELD_OUT_SEED + RUNS - 1
    used = {**USED_ELSEWHERE, "exploratory": (EXPLORATORY_SEED, EXPLORATORY_SEED + RUNS - 1)}
    return all(hi < a or lo > b for a, b in used.values())


# ---- pool formation ----

def pool_rows(target, signal, available, n, spec):
    """Candidate pool (row positions) for one target, plus whether a threshold fell back."""
    if n <= 0 or not available.any():
        return np.array([], dtype=int), False
    production = div.candidate_pool(signal, available, n, PRODUCTION_MULTIPLIER)
    if spec["kind"] == "topn":
        return div.candidate_pool(signal, available, n, spec["multiplier"]), False
    if target not in LABEL_TARGETS:
        return production, False
    idx = np.flatnonzero(available & (signal >= spec["tau"]))
    if len(idx) < len(production):
        return production, True
    cap = spec["cap_multiplier"] and n * spec["cap_multiplier"]
    if cap and len(idx) > cap:
        idx = idx[np.argpartition(-signal[idx], cap - 1)[:cap]]
    return idx, False


def sample_pool(idx, signal, n):
    """Production's selection inside a pool: weighted by s - min(pool s) + 1e-6, best-first."""
    if len(idx) > n:
        w = signal[idx] - signal[idx].min() + 1e-6
        idx = np.random.choice(idx, size=n, replace=False, p=w / w.sum())
    return div._best_first(idx, signal)


def coverage_top_pool(df, targets, top_n, spec, guarded=False, counters=None):
    """recommend._coverage_top with pool formation from `spec` (and the calm guard if `guarded`).
    `counters` (dict) receives fallbacks, shortfall slots and pool sizes. Returns row positions."""
    candidates = guard.guarded_candidates if guarded else recommend._target_candidates
    shares = recommend._allocate_shares([w for _, w in targets], top_n)
    taken = np.zeros(len(df), dtype=bool)
    picks = []

    def select(target, signal, available, n):
        if spec == STRATEGIES[CONTROL]:
            return recommend._sample_best(signal, available, n, PRODUCTION_MULTIPLIER)
        idx, fell_back = pool_rows(target, signal, available, n, spec)
        if counters is not None:
            counters.setdefault("fallbacks", {}).setdefault(target, 0)
            counters["fallbacks"][target] += int(fell_back)
            counters.setdefault("pool_sizes", {}).setdefault(target, []).append(len(idx))
        return sample_pool(idx, signal, n)

    for (target, _), share in zip(targets, shares):
        member, signal = candidates(df, target)
        chosen = select(target, signal, member & ~taken, share)
        taken[chosen] = True
        if counters is not None and len(chosen) < share:
            counters.setdefault("shortfall", {}).setdefault(target, 0)
            counters["shortfall"][target] += share - len(chosen)
        picks.append([target, chosen, member, signal])
    missing = top_n - int(taken.sum())
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


# ---- part 1: root cause (catalog only) ----

def root_cause(catalog, z):
    rng = np.random.RandomState(0)

    def uniform_div(rows, draws=150):
        return round(float(np.mean([ev.intra_list_diversity(z[rng.choice(rows, 30, replace=False)])
                                    for _ in range(draws)])), 4)

    out = {}
    for target in LABEL_TARGETS:
        member, signal = recommend._target_candidates(catalog, target)
        idx = np.flatnonzero(member)
        v = signal[idx]
        order = idx[np.argsort(-v, kind="stable")]
        ranked = signal[order]
        q = np.quantile(v, np.linspace(0, 1, 11))
        out[target] = {
            "eligible": int(len(idx)),
            "uniform30_diversity_all_eligible": uniform_div(idx),
            "signal_quantiles": {p: round(float(np.quantile(v, float(p))), 4) for p in ("0.1", "0.5", "0.9", "0.99")},
            "top_k": {str(k): {"uniform30_diversity": uniform_div(order[:k]),
                               "random_subset_same_size_diversity": uniform_div(rng.choice(idx, k, replace=False)),
                               "min_signal": round(float(ranked[k - 1]), 4),
                               "mean_signal": round(float(ranked[:k].mean()), 4)}
                      for k in TOP_K_CURVE if k <= len(idx)},
            "threshold": {str(t): {"count": int((v >= t).sum()),
                                   "uniform30_diversity": uniform_div(idx[v >= t]) if (v >= t).sum() >= 30 else None}
                          for t in CURVE_THRESHOLDS},
            "confidence_decile_diversity_low_to_high": [uniform_div(idx[(v >= q[i]) & (v <= q[i + 1])])
                                                        for i in range(10)],
        }
    return out


# ---- part 2: evaluation ----

def variant_names(chosen=None):
    names = list(STRATEGIES) + [f"{CONTROL} + calm guard"]
    if chosen:
        names.append(f"{chosen} + calm guard")
    return names


def variant(name):
    base = name.replace(" + calm guard", "")
    return STRATEGIES[base], name.endswith(" + calm guard")


def served_signal(catalog, lists, configs):
    """Distribution of the ranking signal of served tracks, per unique single-target configuration."""
    out = {}
    for config, names in configs.items():
        target = config[0][0]
        _, signal = recommend._target_candidates(catalog, target)
        s = signal[np.concatenate(lists[names[0]])]
        out[target] = {"mean": round(float(s.mean()), 4), "p5": round(float(np.percentile(s, 5)), 4),
                       "p50": round(float(np.median(s)), 4), "min": round(float(s.min()), 4)}
    return out


def evaluate(catalog, z, seed, names):
    configs = h9.unique_single_target_configurations()
    single = [n for ns in configs.values() for n in ns]
    baselines = {"production_eligibility": h9.random_eligible_baseline(catalog, z, seed, RUNS,
                                                                       recommend._target_candidates),
                 "guarded_calm_eligibility": h9.random_eligible_baseline(catalog, z, seed, RUNS,
                                                                         guard.guarded_candidates)}
    reference = {"requests": {n: {"diversity": d} for n, d in baselines["production_eligibility"].items()}}
    results, per_runs = {}, {}
    for name in names:
        spec, guarded = variant(name)
        counters = {}
        cfg, per_runs[name], lists, lat = guard.run(
            catalog, z, lambda t, spec=spec, guarded=guarded: coverage_top_pool(catalog, t, ev.LIST_SIZE, spec,
                                                                                guarded, counters), RUNS, seed)
        res = div.summarize(cfg, reference, single)
        res["latency"] = proto.latency_summary(lat)
        res["speech_like_share"] = proto.speech_like(catalog, lists)
        res["served_signal_by_target"] = served_signal(catalog, lists, configs)
        res["fallbacks"] = counters.get("fallbacks", {})
        res["shortfall_slots"] = counters.get("shortfall", {})
        res["pool_size_by_target"] = {t: {"min": int(min(s)), "median": int(np.median(s)), "max": int(max(s))}
                                      for t, s in counters.get("pool_sizes", {}).items()}
        res["applicable_baseline"] = "guarded_calm_eligibility" if guarded else "production_eligibility"
        results[name] = res
    rng = np.random.RandomState(0)
    prod_speech = results[CONTROL]["speech_like_share"]["calm_requests"]
    for name in names:
        paired = None
        if name != CONTROL:
            paired = div.paired(per_runs[CONTROL], per_runs[name], list(results[name]["requests"]), rng)
            results[name]["paired_vs_production"] = paired
        base = baselines[results[name]["applicable_baseline"]]
        results[name].update(h9.score(results[name], base, configs, paired, prod_speech))
        results[name]["original_gate_failures"] = [k for k, v in results[name]["gates"].items() if v != "PASS"]
        results[name]["gap_to_baseline_closed"] = gap_closed(results[name], results[CONTROL], base, configs)
    return results, baselines


def gap_closed(res, control, baseline, configs):
    """Supplementary (not a gate): share of the control's shortfall to the random-eligible baseline
    that a strategy closes, for configurations where the control is below the baseline."""
    out = {}
    for config, names in configs.items():
        n = names[0]
        gap = baseline[n] - control["requests"][n]["diversity"]
        if gap > 0:
            out[config[0][0]] = round((res["requests"][n]["diversity"] - control["requests"][n]["diversity"]) / gap, 3)
    return out


def choose(results):
    candidates = [n for n in results if n in STRATEGIES and n != CONTROL]
    passing = [n for n in candidates if all(v == "PASS" for v in results[n]["gates"].values())]
    if passing:
        return max(passing, key=lambda n: results[n]["unique_single_target_mean_diversity"]), True
    soft = [n for n in candidates if all(results[n]["gates"][g] == "PASS" for g in
                                         ("ranking_signal_>=0.99", "repeat_rate_<=0.75", "p95_latency_<=25ms"))]
    if not soft:
        return None, False
    return max(soft, key=lambda n: results[n]["unique_single_target_mean_diversity"]), False


# ---- cold start ----

def cold_start_probe(name):
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    spec, guarded = variant(name)
    targets = ev.request_targets("sad", "match")
    times = []
    for i in range(10):
        np.random.seed(i)
        t0 = time.perf_counter()
        coverage_top_pool(catalog, targets, ev.LIST_SIZE, spec, guarded)
        times.append((time.perf_counter() - t0) * 1000)
    print(json.dumps({"first_ms": times[0], "second_ms": times[1], "tenth_ms": times[9]}))


def cold_start(names):
    out = {}
    for name in names:
        rows = []
        for _ in range(COLD_START_PROCESSES):
            res = subprocess.run([sys.executable, os.path.abspath(__file__), "--cold-start-probe", name],
                                 capture_output=True, text=True, check=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__)))
            rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
        out[name] = {k: {"median_ms": round(float(np.median([r[k] for r in rows])), 2),
                         "max_ms": round(float(np.max([r[k] for r in rows])), 2)}
                     for k in ("first_ms", "second_ms", "tenth_ms")}
    return out


def _line(tag, name, r):
    s = r["summary"]
    return (f"[{tag}] {name:>40}: uniq-div {r['unique_single_target_mean_diversity']:.3f} "
            f">=base {r['unique_configs_at_least_baseline']} happy {r['requests']['happy / match']['diversity']:.3f} "
            f"sad {r['requests']['sad / match']['diversity']:.3f} cov {s['catalog_unique_tracks']} "
            f"rep {s['mean_repeat_rate']:.3f} sig {s['mean_ranking_signal_of_served']:.4f} p95 {s['latency_ms']['p95']} "
            f"speech {r['speech_like_share']['calm_requests']:.3f} fails {r['original_gate_failures']}")


def main():
    parser = argparse.ArgumentParser(description="Candidate-pool redesign analysis (analysis only)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--cold-start-probe", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold_start_probe:
        cold_start_probe(args.cold_start_probe)
        return
    if not held_out_disjoint():
        raise SystemExit("Held-out seeds overlap seeds used elsewhere")

    t0 = time.perf_counter()
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    catalog_hash = mp.file_sha256(MUSIC_CATALOG_278K_PATH)
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, dtype={"track_id": str}).reset_index(drop=True)
    X = catalog[mp.AUDIO_FEATURES].to_numpy(dtype=float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)

    for mood, mode in itertools.product(MOODS, ev.MODES):
        targets = ev.request_targets(mood, mode)
        np.random.seed(EXPLORATORY_SEED)
        a = recommend._coverage_top(catalog, targets, ev.LIST_SIZE).index.to_numpy()
        np.random.seed(EXPLORATORY_SEED)
        if not np.array_equal(a, coverage_top_pool(catalog, targets, ev.LIST_SIZE, STRATEGIES[CONTROL])):
            raise SystemExit(f"Control diverges from production for {mood}/{mode}")

    causes = root_cause(catalog, z)
    exploratory, base_x = evaluate(catalog, z, EXPLORATORY_SEED, variant_names())
    for name, r in exploratory.items():
        print(_line("exploratory", name, r))
    chosen, chosen_passes = choose(exploratory)
    print(f"Chosen on exploratory seeds: {chosen} (passes every gate: {chosen_passes})")
    if chosen:
        extra = f"{chosen} + calm guard"
        more, _ = evaluate(catalog, z, EXPLORATORY_SEED, [CONTROL, extra])
        exploratory[extra] = more[extra]
        print(_line("exploratory", extra, more[extra]))
    held_names = variant_names(chosen)
    held_out, base_h = evaluate(catalog, z, HELD_OUT_SEED, held_names)
    for name, r in held_out.items():
        print(_line("held-out", name, r))
    cold = cold_start(held_names)

    if mp.file_sha256(MUSIC_CATALOG_278K_PATH) != catalog_hash:
        raise SystemExit("The catalog changed during the analysis -- this must never happen.")
    report = {
        "analysis_only": True,
        "evaluation_date": datetime.date.today().isoformat(),
        "script_sha256": script_sha,
        "production_configuration": {"pool_multiplier": PRODUCTION_MULTIPLIER, "weighting": "current"},
        "catalog": {"path": os.path.relpath(MUSIC_CATALOG_278K_PATH, os.path.dirname(MODELS_DIR)),
                    "sha256": catalog_hash, "rows": int(len(catalog))},
        "seeds": {"exploratory": f"{EXPLORATORY_SEED}..{EXPLORATORY_SEED + RUNS - 1} (selection)",
                  "held_out": f"{HELD_OUT_SEED}..{HELD_OUT_SEED + RUNS - 1} (never used for selection)",
                  "used_elsewhere": {k: f"{a}..{b}" for k, (a, b) in USED_ELSEWHERE.items()},
                  "held_out_disjoint": held_out_disjoint(), "runs_per_request": RUNS},
        "strategies": STRATEGIES,
        "label_targets_affected_by_thresholds": list(LABEL_TARGETS),
        "selection_rule": SELECTION_RULE,
        "gates": GATES,
        "scoring": {"unique_configurations": "Phase 2H.9 rule C: chill/match and stressed/uplift count once (6)",
                    "baselines": "Phase 2H.9 rule B: guarded variants use the guarded calm random-eligible baseline",
                    "gap_to_baseline_closed": "supplementary, not a gate: (strategy - control) / (baseline - control) "
                                              "per configuration where the control is below its baseline"},
        "root_cause": causes,
        "random_eligible_baselines": {"exploratory": base_x, "held_out": base_h},
        "exploratory": exploratory,
        "chosen_on_exploratory": {"strategy": chosen, "passes_every_gate": chosen_passes},
        "held_out": held_out,
        "cold_start": cold,
        "notes": [
            "Diversity is feature-space diversity (mean pairwise distance in the 9 z-scored audio features), not a "
            "direct measure of listener-perceived variety.",
            "Bootstrap intervals resample one request's 50 runs on one catalog: sampling variability only.",
            "Precision/NDCG are omitted: relevance is the eligibility definition, so they cannot rank strategies.",
            "Latency is machine-dependent; strategies ran sequentially in one process; cold start uses fresh "
            "interpreters on sad/match.",
        ],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(args.output)} in {report['runtime_s']}s")


if __name__ == "__main__":
    main()
