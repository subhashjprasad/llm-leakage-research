#!/usr/bin/env python3
"""Phase 4: Scoring.

Loads data/questions.jsonl and data/forecasts.jsonl, computes Brier score and
log loss per cohort and per variant, runs bootstrap CIs, compares model vs
crowd (H1, H2, H3), and writes results/ outputs.

Run: .venv/bin/python3 score.py
"""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


EPS = 1e-9   # for log loss clipping


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def brier(probs, outcomes):
    return float(np.mean((np.array(probs) - np.array(outcomes)) ** 2))


def log_loss(probs, outcomes):
    p = np.clip(np.array(probs), EPS, 1 - EPS)
    y = np.array(outcomes)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bootstrap_mean_diff(a, b, n=10000, seed=42):
    """Bootstrap CI for mean(a) - mean(b) over matched pairs.

    a and b must have the same length. Returns (mean_diff, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    diffs = []
    n_items = len(a)
    for _ in range(n):
        idx = rng.integers(0, n_items, size=n_items)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    return float(np.mean(a) - np.mean(b)), float(lo), float(hi)


def reliability_bins(probs, outcomes, n_bins=5):
    """Return (mean_pred, mean_actual, count) per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    results = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (np.array(probs) >= lo) & (np.array(probs) < hi)
        if not np.any(mask):
            continue
        results.append((
            float(np.mean(np.array(probs)[mask])),
            float(np.mean(np.array(outcomes)[mask])),
            int(np.sum(mask)),
        ))
    return results


def main():
    cfg          = load_config()
    questions_path  = Path(cfg["paths"]["questions"])
    forecasts_path  = Path(cfg["paths"]["forecasts"])
    results_dir     = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["sampling"]["random_seed"]
    random.seed(seed)

    questions = {q["question_id"]: q for q in load_jsonl(questions_path)}
    forecasts  = load_jsonl(forecasts_path)

    print("=" * 60)
    print("Phase 4: Scoring")
    print(f"  Questions: {len(questions)}")
    print(f"  Forecast records: {len(forecasts)}")
    print("=" * 60)

    # Aggregate forecasts: (qid, variant) -> list of probabilities.
    agg = defaultdict(list)
    for fc in forecasts:
        key = (fc["question_id"], fc["variant"])
        agg[key].append(fc["probability"])

    # Per-question summary: mean prob and stdev across runs.
    def question_stats(cohort, variant):
        """Return (qids, mean_probs, stdevs, outcomes, crowd_prices)."""
        qids = []; means = []; stds = []; outs = []; crowds = []
        for qid, q in questions.items():
            if q["cohort"] != cohort:
                continue
            key = (qid, variant)
            if key not in agg or not agg[key]:
                continue
            runs_probs = agg[key]
            qids.append(qid)
            means.append(float(np.mean(runs_probs)))
            stds.append(float(np.std(runs_probs, ddof=1)) if len(runs_probs) > 1 else 0.0)
            outs.append(float(q["outcome"]))
            crowds.append(float(q.get("open_market_price") or 0.5))
        return qids, means, stds, outs, crowds

    # Compute metrics per (cohort, variant).
    results = {}
    for cohort in ["pre_cutoff", "post_cutoff"]:
        for variant in ["full", "stripped"]:
            qids, means, stds, outs, crowds = question_stats(cohort, variant)
            if not means:
                print(f"  WARNING: no data for {cohort}/{variant}")
                continue
            results[(cohort, variant)] = {
                "qids":    qids,
                "means":   means,
                "stds":    stds,
                "outcomes": outs,
                "crowds":  crowds,
                "brier":   brier(means, outs),
                "log_loss": log_loss(means, outs),
                "n":       len(means),
            }
            print(f"  {cohort}/{variant}: n={len(means)}, "
                  f"brier={brier(means, outs):.4f}, ll={log_loss(means, outs):.4f}")

    # ---- Noise floor ----
    all_stds = []
    for v in results.values():
        all_stds.extend(v["stds"])
    noise_mean_std = float(np.mean(all_stds)) if all_stds else 0.0
    noise_brier_spread = float((noise_mean_std) ** 2)  # rough Brier spread from run variance

    print(f"\nNoise floor: mean within-question stdev = {noise_mean_std:.4f}, "
          f"implied Brier spread ~ {noise_brier_spread:.4f}")

    # ---- H1: pre-cutoff-full vs post-cutoff-full ----
    summary = {}
    if ("pre_cutoff", "full") in results and ("post_cutoff", "full") in results:
        pre_b  = results[("pre_cutoff",  "full")]["brier"]
        post_b = results[("post_cutoff", "full")]["brier"]
        h1_gap = post_b - pre_b  # positive = model worse post-cutoff, as expected if H1 holds

        # Paired bootstrap over the union of questions.
        pre_r  = results[("pre_cutoff",  "full")]
        post_r = results[("post_cutoff", "full")]
        # Not necessarily same questions, so we bootstrap each independently.
        pre_briers  = [(p - o) ** 2 for p, o in zip(pre_r["means"],  pre_r["outcomes"])]
        post_briers = [(p - o) ** 2 for p, o in zip(post_r["means"], post_r["outcomes"])]

        rng = np.random.default_rng(seed)
        boot_gaps = []
        n_pre  = len(pre_briers)
        n_post = len(post_briers)
        for _ in range(10000):
            i = rng.integers(0, n_pre,  size=n_pre)
            j = rng.integers(0, n_post, size=n_post)
            boot_gaps.append(np.mean(np.array(post_briers)[j]) - np.mean(np.array(pre_briers)[i]))
        ci_lo = float(np.percentile(boot_gaps, 2.5))
        ci_hi = float(np.percentile(boot_gaps, 97.5))

        summary["H1"] = {
            "pre_brier":  pre_b,
            "post_brier": post_b,
            "gap":        h1_gap,
            "ci_95":      [ci_lo, ci_hi],
            "noise_floor_brier": noise_brier_spread,
            "conclusion": (
                "gap > noise floor" if abs(h1_gap) > noise_brier_spread
                else "inconclusive: gap within noise floor"
            ),
        }
        print(f"\nH1 (leakage gap): post_brier - pre_brier = {h1_gap:+.4f}  "
              f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"   Noise floor: {noise_brier_spread:.4f}  -> {summary['H1']['conclusion']}")

    # ---- H2: full vs stripped within each cohort ----
    for cohort in ["pre_cutoff", "post_cutoff"]:
        if (cohort, "full") not in results or (cohort, "stripped") not in results:
            continue
        full_b    = results[(cohort, "full")]["brier"]
        stripped_b = results[(cohort, "stripped")]["brier"]
        h2_gap = stripped_b - full_b  # positive = stripped is worse (full has extra signal)

        # Matched bootstrap on same question set.
        full_r    = results[(cohort, "full")]
        strip_r   = results[(cohort, "stripped")]
        # Align on qid.
        qid_to_full   = {qid: (m, o) for qid, m, o in zip(full_r["qids"],  full_r["means"],  full_r["outcomes"])}
        qid_to_strip  = {qid: (m, o) for qid, m, o in zip(strip_r["qids"], strip_r["means"], strip_r["outcomes"])}
        common = list(set(qid_to_full) & set(qid_to_strip))
        full_bs  = [(qid_to_full[q][0]  - qid_to_full[q][1])  ** 2 for q in common]
        strip_bs = [(qid_to_strip[q][0] - qid_to_strip[q][1]) ** 2 for q in common]

        if common:
            diff, ci_lo, ci_hi = bootstrap_mean_diff(strip_bs, full_bs, seed=seed)
            summary[f"H2_{cohort}"] = {
                "full_brier":     full_b,
                "stripped_brier": stripped_b,
                "gap":            diff,
                "ci_95":          [ci_lo, ci_hi],
            }
            print(f"\nH2 ({cohort}): stripped - full brier = {diff:+.4f}  "
                  f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ---- H3: model vs crowd on post-cutoff ----
    # NOTE: Polymarket's Data API does not retain price history for closed
    # markets (returns empty array for all resolved markets).  open_market_price
    # is None for every question.  H3 therefore uses 0.5 as the crowd baseline,
    # which is a "no-information" prior, not the actual opening crowd price.
    # The comparison becomes "is the model better than a coin flip?" — a weaker
    # test than the original H3.  This limitation is noted in the writeup.
    if ("post_cutoff", "full") in results:
        r = results[("post_cutoff", "full")]
        # Use 0.5 baseline instead of actual crowd prices.
        baseline = [0.5] * len(r["outcomes"])
        crowd_b  = brier(baseline, r["outcomes"])
        model_b  = r["brier"]
        model_bs  = [(p - o) ** 2 for p, o in zip(r["means"],  r["outcomes"])]
        baseline_bs = [(0.5 - o) ** 2 for o in r["outcomes"]]
        diff, ci_lo, ci_hi = bootstrap_mean_diff(model_bs, baseline_bs, seed=seed)
        summary["H3"] = {
            "model_brier":      model_b,
            "baseline_brier":   crowd_b,
            "baseline_note":    "0.5 uninformed prior (opening crowd price unavailable — Polymarket Data API does not retain historical prices for resolved markets)",
            "gap":              diff,
            "ci_95":            [ci_lo, ci_hi],
            "conclusion":       "model beats baseline" if ci_hi < 0 else
                                "baseline beats model" if ci_lo > 0 else
                                "no significant difference vs baseline",
        }
        print(f"\nH3 (model vs 0.5 baseline): model_brier - baseline_brier = {diff:+.4f}  "
              f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"   NOTE: baseline is 0.5 (coin flip), not crowd opening price")
        print(f"   -> {summary['H3']['conclusion']}")

    # ---- Calibration plots ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cohort in zip(axes, ["pre_cutoff", "post_cutoff"]):
        key = (cohort, "full")
        if key not in results:
            ax.set_visible(False)
            continue
        r = results[key]
        bins = reliability_bins(r["means"], r["outcomes"], n_bins=5)
        if bins:
            pred, actual, counts = zip(*bins)
            sc = ax.scatter(pred, actual, s=[c * 30 for c in counts], alpha=0.7, zorder=3)
            for px, py, c in zip(pred, actual, counts):
                ax.annotate(f"n={c}", (px, py), textcoords="offset points",
                            xytext=(4, 4), fontsize=7)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Empirical frequency")
        ax.set_title(f"Calibration: {cohort}\n(Brier={results[key]['brier']:.3f}, n={results[key]['n']})")
        ax.legend(fontsize=7)

    plt.tight_layout()
    calib_path = results_dir / "calibration.png"
    plt.savefig(calib_path, dpi=150)
    plt.close()
    print(f"\nCalibration plot: {calib_path}")

    # ---- Comparison plot ----
    fig, ax = plt.subplots(figsize=(7, 4))
    comparisons = []
    labels = []
    for h_key, h_data in summary.items():
        if "gap" in h_data and "ci_95" in h_data:
            comparisons.append((h_data["gap"], h_data["ci_95"][0], h_data["ci_95"][1]))
            labels.append(h_key)

    if comparisons:
        gaps   = [c[0] for c in comparisons]
        lo_err = [c[0] - c[1] for c in comparisons]
        hi_err = [c[2] - c[0] for c in comparisons]
        y_pos  = range(len(labels))
        ax.barh(y_pos, gaps, xerr=[lo_err, hi_err], color="steelblue", alpha=0.7,
                capsize=5, error_kw={"elinewidth": 1.5})
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Brier score difference (positive = worse)")
        ax.set_title("Paired comparisons with 95% bootstrap CIs")
        plt.tight_layout()

    comparison_path = results_dir / "comparison.png"
    plt.savefig(comparison_path, dpi=150)
    plt.close()
    print(f"Comparison plot: {comparison_path}")

    # ---- misses.csv ----
    rows = []
    for (cohort, variant), r in results.items():
        if variant != "full":
            continue
        for qid, prob, std, out in zip(r["qids"], r["means"], r["stds"], r["outcomes"]):
            q   = questions[qid]
            bs  = (prob - out) ** 2
            rows.append({
                "question_id":    qid,
                "cohort":         cohort,
                "category":       q.get("category", ""),
                "prompt_text":    q.get("prompt_text", "")[:200],
                "mean_probability": round(prob, 4),
                "run_stdev":      round(std, 4),
                "outcome":        int(out),
                "brier":          round(bs, 4),
                "error_label":    "",
            })
    rows.sort(key=lambda x: x["brier"], reverse=True)

    misses_path = results_dir / "misses.csv"
    import csv
    with open(misses_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"Misses CSV: {misses_path}  ({len(rows)} rows)")

    # ---- summary.json ----
    full_summary = {
        "hypotheses":   summary,
        "per_cohort_variant": {
            f"{c}/{v}": {
                "n":        d["n"],
                "brier":    round(d["brier"], 4),
                "log_loss": round(d["log_loss"], 4),
                "mean_run_stdev": round(float(np.mean(d["stds"])), 4) if d["stds"] else None,
            }
            for (c, v), d in results.items()
        },
        "noise_floor": {
            "mean_within_question_stdev": round(noise_mean_std, 4),
            "implied_brier_spread":       round(noise_brier_spread, 4),
        },
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
