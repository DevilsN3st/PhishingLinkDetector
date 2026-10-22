"""Derive every metric obtainable from benchmarks/results/*.json, plus cross-model statistics.

Confusion matrices are reconstructed from recall, FPR and the test class counts, so
counts are exact up to the 4-decimal rounding of the stored rates (about ±10 URLs).
Outputs results_derived.csv and ANALYSIS.md.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
TEST_POS, TEST_NEG = 117_142, 117_541
LIVE_N = {"openphish": 299, "phishunt": 753}
PRODUCTION = {"full": "stacked_hgb", "200k": "stacked_hgb_200k"}
RANK_METRICS = ["roc_auc", "pr_auc", "f1", "mcc", "balanced_accuracy", "fpr", "brand_slice_f1", "live_recall_mean"]
LOWER_IS_BETTER = {"fpr", "fnr", "fdr", "ms_per_url", "train_seconds", "size_mb", "neg_lr", "fp_per_1m_legit",
                   "missed_per_1k_phish"}


def derive(r: dict) -> dict:
    P, N = TEST_POS, TEST_NEG
    tp = r["recall"] * P
    fn = P - tp
    fp = r["fpr"] * N
    tn = N - fp
    n = P + N
    prec, rec, spec = tp / (tp + fp), r["recall"], tn / N
    npv = tn / (tn + fn)
    acc = (tp + tn) / n
    p_obs, p_exp = acc, ((tp + fp) * P + (tn + fn) * N) / n**2
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    fbeta = lambda b: (1 + b**2) * prec * rec / (b**2 * prec + rec)  # noqa: E731
    live_hits = {k: r[f"live_{k}_recall"] * v for k, v in LIVE_N.items()}
    out = {
        "split": "full" if r["train_rows"] > 1_000_000 else "200k",
        "TP": round(tp), "FN": round(fn), "FP": round(fp), "TN": round(tn),
        "accuracy": acc,
        "error_rate": 1 - acc,
        "balanced_accuracy": (rec + spec) / 2,
        "specificity": spec,
        "npv": npv,
        "fnr": 1 - rec,
        "fdr": 1 - prec,
        "f0_5": fbeta(0.5),
        "f2": fbeta(2),
        "mcc": mcc,
        "cohen_kappa": (p_obs - p_exp) / (1 - p_exp),
        "youden_j": rec + spec - 1,
        "markedness": prec + npv - 1,
        "g_mean": np.sqrt(rec * spec),
        "fowlkes_mallows": np.sqrt(prec * rec),
        "jaccard": tp / (tp + fp + fn),
        "pos_lr": rec / r["fpr"],
        "neg_lr": (1 - rec) / spec,
        "diagnostic_odds_ratio": (rec / r["fpr"]) / ((1 - rec) / spec),
        "gini": 2 * r["roc_auc"] - 1,
        "predicted_positive_rate": (tp + fp) / n,
        "fp_per_1m_legit": r["fpr"] * 1e6,
        "missed_per_1k_phish": (1 - rec) * 1e3,
        "live_recall_mean": float(np.mean([r["live_openphish_recall"], r["live_phishunt_recall"]])),
        "live_recall_pooled": sum(live_hits.values()) / sum(LIVE_N.values()),
        "live_missed": round(sum(LIVE_N.values()) - sum(live_hits.values())),
        "ms_per_url": 1000 / r["urls_per_sec"],
        "urls_per_sec_per_mb": r["urls_per_sec"] / max(r["size_mb"], 0.05),
        "train_us_per_row": r["train_seconds"] / r["train_rows"] * 1e6,
        "hours_to_score_1b": 1e9 / r["urls_per_sec"] / 3600,
    }
    return {**r, **out}


def ranks(df: pd.DataFrame) -> pd.DataFrame:
    rk = pd.DataFrame(index=df.index)
    for m in RANK_METRICS:
        rk[m] = df[m].rank(ascending=m in LOWER_IS_BETTER, method="min")
    df = df.copy()
    df["mean_rank"] = rk.mean(axis=1)
    return df


def pareto(df: pd.DataFrame) -> pd.Series:
    """Models not dominated on (ROC AUC higher, URLs/s higher, size lower)."""
    keep = []
    for i, a in df.iterrows():
        dominated = any((b.roc_auc >= a.roc_auc and b.urls_per_sec >= a.urls_per_sec and b.size_mb <= a.size_mb
                         and (b.roc_auc > a.roc_auc or b.urls_per_sec > a.urls_per_sec or b.size_mb < a.size_mb))
                        for j, b in df.iterrows() if j != i)
        keep.append(not dominated)
    return pd.Series(keep, index=df.index)


def md(df: pd.DataFrame, cols: list, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    rows = []
    for _, r in df[cols].iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(fmt.get(c, "{:.4f}").format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


def main():
    rows = [derive(json.loads(p.read_text())) for p in sorted(RESULTS_DIR.glob("*.json"))]
    df = pd.DataFrame(rows)
    df = pd.concat([ranks(g) for _, g in df.groupby("split")])
    df["pareto_optimal"] = pd.concat([pareto(g) for _, g in df.groupby("split")]).map({True: "yes", False: "no"})
    for split, prod in PRODUCTION.items():
        base = df[df.model == prod].iloc[0]
        m = df.split == split
        for c in ["roc_auc", "pr_auc", "f1", "mcc", "fpr", "live_recall_mean"]:
            df.loc[m, f"delta_{c}_vs_prod"] = df.loc[m, c] - base[c]
        df.loc[m, "speed_x_vs_prod"] = df.loc[m, "urls_per_sec"] / base["urls_per_sec"]
        df.loc[m, "size_x_vs_prod"] = df.loc[m, "size_mb"] / base["size_mb"]
    df = df.sort_values(["split", "mean_rank"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(HERE / "results_derived.csv", index=False)

    numeric = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("train_rows",)]
    stats = df.groupby("split")[numeric].agg(["mean", "median", "std", "min", "max"]).T
    stats.index.names = ["metric", "stat"]
    stats = stats.unstack("stat")
    stats.to_csv(HERE / "results_stats.csv")

    full, sub = df[df.split == "full"], df[df.split == "200k"]
    fam = full.groupby("family")[["roc_auc", "pr_auc", "f1", "mcc", "fpr", "brand_slice_f1", "live_recall_mean",
                                  "urls_per_sec", "size_mb", "train_seconds"]].agg(["mean", "max"])
    fam.columns = [f"{a}_{b}" for a, b in fam.columns]
    fam = fam.sort_values("roc_auc_mean", ascending=False).reset_index()

    pairs = []
    for name in sub.model:
        base = name.removesuffix("_200k")
        if base in set(full.model):
            a, b = sub[sub.model == name].iloc[0], full[full.model == base].iloc[0]
            pairs.append({"model": base, "auc_200k": a.roc_auc, "auc_full": b.roc_auc,
                          "delta_auc": b.roc_auc - a.roc_auc, "f1_200k": a.f1, "f1_full": b.f1,
                          "delta_f1": b.f1 - a.f1, "delta_fpr": b.fpr - a.fpr,
                          "train_time_x": b.train_seconds / a.train_seconds})
    scaling = pd.DataFrame(pairs).sort_values("delta_auc", ascending=False)

    pct = {c: "{:.2%}" for c in ["fpr", "fnr", "fdr", "error_rate", "predicted_positive_rate"]}
    big = {c: "{:,.0f}" for c in ["fp_per_1m_legit", "urls_per_sec", "train_seconds", "urls_per_sec_per_mb"]}
    f2 = {c: "{:.2f}" for c in ["pos_lr", "diagnostic_odds_ratio", "mean_rank", "ms_per_url", "missed_per_1k_phish",
                                "train_us_per_row", "hours_to_score_1b", "size_mb", "speed_x_vs_prod",
                                "size_x_vs_prod"]}
    F = {**pct, **big, **f2}

    sections = ["# Derived model metrics\n",
                f"Test set: {TEST_POS:,} phishing / {TEST_NEG:,} legitimate URLs. Confusion counts are "
                "reconstructed from stored recall and FPR (±~10 URLs from rounding). `mean_rank` = average "
                f"rank over {', '.join(RANK_METRICS)} within each training-size group. Pareto-optimal = no "
                "other model is at least as good on ROC AUC, speed and size at once.\n"]
    for title, g in [("Full training data (1.83M URLs)", full), ("200k-URL subsample", sub)]:
        sections += [
            f"\n## {title}\n",
            "\n### Confusion matrix and threshold metrics\n",
            md(g, ["model", "TP", "FN", "FP", "TN", "accuracy", "balanced_accuracy", "precision", "recall",
                   "specificity", "npv", "f1", "f0_5", "f2"], F),
            "\n\n### Agreement and diagnostic metrics\n",
            md(g, ["model", "mcc", "cohen_kappa", "youden_j", "markedness", "g_mean", "fowlkes_mallows",
                   "jaccard", "pos_lr", "neg_lr", "diagnostic_odds_ratio"], F),
            "\n\n### Ranking, error and robustness metrics\n",
            md(g, ["model", "roc_auc", "gini", "pr_auc", "fpr", "fnr", "fdr", "fp_per_1m_legit",
                   "missed_per_1k_phish", "brand_slice_f1", "live_recall_mean", "live_recall_pooled",
                   "live_missed"], F),
            "\n\n### Cost metrics\n",
            md(g, ["model", "train_seconds", "train_us_per_row", "urls_per_sec", "ms_per_url", "hours_to_score_1b",
                   "size_mb", "urls_per_sec_per_mb", "speed_x_vs_prod", "size_x_vs_prod"], F),
            "\n\n### Versus production stacked model, and overall standing\n",
            md(g, ["model", "delta_roc_auc_vs_prod", "delta_pr_auc_vs_prod", "delta_f1_vs_prod",
                   "delta_mcc_vs_prod", "delta_fpr_vs_prod", "delta_live_recall_mean_vs_prod", "mean_rank",
                   "pareto_optimal"], {**F, "delta_fpr_vs_prod": "{:+.2%}",
                                       **{c: "{:+.4f}" for c in ["delta_roc_auc_vs_prod", "delta_pr_auc_vs_prod",
                                                                 "delta_f1_vs_prod", "delta_mcc_vs_prod",
                                                                 "delta_live_recall_mean_vs_prod"]}}),
            "\n",
        ]
    key = ["roc_auc", "pr_auc", "f1", "mcc", "accuracy", "balanced_accuracy", "fpr", "brand_slice_f1",
           "live_recall_mean", "urls_per_sec", "size_mb", "train_seconds"]
    for split in ["full", "200k"]:
        s = df[df.split == split][key].agg(["mean", "median", "std", "min", "max"]).T
        s["range"] = s["max"] - s["min"]
        s["cv"] = s["std"] / s["mean"]
        s = s.reset_index().rename(columns={"index": "metric"})
        sections += [f"\n## Summary statistics across models ({split})\n",
                     md(s, ["metric", "mean", "median", "std", "min", "max", "range", "cv"],
                        {c: "{:,.4f}" for c in ["mean", "median", "std", "min", "max", "range", "cv"]}), "\n"]
    sections += ["\n## By model family (full data; mean and best)\n",
                 md(fam, list(fam.columns), {c: ("{:,.0f}" if any(k in c for k in ("urls", "seconds"))
                                                 else "{:.4f}") for c in fam.columns}),
                 "\n\n## Data scaling: 200k vs full training data\n",
                 md(scaling, list(scaling.columns), {"delta_auc": "{:+.4f}", "delta_f1": "{:+.4f}",
                                                     "delta_fpr": "{:+.2%}", "train_time_x": "{:.1f}x"}),
                 "\n\n## Correlation between metrics across all 18 runs\n",
                 md(df[["roc_auc", "pr_auc", "f1", "mcc", "fpr", "brand_slice_f1", "live_recall_mean",
                        "urls_per_sec", "size_mb"]].corr().round(3).reset_index().rename(columns={"index": "metric"}),
                    ["metric", "roc_auc", "pr_auc", "f1", "mcc", "fpr", "brand_slice_f1", "live_recall_mean",
                     "urls_per_sec", "size_mb"], {c: "{:+.3f}" for c in df.columns}),
                 "\n"]
    (HERE / "ANALYSIS.md").write_text("\n".join(sections))
    print("\n".join(sections))


if __name__ == "__main__":
    main()
