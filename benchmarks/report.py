"""Aggregate benchmarks/results/*.json into markdown comparison tables."""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent / "RESULTS.md"

COLS = [
    ("model", "Model", "{}"),
    ("family", "Family", "{}"),
    ("roc_auc", "ROC AUC", "{:.4f}"),
    ("pr_auc", "PR AUC", "{:.4f}"),
    ("f1", "F1", "{:.4f}"),
    ("precision", "Precision", "{:.4f}"),
    ("recall", "Recall", "{:.4f}"),
    ("fpr", "FPR", "{:.2%}"),
    ("brand_slice_f1", "Brand F1", "{:.4f}"),
    ("live_openphish_recall", "OpenPhish", "{:.3f}"),
    ("live_phishunt_recall", "phishunt", "{:.3f}"),
    ("ext_phreshphish_auc", "PhreshPhish AUC", "{:.4f}"),
    ("ext_jpxxx_auc", "JPxxx AUC", "{:.4f}"),
    ("ext_deepurlbench_auc", "DeepURLBench AUC", "{:.4f}"),
    ("train_seconds", "Train s", "{:,.0f}"),
    ("urls_per_sec", "URLs/s", "{:,}"),
    ("size_mb", "Size MB", "{:.1f}"),
]


def table(rows):
    rows = sorted(rows, key=lambda r: -r["roc_auc"])
    lines = ["| " + " | ".join(c[1] for c in COLS) + " |",
             "|" + "---|" * len(COLS)]
    for r in rows:
        cells = []
        for k, _, fmt in COLS:
            v = r.get(k)
            cells.append("—" if v is None else fmt.format(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    full = [r for r in rows if r["train_rows"] > 1_000_000]
    sub = [r for r in rows if r["train_rows"] <= 1_000_000]
    md = (
        "# Model comparison\n\n"
        "Same domain-grouped splits for every model; threshold = best F1 on val; "
        "metrics on the held-out test split (234,683 URLs). OpenPhish / phishunt = recall "
        "on live feeds excluded from training. URLs/s = end-to-end scoring of 20k raw URLs "
        "on an M-series laptop (CPU for classic models, MPS for neural).\n\n"
        f"## Full training data ({full[0]['train_rows']:,} URLs)\n\n{table(full)}\n\n"
        f"## 200k-URL training subsample (fair comparison incl. DistilBERT)\n\n{table(sub)}\n"
    )
    OUT.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
