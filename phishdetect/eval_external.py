"""Score the saved detector on an extra held-out file and merge into metrics.

Usage:
  python -m phishdetect.eval_external --name ext_deepurlbench --parquet data/raw/hf_deepurlbench_sample.parquet
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from . import MODELS_DIR
from .features import normalize_url
from .model import DEFAULT_MODEL_PATH, PhishingDetector


def _recall_at_fpr(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    ok = np.where(fpr <= target)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def metrics(y, p, threshold):
    pred = p >= threshold
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "false_positive_rate": float(pred[y == 0].mean()) if (y == 0).any() else None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y, p)),
        "recall_at_1pct_fpr": _recall_at_fpr(y, p, 0.01),
        "recall_at_0.1pct_fpr": _recall_at_fpr(y, p, 0.001),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--url-col", default="url")
    p.add_argument("--label-col", default="label")
    p.add_argument("--positive", default="phishing")
    args = p.parse_args()

    raw = pd.read_parquet(args.parquet)
    labels = raw[args.label_col]
    if labels.dtype == object or str(labels.dtype).startswith(("string", "category")):
        y = labels.astype(str).eq(args.positive).astype(int).to_numpy()
    else:
        y = labels.astype(int).to_numpy()
    urls = raw[args.url_col].map(normalize_url).to_numpy()
    det = PhishingDetector.load(DEFAULT_MODEL_PATH)
    p_hat = det.predict_proba(urls)
    res = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics(y, p_hat, det.threshold).items()}
    print(json.dumps({args.name: res}, indent=2))

    report_path = MODELS_DIR / "metrics.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.setdefault("external", {})[args.name] = res
    report_path.write_text(json.dumps(report, indent=2))

    scores_path = MODELS_DIR / "scores.parquet"
    extra = pd.DataFrame({"split": args.name, "source": args.name, "y": y, "p": p_hat})
    if scores_path.exists():
        extra = pd.concat([pd.read_parquet(scores_path), extra], ignore_index=True)
    extra.to_parquet(scores_path, index=False)


if __name__ == "__main__":
    main()
