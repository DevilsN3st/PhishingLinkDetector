"""Shared protocol for the model comparison: data, features, score files.

Every model is trained on the same domain-grouped train split (optionally a
fixed subsample) and scores the same evaluation splits: val (threshold and
calibration fitting), test, the live feeds and two external test sets. Scores
are saved per URL to results/scores/<name>.npz so every metric is computed
afterwards by metrics.py, without retraining.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishdetect import PROCESSED_DIR  # noqa: E402
from phishdetect.features import extract_features  # noqa: E402

SEED = 42
SUBSAMPLE = 200_000
EVAL_SPLITS = ("val", "test", "live", "ext_phreshphish", "ext_jpxxx", "ext_deepurlbench")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCORES_DIR = RESULTS_DIR / "scores"
CACHE_DIR = PROCESSED_DIR / "bench_cache"
BRAND_HOST_FEATURES = ["brand_in_subdomain", "brand_token_in_domain", "brand_embedded_in_domain",
                       "brand_typo_domain", "brand_typo_subdomain"]
N_TLDS = 250


def load_splits(subsample: bool = False, eval_cap: int | None = None) -> dict:
    """Each frame keeps a `row` column: its position in the full split, so score
    files from models evaluated on a capped sample can be aligned later."""
    df = pd.read_parquet(PROCESSED_DIR / "urls.parquet")
    splits = {}
    for s in ("train",) + EVAL_SPLITS:
        d = df[df["split"] == s].reset_index(drop=True)
        d["row"] = np.arange(len(d))
        if s == "train" and subsample:
            d = d.sample(SUBSAMPLE, random_state=SEED).reset_index(drop=True)
        elif s != "train" and s != "live" and eval_cap and len(d) > eval_cap:
            d = d.sample(eval_cap, random_state=SEED).sort_values("row").reset_index(drop=True)
        splits[s] = d
    return splits


def features(splits: dict, subsample: bool = False) -> dict:
    """Lexical features per split, cached on disk for the full splits."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, d in splits.items():
        path = CACHE_DIR / f"features_{name}.parquet"
        rows = d["row"].to_numpy()
        is_full = np.array_equal(rows, np.arange(len(d)))
        if path.exists():
            f = pd.read_parquet(path)
            out[name] = f if is_full else f.iloc[rows].reset_index(drop=True)
        elif is_full:
            out[name] = extract_features(d["norm"].to_numpy())
            out[name].to_parquet(path, index=False)
        else:
            out[name] = extract_features(d["norm"].to_numpy())
    return out


def tld_vocab(train_features: pd.DataFrame) -> dict:
    return {t: i + 1 for i, t in enumerate(train_features["tld"].value_counts().index[:N_TLDS])}


def brand_slice(test_features: pd.DataFrame) -> np.ndarray:
    return (test_features[BRAND_HOST_FEATURES].sum(axis=1) > 0).to_numpy()


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def evaluate(name: str, family: str, splits: dict, scores: dict, *, train_seconds: float,
             urls_per_sec: float, size_mb: float, train_rows: int, notes: str = "",
             brand_mask=None, **_) -> dict:
    """Persist per-URL scores and a metrics JSON; print a one-line summary."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    arrays = {}
    for k, s in scores.items():
        arrays[f"score_{k}"] = np.asarray(s, dtype=np.float64)
        arrays[f"row_{k}"] = splits[k]["row"].to_numpy()
    np.savez_compressed(SCORES_DIR / f"{name}.npz", **arrays)

    y_va, y_te = splits["val"]["label"].to_numpy(), splits["test"]["label"].to_numpy()
    thr = best_f1_threshold(y_va, scores["val"])
    p, pred = scores["test"], scores["test"] >= thr
    live = splits["live"]
    live_pred = scores["live"] >= thr
    if brand_mask is None and "test" in splits:
        # brand slice is computed by the caller when features are available
        brand_mask = np.ones(len(y_te), dtype=bool)
    res = {
        "model": name,
        "family": family,
        "train_rows": int(train_rows),
        "roc_auc": roc_auc_score(y_te, p),
        "pr_auc": average_precision_score(y_te, p),
        "f1": f1_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "fpr": float(pred[y_te == 0].mean()) if (y_te == 0).any() else None,
        "brand_slice_f1": f1_score(y_te[brand_mask], pred[brand_mask]) if brand_mask is not None and brand_mask.any() else None,
        "live_openphish_recall": float(live_pred[(live["source"] == "openphish").to_numpy()].mean()) if (live["source"] == "openphish").any() else None,
        "live_phishunt_recall": float(live_pred[(live["source"] == "phishunt").to_numpy()].mean()) if (live["source"] == "phishunt").any() else None,
        "train_seconds": round(train_seconds, 1),
        "urls_per_sec": round(urls_per_sec),
        "size_mb": round(size_mb, 1),
        "notes": notes,
    }
    for k in ("ext_phreshphish", "ext_jpxxx", "ext_deepurlbench"):
        if k in scores and k in splits and len(splits[k]):
            res[f"{k}_auc"] = roc_auc_score(splits[k]["label"], scores[k])
    res = {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(res, indent=2))
    (SCORES_DIR / f"{name}.json").write_text(json.dumps(
        {k: res[k] for k in ("model", "family", "train_rows", "train_seconds", "urls_per_sec", "size_mb", "notes")
         if k in res}, indent=2))
    print(f"{name:28s} AUC={res['roc_auc']:.4f} F1={res['f1']:.4f} FPR={res.get('fpr') or 0:.4f} "
          f"live={res.get('live_openphish_recall') or 0:.3f}/{res.get('live_phishunt_recall') or 0:.3f} "
          f"train={res['train_seconds']}s {res['urls_per_sec']}/s {res['size_mb']}MB", flush=True)
    return res


def throughput(score_fn, urls, n: int = 20_000) -> float:
    """End-to-end URLs/second on raw test URLs (normalization + features + model)."""
    sample = list(urls[:n])
    t0 = time.perf_counter()
    score_fn(sample)
    return len(sample) / (time.perf_counter() - t0)


def size_mb(obj=None, path=None) -> float:
    if path is not None:
        return os.path.getsize(path) / 1e6
    import tempfile

    import joblib
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        tmp = f.name
    joblib.dump(obj, tmp, compress=3)
    mb = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return mb


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0

    @property
    def elapsed(self):
        return time.perf_counter() - self.t0
