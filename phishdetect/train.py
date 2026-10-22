"""Train the phishing URL detector and report metrics.

Usage: python -m phishdetect.train [--max-train N]
Reads data/processed/urls.parquet, writes models/phishing_detector.joblib and
models/metrics.json. Also writes per-URL scores to models/scores.parquet so
calibration and operating-point metrics can be recomputed without retraining.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold

from . import MODELS_DIR, PROCESSED_DIR
from .features import extract_features, get_brands
from .model import (DEFAULT_MODEL_PATH, N_TLDS, PhishingDetector, PlattCalibrator, as_gbm_frame,
                    fit_gbm, gbm_iterations, make_gbm, make_text_model, tabular_matrix)

SEED = 42
BRAND_HOST_FEATURES = ["brand_in_subdomain", "brand_token_in_domain", "brand_embedded_in_domain",
                       "brand_typo_domain", "brand_typo_subdomain"]
EXT_SPLITS = ("ext_phreshphish", "ext_jpxxx", "ext_deepurlbench")


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    if (y == 0).all():
        return {"n": int(len(y)), "false_positive_rate": round(float(pred.mean()), 4)}
    if (y == 1).all():
        return {"n": int(len(y)), "recall": round(float(pred.mean()), 4)}
    out = {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "false_positive_rate": float(pred[y == 0].mean()),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y, p),
        "recall_at_1pct_fpr": _recall_at_fpr(y, p, 0.01),
        "recall_at_0.1pct_fpr": _recall_at_fpr(y, p, 0.001),
    }
    return {k: round(float(v), 4) if isinstance(v, float) else v for k, v in out.items()}


def _recall_at_fpr(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    ok = np.where(fpr <= target)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def oof_text_scores(text_model, urls, y, groups, n_splits=3):
    """Out-of-fold text-model scores so the stacker never sees leaked scores."""
    oof = np.zeros(len(urls), dtype=np.float32)
    for i, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(urls, y, groups)):
        m = clone(text_model).fit(urls[tr], y[tr])
        oof[va] = m.predict_proba(urls[va])[:, 1]
        print(f"    fold {i + 1}/{n_splits} done")
    return oof


def _score(gbm, X, cols, calibrator=None):
    raw = gbm.predict_proba(as_gbm_frame(X[cols]))[:, 1]
    return calibrator.transform(raw) if calibrator is not None else raw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-train", type=int, default=0, help="subsample training rows (0 = all)")
    p.add_argument("--no-refit", action="store_true",
                   help="ship the evaluation model instead of refitting on train+val+test")
    args = p.parse_args()
    t0 = time.time()

    df = pd.read_parquet(PROCESSED_DIR / "urls.parquet")
    train = df[df["split"] == "train"]
    if args.max_train and len(train) > args.max_train:
        train = train.sample(args.max_train, random_state=SEED)
    val, test, live = (df[df["split"] == s] for s in ("val", "test", "live"))
    ext = {s: df[df["split"] == s] for s in EXT_SPLITS if (df["split"] == s).any()}
    print(f"train={len(train):,d} val={len(val):,d} test={len(test):,d} live={len(live):,d} "
          f"ext={{{', '.join(f'{k}={len(v):,d}' for k, v in ext.items())}}}")

    y_tr, y_va, y_te, y_live = (d["label"].to_numpy() for d in (train, val, test, live))
    u_tr, u_va, u_te, u_live = (d["norm"].to_numpy() for d in (train, val, test, live))

    print("[1/4] lexical features")
    F_tr, F_va, F_te, F_live = (extract_features(u) for u in (u_tr, u_va, u_te, u_live))
    F_ext = {s: extract_features(d["norm"].to_numpy()) for s, d in ext.items()}
    tld_vocab = {t: i + 1 for i, t in enumerate(F_tr["tld"].value_counts().index[:N_TLDS])}

    print("[2/4] char n-gram model (out-of-fold for stacking)")
    text_model = make_text_model(SEED)
    oof = oof_text_scores(text_model, u_tr, y_tr, train["reg_domain"].to_numpy())
    text_model.fit(u_tr, y_tr)
    ts_va, ts_te, ts_live = (text_model.predict_proba(u)[:, 1] for u in (u_va, u_te, u_live))
    ts_ext = {s: text_model.predict_proba(d["norm"].to_numpy())[:, 1] for s, d in ext.items()}

    print("[3/4] gradient boosting on lexical features only (baseline)")
    X_tr0, X_va0, X_te0 = (tabular_matrix(F, tld_vocab) for F in (F_tr, F_va, F_te))
    gbm_lex = fit_gbm(make_gbm(), X_tr0, y_tr, X_va0, y_va)

    print("[4/4] stacked gradient boosting (lexical features + char n-gram score)")
    X_tr, X_va, X_te, X_live = (tabular_matrix(F, tld_vocab, s)
                                for F, s in ((F_tr, oof), (F_va, ts_va), (F_te, ts_te), (F_live, ts_live)))
    feature_cols = list(X_tr.columns)
    gbm = fit_gbm(make_gbm(), X_tr, y_tr, X_va, y_va)
    raw_va = _score(gbm, X_va, feature_cols)
    calibrator = PlattCalibrator().fit(raw_va, y_va)
    p_va = calibrator.transform(raw_va)
    p_te = _score(gbm, X_te, feature_cols, calibrator)
    p_live = _score(gbm, X_live, feature_cols, calibrator)
    threshold = best_f1_threshold(y_va, p_va)
    brand_slice = (F_te[BRAND_HOST_FEATURES].sum(axis=1) > 0).to_numpy()

    report = {
        "threshold": round(threshold, 4),
        "second_stage": "lightgbm",
        "calibrated": True,
        "test": {
            "char_ngram_only": metrics(y_te, ts_te, best_f1_threshold(y_va, ts_va)),
            "lexical_gbm_only": metrics(y_te, _score(gbm_lex, X_te0, list(X_te0.columns)),
                                        best_f1_threshold(y_va, _score(gbm_lex, X_va0, list(X_va0.columns)))),
            "stacked": metrics(y_te, p_te, threshold),
            "stacked_uncalibrated": metrics(y_te, _score(gbm, X_te, feature_cols), threshold),
        },
        "test_by_source": {src: metrics(y_te[m], p_te[m], threshold)
                           for src in sorted(test["source"].unique())
                           for m in [(test["source"] == src).to_numpy()]},
        "test_brand_impersonation_slice": metrics(y_te[brand_slice], p_te[brand_slice], threshold),
        "live_recall_by_feed": {src: metrics(y_live[m], p_live[m], threshold)["recall"]
                                for src in sorted(live["source"].unique())
                                for m in [(live["source"] == src).to_numpy()]},
        "external": {},
        "rows": {"train": len(train), "val": len(val), "test": len(test), "live": len(live)},
        "boosting_iterations": {"lexical": gbm_iterations(gbm_lex), "stacked": gbm_iterations(gbm)},
    }

    score_parts = [
        pd.DataFrame({"split": "val", "source": val["source"].to_numpy(), "y": y_va, "p": p_va}),
        pd.DataFrame({"split": "test", "source": test["source"].to_numpy(), "y": y_te, "p": p_te}),
        pd.DataFrame({"split": "live", "source": live["source"].to_numpy(), "y": y_live, "p": p_live}),
    ]
    for s, d in ext.items():
        X_s = tabular_matrix(F_ext[s], tld_vocab, ts_ext[s])
        p_s = _score(gbm, X_s, feature_cols, calibrator)
        y_s = d["label"].to_numpy()
        report["external"][s] = metrics(y_s, p_s, threshold)
        if "seen_domain" in d.columns:
            seen = d["seen_domain"].to_numpy()
            if seen.any() and (~seen).any():
                report["external"][f"{s}_unseen_domain"] = metrics(y_s[~seen], p_s[~seen], threshold)
                report["external"][f"{s}_seen_domain"] = metrics(y_s[seen], p_s[seen], threshold)
        score_parts.append(pd.DataFrame({"split": s, "source": d["source"].to_numpy(), "y": y_s, "p": p_s}))
        report["rows"][s] = len(d)

    print(json.dumps(report, indent=2))

    if not args.no_refit:
        print("[refit] text model + gradient boosting on train+val+test")
        del X_tr0, X_va0, X_te0, gbm_lex
        refit_only = df[df["split"] == "refit_only"]
        labeled = pd.concat([train, val, test, refit_only])
        u_all, y_all = labeled["norm"].to_numpy(), labeled["label"].to_numpy()
        F_all = pd.concat([F_tr, F_va, F_te, extract_features(refit_only["norm"].to_numpy())], ignore_index=True)
        oof_all = oof_text_scores(make_text_model(SEED), u_all, y_all, labeled["reg_domain"].to_numpy())
        text_model = make_text_model(SEED).fit(u_all, y_all)
        gbm = fit_gbm(make_gbm(n_estimators=gbm_iterations(gbm)),
                      tabular_matrix(F_all, tld_vocab, oof_all), y_all)
        X_live = tabular_matrix(F_live, tld_vocab, text_model.predict_proba(u_live)[:, 1])
        p_live = _score(gbm, X_live, feature_cols, calibrator)
        report["refit_live_recall_by_feed"] = {
            src: metrics(y_live[m], p_live[m], threshold)["recall"]
            for src in sorted(live["source"].unique()) for m in [(live["source"] == src).to_numpy()]}
        report["rows"]["refit"] = len(labeled)
        print("refit live recall:", report["refit_live_recall_by_feed"])

    detector = PhishingDetector(
        text_model, gbm, tld_vocab, feature_cols, threshold, brands=get_brands(),
        calibrator=calibrator,
        metadata={"sources": sorted(df["source"].unique()), "metrics": report["test"]["stacked"],
                  "second_stage": "lightgbm", "calibrated": True},
    )
    detector.save(DEFAULT_MODEL_PATH)
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.concat(score_parts, ignore_index=True).to_parquet(MODELS_DIR / "scores.parquet", index=False)
    print(f"\nsaved {DEFAULT_MODEL_PATH} and {MODELS_DIR / 'scores.parquet'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
