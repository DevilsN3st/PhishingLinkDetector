"""Train the phishing URL detector and report metrics.

Usage: python -m phishdetect.train [--max-train N]
Reads data/processed/urls.parquet, writes models/phishing_detector.joblib and
models/metrics.json.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

from . import MODELS_DIR, PROCESSED_DIR
from .features import extract_features, get_brands
from .model import DEFAULT_MODEL_PATH, PhishingDetector, make_text_model, tabular_matrix

SEED = 42
BRAND_HOST_FEATURES = ["brand_in_subdomain", "brand_token_in_domain", "brand_embedded_in_domain",
                       "brand_typo_domain", "brand_typo_subdomain"]
N_TLDS = 250  # HistGradientBoosting supports at most 255 categories


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
    }
    out["roc_auc"] = roc_auc_score(y, p)
    out["pr_auc"] = average_precision_score(y, p)
    out["false_positive_rate"] = float(pred[y == 0].mean())
    return {k: round(float(v), 4) if isinstance(v, float) else v for k, v in out.items()}


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
    print(f"train={len(train):,d} val={len(val):,d} test={len(test):,d} live={len(live):,d}")

    y_tr, y_va, y_te, y_live = (d["label"].to_numpy() for d in (train, val, test, live))
    u_tr, u_va, u_te, u_live = (d["norm"].to_numpy() for d in (train, val, test, live))

    print("[1/4] lexical features")
    F_tr, F_va, F_te, F_live = (extract_features(u) for u in (u_tr, u_va, u_te, u_live))
    tld_vocab = {t: i + 1 for i, t in enumerate(F_tr["tld"].value_counts().index[:N_TLDS])}

    print("[2/4] char n-gram model (out-of-fold for stacking)")
    text_model = make_text_model(SEED)
    oof = oof_text_scores(text_model, u_tr, y_tr, train["reg_domain"].to_numpy())
    text_model.fit(u_tr, y_tr)
    ts_va, ts_te, ts_live = (text_model.predict_proba(u)[:, 1] for u in (u_va, u_te, u_live))

    def make_gbm(max_iter=None):
        return HistGradientBoostingClassifier(
            max_iter=max_iter or 1500, learning_rate=0.1, max_leaf_nodes=127, min_samples_leaf=40,
            l2_regularization=1.0, categorical_features=["tld"], early_stopping=max_iter is None,
            n_iter_no_change=50, random_state=SEED)

    print("[3/4] gradient boosting on lexical features only (baseline)")
    X_tr0, X_va0, X_te0 = (tabular_matrix(F, tld_vocab) for F in (F_tr, F_va, F_te))
    gbm_lex = make_gbm().fit(X_tr0, y_tr, X_val=X_va0, y_val=y_va)

    print("[4/4] stacked gradient boosting (lexical features + char n-gram score)")
    X_tr, X_va, X_te, X_live = (tabular_matrix(F, tld_vocab, s)
                                for F, s in ((F_tr, oof), (F_va, ts_va), (F_te, ts_te), (F_live, ts_live)))
    gbm = make_gbm().fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    p_va, p_te, p_live = (gbm.predict_proba(X)[:, 1] for X in (X_va, X_te, X_live))
    threshold = best_f1_threshold(y_va, p_va)
    brand_slice = (F_te[BRAND_HOST_FEATURES].sum(axis=1) > 0).to_numpy()

    report = {
        "threshold": round(threshold, 4),
        "test": {
            "char_ngram_only": metrics(y_te, ts_te, best_f1_threshold(y_va, ts_va)),
            "lexical_gbm_only": metrics(y_te, gbm_lex.predict_proba(X_te0)[:, 1],
                                        best_f1_threshold(y_va, gbm_lex.predict_proba(X_va0)[:, 1])),
            "stacked": metrics(y_te, p_te, threshold),
        },
        "test_by_source": {src: metrics(y_te[m], p_te[m], threshold)
                           for src in sorted(test["source"].unique())
                           for m in [(test["source"] == src).to_numpy()]},
        "test_brand_impersonation_slice": metrics(y_te[brand_slice], p_te[brand_slice], threshold),
        "live_recall_by_feed": {src: metrics(y_live[m], p_live[m], threshold)["recall"]
                                for src in sorted(live["source"].unique())
                                for m in [(live["source"] == src).to_numpy()]},
        "rows": {"train": len(train), "val": len(val), "test": len(test), "live": len(live)},
        "boosting_iterations": {"lexical": int(gbm_lex.n_iter_), "stacked": int(gbm.n_iter_)},
    }
    print(json.dumps(report, indent=2))

    if not args.no_refit:
        # Metrics above come from held-out sites; the shipped model additionally
        # learns from val/test so large platforms (e.g. google.com, which the
        # grouped split puts wholly in one side) are covered.
        print("[refit] text model + gradient boosting on train+val+test")
        del X_tr0, X_va0, X_te0, gbm_lex
        refit_only = df[df["split"] == "refit_only"]
        labeled = pd.concat([train, val, test, refit_only])
        u_all, y_all = labeled["norm"].to_numpy(), labeled["label"].to_numpy()
        F_all = pd.concat([F_tr, F_va, F_te, extract_features(refit_only["norm"].to_numpy())], ignore_index=True)
        oof_all = oof_text_scores(make_text_model(SEED), u_all, y_all, labeled["reg_domain"].to_numpy())
        text_model = make_text_model(SEED).fit(u_all, y_all)
        gbm = make_gbm(max_iter=gbm.n_iter_).fit(tabular_matrix(F_all, tld_vocab, oof_all), y_all)
        X_live = tabular_matrix(F_live, tld_vocab, text_model.predict_proba(u_live)[:, 1])
        p_live = gbm.predict_proba(X_live)[:, 1]
        report["refit_live_recall_by_feed"] = {
            src: metrics(y_live[m], p_live[m], threshold)["recall"]
            for src in sorted(live["source"].unique()) for m in [(live["source"] == src).to_numpy()]}
        report["rows"]["refit"] = len(labeled)
        print("refit live recall:", report["refit_live_recall_by_feed"])

    detector = PhishingDetector(text_model, gbm, tld_vocab, list(X_tr.columns), threshold, brands=get_brands(),
                                metadata={"sources": sorted(df["source"].unique()), "metrics": report["test"]["stacked"]})
    detector.save(DEFAULT_MODEL_PATH)
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {DEFAULT_MODEL_PATH}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
