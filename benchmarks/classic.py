"""Classic ML baselines on the shared protocol.

Usage:
  python benchmarks/classic.py                 # full train split
  python benchmarks/classic.py --subsample     # 200k-row train subsample
  python benchmarks/classic.py --models lr_ngrams,lgbm_features
"""

import argparse
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from common import (EVAL_SPLITS, SEED, Timer, brand_slice, evaluate, features, load_splits, size_mb,
                    throughput, tld_vocab)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from phishdetect.features import encode_tld, extract_features, normalize_url

NGRAM_MODELS = ["lr_ngrams", "nb_ngrams", "svm_ngrams"]
FEATURE_MODELS = ["lr_features", "rf_features", "hgb_features", "xgb_features", "lgbm_features"]
STACKED_MODELS = ["stacked_hgb", "stacked_lgbm", "stacked_xgb"]
# stacked_lgbm variants: per-domain down-weighting, and a time-split model that
# never sees PhreshPhish rows collected on/after RECENT_CUTOFF.
VARIANT_MODELS = ["stacked_lgbm_domw", "stacked_lgbm_norecent"]
ALL_MODELS = NGRAM_MODELS + FEATURE_MODELS + STACKED_MODELS + VARIANT_MODELS
SUBSAMPLE_MODELS = ["lr_ngrams", "hgb_features", "lgbm_features", "stacked_hgb", "stacked_lgbm"]
DOMAIN_CAP = 1000
RECENT_CUTOFF = pd.Timestamp("2025-07-01")
N_TLDS_OH = 251

VECTORIZER = HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2**21,
                               alternate_sign=False, norm=None, dtype=np.float32)


def sgd_lr():
    return SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=15, tol=1e-4, random_state=SEED)


def as_tree_frame(F, vocab, text_score=None):
    X = F.copy()
    X["tld"] = encode_tld(X["tld"], vocab)
    if text_score is not None:
        X["text_score"] = text_score
    return X


def as_category_frame(F, vocab, text_score=None):
    X = as_tree_frame(F, vocab, text_score)
    X["tld"] = pd.Categorical(X["tld"], categories=range(len(vocab) + 1))
    return X


def hgb():
    return HistGradientBoostingClassifier(
        max_iter=1500, learning_rate=0.1, max_leaf_nodes=127, min_samples_leaf=40, l2_regularization=1.0,
        categorical_features=["tld"], early_stopping=True, n_iter_no_change=50, random_state=SEED)


def lgbm():
    return lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.1, num_leaves=127, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=SEED, n_jobs=-1, verbose=-1)


def xgbc():
    return xgb.XGBClassifier(n_estimators=1500, learning_rate=0.1, max_depth=0, max_leaves=127,
                             grow_policy="lossguide", min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                             early_stopping_rounds=50, eval_metric="logloss", random_state=SEED, n_jobs=-1)


def fit_tree(kind, X_tr, y_tr, X_va, y_va, weight=None):
    if kind == "hgb":
        return hgb().fit(X_tr, y_tr, sample_weight=weight, X_val=X_va, y_val=y_va)
    if kind == "lgbm":
        return lgbm().fit(X_tr, y_tr, sample_weight=weight, eval_set=[(X_va, y_va)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
    return xgbc().fit(X_tr, y_tr, sample_weight=weight, eval_set=[(X_va, y_va)], verbose=False)


def domain_weights(train: pd.DataFrame, cap: int) -> np.ndarray:
    n = train.groupby(["reg_domain", "label"])["norm"].transform("size").to_numpy()
    return np.minimum(1.0, cap / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", action="store_true")
    p.add_argument("--models", default="")
    args = p.parse_args()
    models = args.models.split(",") if args.models else (SUBSAMPLE_MODELS if args.subsample else ALL_MODELS)
    suffix = "_200k" if args.subsample else ""

    S = load_splits(args.subsample)
    F = features(S, args.subsample)
    y = {k: S[k]["label"].to_numpy() for k in S}
    u = {k: S[k]["norm"].to_numpy() for k in S}
    vocab = tld_vocab(F["train"])
    raw_test = S["test"]["url"].to_numpy()
    n_tr = len(S["train"])
    brand_mask = brand_slice(F["test"])
    print(f"train={n_tr:,d}  models={models}", flush=True)

    def report(name, family, model_obj, scores, seconds, score_fn, notes="", train_rows=n_tr):
        evaluate(name + suffix, family, S, scores, train_seconds=seconds, urls_per_sec=throughput(score_fn, raw_test),
                 size_mb=size_mb(model_obj), train_rows=train_rows, notes=notes, brand_mask=brand_mask)

    need_ngrams = any(m in models for m in NGRAM_MODELS + STACKED_MODELS + VARIANT_MODELS)
    if need_ngrams:
        with Timer() as t_vec:
            tfidf = TfidfTransformer(sublinear_tf=True)
            N = {"train": tfidf.fit_transform(VECTORIZER.transform(u["train"]))}
            for k in EVAL_SPLITS:
                N[k] = tfidf.transform(VECTORIZER.transform(u[k]))
        print(f"n-gram matrices built in {t_vec.seconds:.0f}s (nnz train={N['train'].nnz:,d})", flush=True)

        def ngram_scorer(clf, kind="proba"):
            def fn(urls):
                X = tfidf.transform(VECTORIZER.transform([normalize_url(x) for x in urls]))
                return clf.predict_proba(X)[:, 1] if kind == "proba" else clf.decision_function(X)
            return fn

        def ngram_scores(clf, kind="proba"):
            f = (lambda X: clf.predict_proba(X)[:, 1]) if kind == "proba" else clf.decision_function
            return {k: f(N[k]) for k in EVAL_SPLITS}

    # --- n-gram text models -------------------------------------------------
    for name, make, kind in (("lr_ngrams", sgd_lr, "proba"),
                             ("nb_ngrams", lambda: ComplementNB(alpha=0.1), "proba"),
                             ("svm_ngrams", lambda: LinearSVC(C=0.5, tol=1e-3, max_iter=2000, random_state=SEED),
                              "decision")):
        if name not in models:
            continue
        with Timer() as t:
            clf = make().fit(N["train"], y["train"])
        report(name, "text n-grams", (tfidf, clf), ngram_scores(clf, kind), t.seconds + t_vec.seconds,
               ngram_scorer(clf, kind), notes="char 3-5 grams, hashed 2^21, TF-IDF")
        del clf
        gc.collect()

    # --- feature models -----------------------------------------------------
    def feature_scorer(predict, frame_fn):
        def fn(urls):
            Fx = extract_features([normalize_url(x) for x in urls])
            return predict(frame_fn(Fx))
        return fn

    if "lr_features" in models:
        num_cols = [c for c in F["train"].columns if c != "tld"]
        pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                                 ("tld", OneHotEncoder(handle_unknown="ignore", max_categories=N_TLDS_OH), ["tld"])])
        clf = make_pipeline(pre, LogisticRegression(max_iter=500, C=1.0))
        with Timer() as t:
            clf.fit(F["train"], y["train"])
        sc = {k: clf.predict_proba(F[k])[:, 1] for k in EVAL_SPLITS}
        report("lr_features", "lexical features", clf, sc, t.seconds,
               feature_scorer(lambda X: clf.predict_proba(X)[:, 1], lambda Fx: Fx), notes="standardized + one-hot TLD")
        del clf
        gc.collect()

    if "rf_features" in models:
        X_tree = {k: as_tree_frame(F[k], vocab) for k in F}
        clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features="sqrt", max_samples=0.5,
                                     n_jobs=-1, random_state=SEED)
        with Timer() as t:
            clf.fit(X_tree["train"], y["train"])
        sc = {k: clf.predict_proba(X_tree[k])[:, 1] for k in EVAL_SPLITS}
        report("rf_features", "lexical features", clf, sc, t.seconds,
               feature_scorer(lambda X: clf.predict_proba(X)[:, 1], lambda Fx: as_tree_frame(Fx, vocab)),
               notes="150 trees, TLD as ordinal code")
        del clf, X_tree
        gc.collect()

    for name, kind in (("hgb_features", "hgb"), ("lgbm_features", "lgbm"), ("xgb_features", "xgb")):
        if name not in models:
            continue
        frame = as_tree_frame if kind == "hgb" else as_category_frame
        Xs = {k: frame(F[k], vocab) for k in F}
        with Timer() as t:
            clf = fit_tree(kind, Xs["train"], y["train"], Xs["val"], y["val"])
        sc = {k: clf.predict_proba(Xs[k])[:, 1] for k in EVAL_SPLITS}
        report(name, "lexical features", clf, sc, t.seconds,
               feature_scorer(lambda X, c=clf: c.predict_proba(X)[:, 1], lambda Fx, f=frame: f(Fx, vocab)))
        del clf
        gc.collect()

    # --- stacked: n-gram LR score + lexical features -> boosted trees --------
    groups_all = S["train"]["reg_domain"].to_numpy()

    def text_stage(mask):
        """Out-of-fold text scores on train[mask] plus a text model fit on all of train[mask]."""
        idx = np.flatnonzero(mask)
        Ntr, ytr = N["train"][idx], y["train"][idx]
        oof = np.zeros(len(idx), dtype=np.float32)
        for tr, va in GroupKFold(n_splits=3).split(Ntr, ytr, groups_all[idx]):
            oof[va] = sgd_lr().fit(Ntr[tr], ytr[tr]).predict_proba(Ntr[va])[:, 1]
        clf = sgd_lr().fit(Ntr, ytr)
        return idx, {"train": oof, **{k: clf.predict_proba(N[k])[:, 1] for k in EVAL_SPLITS}}, clf

    def run_stacked(name, kind, idx, ts, text_clf, t_text, weight=None, notes=""):
        frame = as_tree_frame if kind == "hgb" else as_category_frame
        X_tr = frame(F["train"].iloc[idx].reset_index(drop=True), vocab, ts["train"])
        Xs = {k: frame(F[k], vocab, ts[k]) for k in EVAL_SPLITS}
        with Timer() as t:
            clf = fit_tree(kind, X_tr, y["train"][idx], Xs["val"], y["val"], weight)
        sc = {k: clf.predict_proba(Xs[k])[:, 1] for k in EVAL_SPLITS}

        def fn(urls, c=clf, f=frame):
            norms = [normalize_url(x) for x in urls]
            s = text_clf.predict_proba(tfidf.transform(VECTORIZER.transform(norms)))[:, 1]
            return c.predict_proba(f(extract_features(norms), vocab, s))[:, 1]

        report(name, "stacked (text + features)", (tfidf, text_clf, clf), sc, t.seconds + t_text + t_vec.seconds,
               fn, notes=notes, train_rows=len(idx))

    stacked = [m for m in STACKED_MODELS + ["stacked_lgbm_domw"] if m in models]
    if stacked:
        with Timer() as t_text:
            idx, ts, text_clf = text_stage(np.ones(n_tr, dtype=bool))
        print(f"stacking text scores in {t_text.seconds:.0f}s", flush=True)
        for name in stacked:
            if name == "stacked_lgbm_domw":
                w = domain_weights(S["train"], DOMAIN_CAP)
                run_stacked(name, "lgbm", idx, ts, text_clf, t_text.seconds, weight=w,
                            notes=f"rows per (domain, label) down-weighted to at most {DOMAIN_CAP}")
            else:
                run_stacked(name, name.split("_")[1], idx, ts, text_clf, t_text.seconds,
                            notes="current production design" if name == "stacked_hgb" else "")
            gc.collect()

    if "stacked_lgbm_norecent" in models:
        tr = S["train"]
        recent = (tr["source"] == "phreshphish") & (pd.to_datetime(tr["date"]) >= RECENT_CUTOFF)
        with Timer() as t_text:
            idx, ts, text_clf = text_stage(~recent.to_numpy())
        run_stacked("stacked_lgbm_norecent", "lgbm", idx, ts, text_clf, t_text.seconds,
                    notes=f"PhreshPhish rows dated >= {RECENT_CUTOFF.date()} removed from train "
                          f"({int(recent.sum()):,d} rows)")


if __name__ == "__main__":
    main()
