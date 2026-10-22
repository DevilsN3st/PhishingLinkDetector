import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline

from . import MODELS_DIR
from .features import encode_tld, extract_features, get_brands, normalize_url, set_brands

DEFAULT_MODEL_PATH = MODELS_DIR / "phishing_detector.joblib"
N_TLDS = 250


def make_text_model(seed: int = 42):
    """Character n-gram TF-IDF + linear classifier over the normalized URL."""
    return make_pipeline(
        HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2**21,
                          alternate_sign=False, norm=None, dtype=np.float32),
        TfidfTransformer(sublinear_tf=True),
        SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=15, tol=1e-4, random_state=seed),
    )


def make_gbm(n_estimators: int | None = None, seed: int = 42):
    """LightGBM second stage. n_estimators=None means 'use 1500 + early stopping'."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=n_estimators or 1500, learning_rate=0.1, num_leaves=127, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )


def as_gbm_frame(X: pd.DataFrame, n_tlds: int = N_TLDS) -> pd.DataFrame:
    """LightGBM wants TLD as a pandas category with a fixed set of codes."""
    out = X.copy()
    out["tld"] = pd.Categorical(out["tld"].astype(int), categories=range(n_tlds + 1))
    return out


def fit_gbm(gbm, X_tr, y_tr, X_va=None, y_va=None):
    import lightgbm as lgb

    X_tr = as_gbm_frame(X_tr)
    if X_va is None:
        gbm.fit(X_tr, y_tr, categorical_feature=["tld"])
        return gbm
    X_va = as_gbm_frame(X_va)
    gbm.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=["tld"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return gbm


def gbm_iterations(gbm) -> int:
    return int(getattr(gbm, "best_iteration_", None) or gbm.n_estimators)


def tabular_matrix(features: pd.DataFrame, tld_vocab: dict, text_score=None, extra=None) -> pd.DataFrame:
    X = features.copy()
    X["tld"] = encode_tld(X["tld"], tld_vocab)
    if text_score is not None:
        X["text_score"] = text_score
    if extra:
        for name, values in extra.items():
            X[name] = values
    return X


class PlattCalibrator:
    """Maps raw booster scores to probabilities using logistic regression on val."""

    def __init__(self):
        self.lr = LogisticRegression(solver="lbfgs", max_iter=200)

    def fit(self, scores, y):
        self.lr.fit(np.asarray(scores, dtype=np.float64).reshape(-1, 1), y)
        return self

    def transform(self, scores):
        return self.lr.predict_proba(np.asarray(scores, dtype=np.float64).reshape(-1, 1))[:, 1]


class PhishingDetector:
    """Stacked model: char n-gram score + lexical features -> gradient boosting."""

    def __init__(self, text_model, gbm, tld_vocab, feature_cols, threshold=0.5, brands=None,
                 calibrator=None, metadata=None):
        self.text_model = text_model
        self.gbm = gbm
        self.tld_vocab = tld_vocab
        self.feature_cols = feature_cols
        self.threshold = threshold
        self.brands = frozenset(brands) if brands is not None else get_brands()
        self.calibrator = calibrator
        self.metadata = metadata or {}

    def _raw_score(self, norms) -> np.ndarray:
        text_score = self.text_model.predict_proba(norms)[:, 1]
        X = as_gbm_frame(tabular_matrix(extract_features(norms), self.tld_vocab, text_score))
        return self.gbm.predict_proba(X[self.feature_cols])[:, 1]

    def predict_proba(self, urls) -> np.ndarray:
        brands = getattr(self, "brands", None)
        if brands is not None and get_brands() != brands:
            set_brands(brands)
        norms = [normalize_url(u) for u in urls]
        raw = self._raw_score(norms)
        cal = getattr(self, "calibrator", None)
        return cal.transform(raw) if cal is not None else raw

    def predict(self, urls) -> np.ndarray:
        return (self.predict_proba(urls) >= self.threshold).astype(int)

    def save(self, path=DEFAULT_MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path=DEFAULT_MODEL_PATH) -> "PhishingDetector":
        return joblib.load(path)
