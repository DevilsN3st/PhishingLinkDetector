import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

from . import MODELS_DIR
from .features import encode_tld, extract_features, get_brands, normalize_url, set_brands

DEFAULT_MODEL_PATH = MODELS_DIR / "phishing_detector.joblib"


def make_text_model(seed: int = 42):
    """Character n-gram TF-IDF + linear classifier over the normalized URL."""
    return make_pipeline(
        HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2**21,
                          alternate_sign=False, norm=None, dtype=np.float32),
        TfidfTransformer(sublinear_tf=True),
        SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=15, tol=1e-4, random_state=seed),
    )


def tabular_matrix(features: pd.DataFrame, tld_vocab: dict, text_score=None) -> pd.DataFrame:
    X = features.copy()
    X["tld"] = encode_tld(X["tld"], tld_vocab)
    if text_score is not None:
        X["text_score"] = text_score
    return X


class PhishingDetector:
    """Stacked model: char n-gram score + lexical features -> gradient boosting."""

    def __init__(self, text_model, gbm, tld_vocab, feature_cols, threshold=0.5, brands=None, metadata=None):
        self.text_model = text_model
        self.gbm = gbm
        self.tld_vocab = tld_vocab
        self.feature_cols = feature_cols
        self.threshold = threshold
        self.brands = frozenset(brands) if brands is not None else get_brands()
        self.metadata = metadata or {}

    def predict_proba(self, urls) -> np.ndarray:
        brands = getattr(self, "brands", None)
        if brands is not None and get_brands() != brands:
            set_brands(brands)
        norms = [normalize_url(u) for u in urls]
        text_score = self.text_model.predict_proba(norms)[:, 1]
        X = tabular_matrix(extract_features(norms), self.tld_vocab, text_score)
        return self.gbm.predict_proba(X[self.feature_cols])[:, 1]

    def predict(self, urls) -> np.ndarray:
        return (self.predict_proba(urls) >= self.threshold).astype(int)

    def save(self, path=DEFAULT_MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path=DEFAULT_MODEL_PATH) -> "PhishingDetector":
        return joblib.load(path)
