"""Does a frozen URLBERT-tiny encoder add anything on top of the production stack?

Trains a linear probe on 100k training URLs, then a 2-feature logistic
combination of (production score, urlbert score) on val. Reports test metrics.
Does not change the shipped model.

Usage: python urlbert_stack.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishdetect import PROCESSED_DIR  # noqa: E402
from phishdetect.model import DEFAULT_MODEL_PATH, PhishingDetector  # noqa: E402

REPO = "CrabInHoney/urlbert-tiny-v6"
TRAIN_N, VAL_N, TEST_N = 20_000, 8_000, 8_000
BATCH = 32
MAX_LEN = 64
SEED = 42


@torch.no_grad()
def embed(urls, tok, model, device):
    vecs = []
    model.eval()
    urls = [str(u) if u is not None else "" for u in urls]
    for i in range(0, len(urls), BATCH):
        batch = urls[i:i + BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state
        cls = out[:, 0]
        mask = enc["attention_mask"].unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        vecs.append(torch.cat([cls, mean], dim=1).cpu().numpy())
        if i and i % (BATCH * 50) == 0:
            print(f"    embedded {i}/{len(urls)}", flush=True)
    return np.vstack(vecs)


def main():
    # MPS segfaults on this ModernBERT checkpoint under PyTorch 2.x / Python 3.14.
    device = "cpu"
    df = pd.read_parquet(PROCESSED_DIR / "urls.parquet", columns=["norm", "label", "split"])
    rng = np.random.default_rng(SEED)
    splits = {}
    for name, n in (("train", TRAIN_N), ("val", VAL_N), ("test", TEST_N)):
        part = df[df["split"] == name]
        idx = rng.choice(len(part), size=min(n, len(part)), replace=False)
        splits[name] = part.iloc[idx].reset_index(drop=True)
        print(f"{name}={len(splits[name]):,d}", flush=True)

    tok = AutoTokenizer.from_pretrained(REPO)
    enc = AutoModel.from_pretrained(REPO).to(device)
    X = {s: embed(d["norm"].tolist(), tok, enc, device) for s, d in splits.items()}
    y = {s: d["label"].to_numpy() for s, d in splits.items()}

    probe = LogisticRegression(max_iter=300, C=1.0).fit(X["train"], y["train"])
    ub = {s: probe.predict_proba(X[s])[:, 1] for s in splits}

    det = PhishingDetector.load(DEFAULT_MODEL_PATH)
    prod = {s: det.predict_proba(d["norm"].tolist()) for s, d in splits.items()}

    combo = LogisticRegression(max_iter=200).fit(
        np.column_stack([prod["val"], ub["val"]]), y["val"])
    stacked = combo.predict_proba(np.column_stack([prod["test"], ub["test"]]))[:, 1]

    def pack(scores):
        return {"roc_auc": round(float(roc_auc_score(y["test"], scores)), 4),
                "pr_auc": round(float(average_precision_score(y["test"], scores)), 4)}

    report = {
        "urlbert_probe": pack(ub["test"]),
        "production": pack(prod["test"]),
        "prod_plus_urlbert": pack(stacked),
        "combo_weights": {"production": float(combo.coef_[0, 0]), "urlbert": float(combo.coef_[0, 1])},
        "n": {s: len(splits[s]) for s in splits},
        "device": device,
    }
    print(json.dumps(report, indent=2))
    Path(__file__).resolve().parent.joinpath("results", "urlbert_stack.json").write_text(
        json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
