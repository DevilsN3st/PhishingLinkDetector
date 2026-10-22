"""Character-level CNN (URLNet-style character branch) on the shared protocol.

Usage:
  python benchmarks/neural.py --subsample --epochs 6
  python benchmarks/neural.py --epochs 3
"""

import argparse
import copy

import numpy as np
import torch
from common import EVAL_SPLITS, SEED, Timer, evaluate, load_splits, throughput
from sklearn.metrics import roc_auc_score
from torch import nn

from phishdetect.features import normalize_url

MAX_LEN = 200
VOCAB = 98  # 0 = pad, 1 = unknown, 2..96 = printable ASCII 32..126
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode(urls) -> np.ndarray:
    out = np.zeros((len(urls), MAX_LEN), dtype=np.uint8)
    for i, u in enumerate(urls):
        b = np.frombuffer(u[:MAX_LEN].encode("ascii", "replace"), dtype=np.uint8)
        out[i, :len(b)] = b
    printable = (out >= 32) & (out <= 126)
    return np.where(printable, out - 30, np.where(out > 0, 1, 0)).astype(np.uint8)


class CharCNN(nn.Module):
    def __init__(self, emb=32, filters=256, kernels=(3, 4, 5, 6), hidden=256, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, k) for k in kernels)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(filters * len(kernels), hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(h)).amax(dim=2) for c in self.convs], dim=1)
        return self.head(h).squeeze(1)


@torch.no_grad()
def predict(model, X: np.ndarray, batch=4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(DEVICE).long()
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    args = p.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    S = load_splits(args.subsample)
    y = {k: S[k]["label"].to_numpy() for k in S}
    X = {k: encode(S[k]["norm"].tolist()) for k in S}
    n_tr = len(S["train"])
    print(f"device={DEVICE} train={n_tr:,d}", flush=True)

    model = CharCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((n_tr + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    y_tr = torch.from_numpy(y["train"].astype(np.float32))

    best_auc, best_state = -1.0, None
    with Timer() as t:
        for epoch in range(args.epochs):
            model.train()
            perm = np.random.permutation(n_tr)
            total = 0.0
            for i in range(0, n_tr, args.batch):
                idx = perm[i:i + args.batch]
                xb = torch.from_numpy(X["train"][idx]).to(DEVICE).long()
                yb = y_tr[idx].to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            val_auc = roc_auc_score(y["val"], predict(model, X["val"]))
            print(f"  epoch {epoch + 1}: loss={total / n_tr:.4f} val_auc={val_auc:.4f}", flush=True)
            if val_auc > best_auc:
                best_auc, best_state = val_auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    scores = {k: predict(model, X[k]) for k in EVAL_SPLITS}
    n_params = sum(p.numel() for p in model.parameters())
    evaluate("char_cnn" + ("_200k" if args.subsample else ""), "neural (char CNN)", S, scores,
             train_seconds=t.seconds,
             urls_per_sec=throughput(lambda urls: predict(model, encode([normalize_url(u) for u in urls])),
                                     S["test"]["url"].to_numpy()),
             size_mb=n_params * 4 / 1e6, train_rows=n_tr,
             notes=f"{n_params / 1e6:.2f}M params, {args.epochs} epochs, best val epoch kept, {DEVICE}")


if __name__ == "__main__":
    main()
