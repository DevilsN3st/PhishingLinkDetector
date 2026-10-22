"""Fine-tuned transformer on the shared protocol.

Usage:
  python benchmarks/transformer.py --model distilbert-base-uncased --subsample --epochs 1 --eval-cap 60000
  python benchmarks/transformer.py --model CrabInHoney/urlbert-tiny-v6 --name urlbert_tiny \
      --batch 256 --lr 3e-4 --epochs 2
"""

import argparse
import copy

import numpy as np
import torch
from common import EVAL_SPLITS, SEED, Timer, evaluate, load_splits, throughput
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from phishdetect.features import normalize_url

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def tokenize(tok, texts, max_len, chunk=200_000):
    ids, mask = [], []
    texts = list(texts)
    for i in range(0, len(texts), chunk):
        enc = tok(texts[i:i + chunk], truncation=True, max_length=max_len, padding="max_length", return_tensors="np")
        ids.append(enc["input_ids"].astype(np.int32))
        mask.append(enc["attention_mask"].astype(np.int32))
    return np.concatenate(ids), np.concatenate(mask)


def _batch(ids, mask, idx):
    return (torch.from_numpy(ids[idx]).to(DEVICE).long(), torch.from_numpy(mask[idx]).to(DEVICE).long())


@torch.no_grad()
def predict(model, ids, mask, batch=1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(ids), batch):
        b_ids, b_mask = _batch(ids, mask, slice(i, i + batch))
        out.append(torch.sigmoid(model(input_ids=b_ids, attention_mask=b_mask).logits.squeeze(-1)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--name", default="distilbert")
    p.add_argument("--subsample", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--eval-cap", type=int, default=None, help="score at most N rows per val/test/external split")
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    S = load_splits(args.subsample, eval_cap=args.eval_cap)
    y = {k: S[k]["label"].to_numpy() for k in S}
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = {k: tokenize(tok, S[k]["norm"], args.max_len) for k in S}
    n_tr = len(S["train"])
    print(f"device={DEVICE} model={args.model} train={n_tr:,d}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((n_tr + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    ids_tr, mask_tr = enc["train"]
    y_tr = torch.from_numpy(y["train"].astype(np.float32))

    best_auc, best_state = -1.0, None
    with Timer() as t:
        for epoch in range(args.epochs):
            model.train()
            perm = np.random.permutation(n_tr)
            total, step = 0.0, 0
            for i in range(0, n_tr, args.batch):
                idx = perm[i:i + args.batch]
                b_ids, b_mask = _batch(ids_tr, mask_tr, idx)
                logits = model(input_ids=b_ids, attention_mask=b_mask).logits.squeeze(-1)
                loss = loss_fn(logits, y_tr[idx].to(DEVICE))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
                step += 1
                if step % args.log_every == 0:
                    print(f"    step {step}/{steps // args.epochs}  loss={total / (i + len(idx)):.4f}  "
                          f"{t.elapsed:.0f}s", flush=True)
            val_auc = roc_auc_score(y["val"], predict(model, *enc["val"]))
            print(f"  epoch {epoch + 1}: loss={total / n_tr:.4f} val_auc={val_auc:.4f}", flush=True)
            if val_auc > best_auc:
                best_auc, best_state = val_auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    scores = {k: predict(model, *enc[k]) for k in EVAL_SPLITS}
    n_params = sum(p.numel() for p in model.parameters())

    def score_raw(urls):
        return predict(model, *tokenize(tok, [normalize_url(u) for u in urls], args.max_len))

    evaluate(args.name + ("_200k" if args.subsample else ""), "transformer (fine-tuned)", S, scores,
             train_seconds=t.seconds, urls_per_sec=throughput(score_raw, S["test"]["url"].to_numpy()),
             size_mb=n_params * 4 / 1e6, train_rows=n_tr,
             notes=f"{args.model}, {n_params / 1e6:.1f}M params, {args.epochs} epoch(s), max_len {args.max_len}, "
                   f"{DEVICE}" + (f", eval capped at {args.eval_cap:,d} rows/split" if args.eval_cap else ""))


if __name__ == "__main__":
    main()
