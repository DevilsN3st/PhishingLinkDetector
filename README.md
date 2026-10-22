# Phishing Link Detector

URL-only classifier that scores a link **before anyone visits it**. No page fetch, DNS, WHOIS, or TLS lookup at inference.

It is a two-stage stack: a linear model over character n-grams, then LightGBM on lexical / brand-impersonation features plus that n-gram score. The decision threshold is 0.462 (best F1 on validation). Scores are Platt-calibrated.

```
URL  →  normalize  →  char 3–5 grams (hashed TF-IDF + logistic)  →  text_score
                  ↘  ~76 lexical + brand features + TLD               ↘
                         LightGBM  →  P(phishing)  →  threshold 0.462
```

Full design notes: [docs/architecture.md](docs/architecture.md).  
Full model comparison: [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Results (held-out)

Trained on **2,399,377** URLs. Test is **280,337** URLs, grouped by registered domain so the same site never appears in both train and test.

| Set | n | ROC AUC | F1 | Notes |
|---|---|---|---|---|
| In-domain test | 280,337 | 0.970 | 0.909 | FPR 9.8% at the F1 threshold |
| Brand-impersonation slice | 39,983 | 0.987 | 0.959 | Brand in subdomain / typo / embedding |
| PhreshPhish test (never trained on) | 163,388 | 0.988 | 0.937 | Unseen-domain subset: 0.974 AUC |
| JPxxx fold 0 (9:1 benign) | 82,924 | 0.939 | — | F1 is misleading on this mix; recall @ 1% FPR = 0.71 |
| DeepURLBench 2022–23 sample | 200,000 | 0.945 | — | Eval only (CC-BY-NC). Raise the threshold if FPR must stay low |
| Live OpenPhish / phishunt / PhishStats | 2,368 | — | recall 0.99 / 0.99 / 0.96 | Feeds held out of training |

A 13-model comparison on the same splits is in `benchmarks/RESULTS.md`. The stack beats a URLNet-style char CNN (0.961) and matches DistilBERT trained on 200k URLs (0.970), at ~1.8k URLs/s and 13 MB vs 496 URLs/s and 268 MB.

## Quick start

Needs Python 3.11+, and on macOS `brew install libomp` (LightGBM).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# score with a trained bundle (models/phishing_detector.joblib)
python -m phishdetect.predict https://paypal.com/signin \
    "https://carousell.order-distribution.com/?I-sEwkdawKvin"
```

```
 0.031  legit      https://paypal.com/signin
 0.975  PHISHING   https://carousell.order-distribution.com/?I-sEwkdawKvin
```

`--file urls.txt` reads one URL per line; `--json` prints JSON lines. `--threshold` overrides the bundled cut.

The `.joblib` bundle is gitignored (~13 MB). Train locally to create it (below).

## Rebuild data and model

```bash
python -m phishdetect.download          # public feeds + Hugging Face / Kaggle
python -m phishdetect.build_dataset     # merge, de-dupe, domain-grouped split
python -m phishdetect.train             # writes models/phishing_detector.joblib
```

`download` skips files already in `data/raw/`. Some Kaggle sources need a Kaggle API token. Sources that fail are skipped; the rest still build.

`train` reports test / live / external metrics to `models/metrics.json`, writes per-URL scores to `models/scores.parquet` (gitignored), then **refits** on train+val+test so large platforms the grouped split put entirely in test (e.g. `google.com`) are in the shipped model. `--no-refit` ships the evaluation model instead.

## How the data is split

| Split | Role |
|---|---|
| `train` / `val` / `test` | Grouped by registered domain (`GroupShuffleSplit`) |
| `live` | OpenPhish, phishunt, PhishStats — never trained on |
| `refit_only` | Synthetic rows whose domain is in val/test/live |
| `ext_phreshphish` | PhreshPhish's own test split |
| `ext_jpxxx` | Fold 0 of the JPxxx site-aware URL benchmark |
| `ext_deepurlbench` | 200k phishing/benign sample, 2022–23, not used in training |

URLs are normalized (no scheme, no leading `www.`, lowercase host, no trailing slash) so the model cannot cheat on source-specific formatting. Label conflicts are resolved by dropping two low-trust Kaggle/HF dumps when they disagree with everyone else.

Legitimate domains come from **Tranco ∩ Majestic** (Tranco alone now ranks a lot of spam). Phishing comes from blocklists, research dumps, and PhreshPhish. About 40k synthetic brand-lure pairs teach “brand name on a domain the brand does not own,” each paired with a real-brand login URL so the model cannot flag every `signin` path.

## Project layout

```
phishdetect/
  download.py         fetch raw sources
  build_dataset.py    merge, conflict resolution, splits
  features.py         normalize + lexical / brand features
  brands.py           curated brands + Tranco second-level labels
  synthetic.py        brand-impersonation templates
  model.py            text model, LightGBM, calibrator, PhishingDetector
  train.py            OOF stacking, threshold, refit, metrics
  predict.py          CLI
  eval_external.py    score an extra parquet against the saved model
benchmarks/           shared protocol + classic / CNN / DistilBERT runners
models/metrics.json   last training report (committed)
```

## Benchmarks

Same splits, same F1-on-val threshold, same test set:

```bash
cd benchmarks
python classic.py                 # 13 classic / stacked models on full train
python classic.py --subsample     # 200k-row train
python neural.py --epochs 3       # char CNN
python transformer.py --subsample --epochs 1 --eval-cap 80000
python report.py                  # writes RESULTS.md
```

## Limitations

- **URL string only.** A real `paypal.com/...` path after account takeover looks legitimate. An optional HTML/TLS second pass would cover that; this repo does not fetch pages.
- **Operating point.** 0.462 maximises F1 on a ~50/50 val set. On low-prevalence traffic (DeepURLBench, JPxxx) that cut over-predicts. Raise the threshold, or use recall @ 1% FPR (0.67 on in-domain test).
- **Scam / abuse domains** without phishing-kit URL structure are weaker (scam-list recall was ~0.37 on test).
- **Google Forms / free-host phishing** that sits on a trusted apex can still be flagged or missed depending on the path.
- DeepURLBench is CC-BY-NC and **evaluation-only**.

## License

Use the detector for defensive / research purposes. Upstream datasets keep their own licenses (PhreshPhish CC-BY-4.0 for anti-phishing research; DeepURLBench CC-BY-NC).
