"""Merge raw sources into one labeled, de-duplicated, split dataset.

Usage: python -m phishdetect.build_dataset
Output: data/processed/urls.parquet  (columns: url, norm, label, source, reg_domain, split)
        phishdetect/resources/brands.txt (brand vocabulary for the features)

label: 1 = phishing, 0 = legitimate
split: train / val / test are grouped by registered domain so the same site
never appears on both sides. "live" is the OpenPhish + phishunt feeds, held out
entirely as a fresh, out-of-distribution check. Synthetic brand-impersonation
rows are only placed in train, or in "refit_only" when their site is held out.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import PROCESSED_DIR, RAW_DIR
from .brands import CURATED_BRANDS, build_brand_list, write_brand_list
from .features import normalize_url, registered_domain, registered_label
from .synthetic import generate as generate_synthetic

SEED = 42
TRANCO_CSV = RAW_DIR / "tranco_top1m" / "top-1m.csv"
LIVE_SOURCES = {"openphish", "phishunt"}
# Kept out of synthetic generation so the URL that motivated it stays a fair test.
SYNTHETIC_EXCLUDE = ("order-distribution.com",)


def _frame(urls, labels, source):
    return pd.DataFrame({"url": urls, "label": labels, "source": source})


def _read_lines(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith(("#", "!"))]


def _sample(items, cap):
    if cap and len(items) > cap:
        return list(np.random.default_rng(SEED).choice(items, size=cap, replace=False))
    return items


def load_kaggle_phishing_site_urls():
    df = pd.read_csv(RAW_DIR / "kaggle_phishing_site_urls" / "phishing_site_urls.csv",
                     encoding="latin-1", on_bad_lines="skip")
    df = df[df["Label"].isin(["bad", "good"])]
    return _frame(df["URL"], (df["Label"] == "bad").astype(int), "kaggle_phishing_site_urls")


def load_kaggle_malicious_urls():
    df = pd.read_csv(RAW_DIR / "kaggle_malicious_urls" / "malicious_phish.csv", on_bad_lines="skip")
    # Defacement/malware are different threat classes; keep phishing vs benign only.
    df = df[df["type"].isin(["phishing", "benign"])]
    return _frame(df["url"], (df["type"] == "phishing").astype(int), "kaggle_malicious_urls")


def load_kaggle_harisudhan():
    df = pd.read_csv(RAW_DIR / "kaggle_harisudhan_urls" / "new_data_urls.csv", on_bad_lines="skip")
    # status 0 = phishing, 1 = legitimate.
    return _frame(df["url"], (df["status"] == 0).astype(int), "kaggle_harisudhan_urls")


def load_uci_phiusiil():
    df = pd.read_csv(RAW_DIR / "uci_phiusiil" / "PhiUSIIL_Phishing_URL_Dataset.csv",
                     usecols=["URL", "label"], encoding="utf-8-sig")
    # In PhiUSIIL label 1 means legitimate.
    return _frame(df["URL"], (df["label"] == 0).astype(int), "uci_phiusiil")


def load_hf_pirocheto():
    parts = [pd.read_parquet(RAW_DIR / f, columns=["url", "status"])
             for f in ("hf_pirocheto_train.parquet", "hf_pirocheto_test.parquet")]
    df = pd.concat(parts)
    return _frame(df["url"], (df["status"] == "phishing").astype(int), "hf_pirocheto")


def load_hf_ealvaradob():
    with open(RAW_DIR / "hf_ealvaradob_urls.json") as f:
        df = pd.DataFrame(json.load(f))
    return _frame(df["text"], df["label"].astype(int), "hf_ealvaradob")


def load_hf_kmack():
    df = pd.concat([pd.read_parquet(RAW_DIR / f"hf_kmack_{s}.parquet") for s in ("train", "valid", "test")])
    return _frame(df["text"], df["label"].astype(int), "hf_kmack")


def load_phishing_database(cap):
    return _frame(_sample(_read_lines(RAW_DIR / "phishing_database_active.txt"), cap), 1, "phishing_database")


def load_phishing_database_domains(cap):
    return _frame(_sample(_read_lines(RAW_DIR / "phishing_database_domains_active.txt"), cap), 1,
                  "phishing_database_domains")


def load_phishing_army():
    return _frame(_read_lines(RAW_DIR / "phishing_army_extended.txt"), 1, "phishing_army")


def load_scam_domains():
    return _frame(_read_lines(RAW_DIR / "scam_domains_shadowwhisperer.txt"), 1, "scam_domains")


def load_openphish():
    return _frame(_read_lines(RAW_DIR / "openphish_feed.txt"), 1, "openphish")


def load_phishunt():
    return _frame(_read_lines(RAW_DIR / "phishunt_feed.txt"), 1, "phishunt")


def _tranco(nrows=None):
    return pd.read_csv(TRANCO_CSV, header=None, names=["rank", "domain"], nrows=nrows)


def _majestic():
    return set(pd.read_csv(RAW_DIR / "majestic_million.csv", usecols=["Domain"])["Domain"].str.lower())


# Tranco alone now ranks many spam/DNS-noise domains (random .xyz/.top names,
# gambling sites), so a domain is only used as legitimate when Majestic's
# backlink-based list also contains it.
def load_tranco(top_n):
    tranco = _tranco(top_n)
    return _frame(tranco[tranco["domain"].isin(_majestic())]["domain"], 0, "tranco")


def load_tranco_majestic(skip_top):
    tranco = _tranco().iloc[skip_top:]
    return _frame(tranco[tranco["domain"].isin(_majestic())]["domain"], 0, "tranco_majestic")


LOADERS = {
    "kaggle_phishing_site_urls": lambda a: load_kaggle_phishing_site_urls(),
    "kaggle_malicious_urls": lambda a: load_kaggle_malicious_urls(),
    "kaggle_harisudhan_urls": lambda a: load_kaggle_harisudhan(),
    "uci_phiusiil": lambda a: load_uci_phiusiil(),
    "hf_pirocheto": lambda a: load_hf_pirocheto(),
    "hf_ealvaradob": lambda a: load_hf_ealvaradob(),
    "hf_kmack": lambda a: load_hf_kmack(),
    "phishing_database": lambda a: load_phishing_database(a.phishdb_cap),
    "phishing_database_domains": lambda a: load_phishing_database_domains(a.phishdb_domains_cap),
    "phishing_army": lambda a: load_phishing_army(),
    "scam_domains": lambda a: load_scam_domains(),
    "tranco": lambda a: load_tranco(a.tranco_top),
    "tranco_majestic": lambda a: load_tranco_majestic(a.tranco_top),
    "openphish": lambda a: load_openphish(),
    "phishunt": lambda a: load_phishunt(),
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["url"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    ok = df["url"].str.len().between(4, 2048) & df["url"].str.contains(".", regex=False)
    # Drop rows mangled by encoding errors (several Kaggle rows are binary garbage).
    ok &= df["url"].map(lambda s: s.isprintable() and sum(ord(c) > 127 for c in s) <= 0.1 * len(s))
    ok &= ~df["url"].str.contains(r"\s", regex=True)
    return df[ok]


# sid321axn/malicious-urls-dataset marks ~48k phishing URLs (copied from other
# corpora) as benign, and kmack/Phishing_urls labels pages like tools.ietf.org as
# phishing, so when only these sources dissent they are overruled.
LOW_TRUST_SOURCES = {"kaggle_malicious_urls", "hf_kmack"}


def _resolve_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    n_labels = df.groupby("norm")["label"].nunique()
    conflicting = df[df["norm"].isin(n_labels[n_labels > 1].index)]
    trusted = conflicting[~conflicting["source"].isin(LOW_TRUST_SOURCES)]
    trusted_labels = trusted.groupby("norm")["label"].agg(["nunique", "first"])
    resolved = trusted_labels[trusted_labels["nunique"] == 1]["first"]

    keep = df[~df["norm"].isin(conflicting["norm"])]
    fixed = conflicting[conflicting["norm"].isin(resolved.index)].copy()
    fixed["label"] = fixed["norm"].map(resolved)
    n_conf = conflicting["norm"].nunique()
    print(f"\nconflicting labels: {n_conf:,d} unique URLs; "
          f"resolved via trusted sources {len(resolved):,d}, dropped {n_conf - len(resolved):,d}")
    return pd.concat([keep, fixed], ignore_index=True)


def _brand_domains(brands: list) -> dict:
    """Real registered domains for each curated brand (Tranco matches, e.g. shopee.sg,
    shopee.co.id; falls back to <brand>.com)."""
    curated = {b.replace("-", "") for b in CURATED_BRANDS}
    found = {}
    for domain in _tranco(200_000)["domain"]:
        label = registered_label(domain)
        if label in curated and len(found.setdefault(label, [])) < 8:
            found[label].append(domain)
    return {b: found.get(b) or [f"{b}.com"] for b in brands if b in curated}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phishdb-cap", type=int, default=400_000, help="max URLs sampled from Phishing.Database links")
    p.add_argument("--phishdb-domains-cap", type=int, default=150_000, help="max domains from Phishing.Database domains")
    p.add_argument("--tranco-top", type=int, default=150_000, help="number of top Tranco domains used as legitimate")
    p.add_argument("--synthetic-pairs", type=int, default=40_000,
                   help="synthetic brand-impersonation phishing URLs (each paired with a legit one); 0 disables")
    args = p.parse_args()

    brands = build_brand_list(TRANCO_CSV)
    write_brand_list(brands)
    print(f"brand vocabulary: {len(brands):,d} names\n")

    frames = []
    for name, loader in LOADERS.items():
        try:
            df = _clean(loader(args))
            print(f"{name:28s} {len(df):>9,d} rows  phishing={df['label'].mean():.2%}")
            frames.append(df)
        except FileNotFoundError as e:
            print(f"{name:28s} skipped ({e.filename} missing; run phishdetect.download)")
    df = pd.concat(frames, ignore_index=True)

    df["norm"] = df["url"].map(normalize_url)
    df = df[df["norm"].str.len() >= 4]

    df = _resolve_conflicts(df)

    # Live feeds are held out: remove their URLs from every other source first.
    is_live = df["source"].isin(LIVE_SOURCES)
    live = df[is_live].drop_duplicates("norm")
    rest = df[~is_live & ~df["norm"].isin(live["norm"])].drop_duplicates("norm")

    rest = rest.assign(reg_domain=rest["norm"].map(registered_domain))
    live = live.assign(reg_domain=live["norm"].map(registered_domain), split="live")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, hold_idx = next(gss.split(rest, groups=rest["reg_domain"]))
    hold = rest.iloc[hold_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(gss2.split(hold, groups=hold["reg_domain"]))

    split = np.empty(len(rest), dtype=object)
    split[train_idx] = "train"
    split[hold_idx[val_idx]] = "val"
    split[hold_idx[test_idx]] = "test"
    rest = rest.assign(split=split)

    parts = [rest, live]
    if args.synthetic_pairs:
        syn = _clean(generate_synthetic(_brand_domains(brands), args.synthetic_pairs, SEED, SYNTHETIC_EXCLUDE))
        syn["norm"] = syn["url"].map(normalize_url)
        syn = syn.drop_duplicates("norm")
        syn = syn[~syn["norm"].isin(rest["norm"]) & ~syn["norm"].isin(live["norm"])]
        syn = syn.assign(reg_domain=syn["norm"].map(registered_domain), split="train")
        # Rows on held-out sites stay out of evaluation but are used by the final refit.
        held_out_domains = set(rest.loc[rest["split"] != "train", "reg_domain"]) | set(live["reg_domain"])
        syn.loc[syn["reg_domain"].isin(held_out_domains), "split"] = "refit_only"
        parts.append(syn)

    out = pd.concat(parts, ignore_index=True)[["url", "norm", "label", "source", "reg_domain", "split"]]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED_DIR / "urls.parquet", index=False)

    print(f"\nfinal dataset: {len(out):,d} unique URLs")
    print(out.groupby("split")["label"].agg(rows="size", phishing_rate="mean").to_string())
    print()
    print(out.groupby("source")["label"].agg(rows="size", phishing_rate="mean").to_string())


if __name__ == "__main__":
    main()
