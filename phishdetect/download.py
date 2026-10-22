"""Download raw phishing / legitimate URL data from public sources.

Usage: python -m phishdetect.download [--force]
"""

import argparse
import os
import shutil
import zipfile

import requests

from . import RAW_DIR

KAGGLE_DATASETS = {
    "kaggle_phishing_site_urls": "taruntiwarihp/phishing-site-urls",
    "kaggle_malicious_urls": "sid321axn/malicious-urls-dataset",
    "kaggle_harisudhan_urls": "harisudhan411/phishing-and-legitimate-urls",
}

_HF = "https://huggingface.co/datasets"
_KMACK = f"{_HF}/kmack/Phishing_urls/resolve/main/data"
HTTP_SOURCES = {
    "uci_phiusiil.zip": "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip",
    "hf_pirocheto_train.parquet": f"{_HF}/pirocheto/phishing-url/resolve/main/data/train.parquet",
    "hf_pirocheto_test.parquet": f"{_HF}/pirocheto/phishing-url/resolve/main/data/test.parquet",
    "hf_ealvaradob_urls.json": f"{_HF}/ealvaradob/phishing-dataset/resolve/main/urls.json",
    "hf_kmack_train.parquet": f"{_KMACK}/train-00000-of-00001-d8afc95a165ea87b.parquet",
    "hf_kmack_valid.parquet": f"{_KMACK}/valid-00000-of-00001-4e1abfe96aa382c2.parquet",
    "hf_kmack_test.parquet": f"{_KMACK}/test-00000-of-00001-4d6cbda5297196e7.parquet",
    "openphish_feed.txt": "https://openphish.com/feed.txt",
    "phishunt_feed.txt": "https://phishunt.io/feed.txt",
    "phishing_database_active.txt": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "phishing_database_domains_active.txt": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt",
    "phishing_army_extended.txt": "https://phishing.army/download/phishing_army_blocklist_extended.txt",
    "scam_domains_shadowwhisperer.txt": "https://raw.githubusercontent.com/ShadowWhisperer/BlockLists/master/Lists/Scam",
    "tranco_top1m.zip": "https://tranco-list.eu/top-1m.csv.zip",
    "majestic_million.csv": "https://downloads.majestic.com/majestic_million.csv",
    "cert_pl_domains.txt": "https://hole.cert.pl/domains/v2/domains.txt",
    "phishdestroy_domains.txt": "https://cdn.jsdelivr.net/gh/phishdestroy/destroylist@main/rootlist/formats/primary/domains.txt",
    "jpxxx_url_benchmark.parquet": f"{_HF}/JPxxx/url-benchmark-dataset/resolve/main/data/train-00000-of-00001.parquet",
}

PHRESHPHISH_REPO = "datasets/phreshphish/phreshphish"
PHISHSTATS_API = "https://api.phishstats.info/api/phishing"
PHISHSTATS_PAGES = 20  # anonymous access allows 50 requests/day
DEEPURLBENCH_URL = f"{_HF}/DeepInstinct/DeepURLBench/resolve/main/urls_without_dns.parquet"
DEEPURLBENCH_SAMPLE = 200_000

HEADERS = {"User-Agent": "Mozilla/5.0 (phishdetect dataset builder)"}


def _http_get(url: str, dest, force: bool) -> None:
    if dest.exists() and not force:
        print(f"  cached  {dest.name}")
        return
    print(f"  fetch   {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def _kaggle_get(name: str, handle: str, force: bool) -> None:
    dest = RAW_DIR / name
    if dest.exists() and any(dest.glob("*.csv")) and not force:
        print(f"  cached  {name}")
        return
    os.environ.setdefault("KAGGLEHUB_CACHE", str(RAW_DIR / ".kagglehub"))
    import kagglehub

    print(f"  kaggle  {handle}")
    path = kagglehub.dataset_download(handle, force_download=force)
    dest.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(path):
        shutil.copy2(os.path.join(path, f), dest / f)


def _phreshphish_get(force: bool) -> None:
    """URL/label/date columns only; the dataset's HTML column is several GB."""
    dest = RAW_DIR / "hf_phreshphish.parquet"
    if dest.exists() and not force:
        print(f"  cached  {dest.name}")
        return
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    print(f"  hf      {PHRESHPHISH_REPO} (url, label, date)")
    fs = HfFileSystem()
    parts = []
    for path in sorted(fs.glob(f"{PHRESHPHISH_REPO}/data/*.parquet")):
        with fs.open(path) as f:
            df = pq.ParquetFile(f).read(columns=["url", "label", "date"]).to_pandas()
        parts.append(df.assign(hf_split=path.rsplit("/", 1)[-1].split("-")[0]))
    pd.concat(parts, ignore_index=True).to_parquet(dest, index=False)


def _deepurlbench_get(force: bool) -> None:
    """Download DeepURLBench and keep a 200k phishing/benign sample for external eval.

    The full file is ~2 GB (CC-BY-NC). We never train on it; build_dataset treats
    the sample as ext_deepurlbench.
    """
    dest = RAW_DIR / "hf_deepurlbench_sample.parquet"
    raw = RAW_DIR / "hf_deepurlbench.parquet"
    if dest.exists() and not force:
        print(f"  cached  {dest.name}")
        return
    _http_get(DEEPURLBENCH_URL, raw, force)
    import pandas as pd
    import pyarrow.parquet as pq

    print(f"  sample  DeepURLBench -> {DEEPURLBENCH_SAMPLE:,d} phishing/benign rows (latest row groups)")
    cap = DEEPURLBENCH_SAMPLE // 2
    counts = {"phishing": 0, "benign": 0}
    frames = []
    pf = pq.ParquetFile(raw)
    # File is chronological; later row groups are the temporal hold-out we want.
    for i in range(pf.num_row_groups - 1, -1, -1):
        df = pf.read_row_group(i, columns=["url", "label", "first_seen"]).to_pandas()
        df = df[df["label"].isin(counts)]
        frames.append(df)
        for label, n in df["label"].value_counts().items():
            counts[label] = counts.get(label, 0) + int(n)
        if all(counts[k] >= cap * 2 for k in counts):
            break
    all_rows = pd.concat(frames, ignore_index=True).drop_duplicates("url")
    all_rows = all_rows[all_rows["label"].isin(("phishing", "benign"))].copy()
    all_rows["label"] = all_rows["label"].astype(str)
    parts = [all_rows[all_rows["label"] == label].sample(min(cap, (all_rows["label"] == label).sum()),
                                                         random_state=42)
             for label in counts]
    pd.concat(parts, ignore_index=True).to_parquet(dest, index=False)
    raw.unlink(missing_ok=True)


def _phishstats_get(force: bool) -> None:
    dest = RAW_DIR / "phishstats_feed.txt"
    if dest.exists() and not force:
        print(f"  cached  {dest.name}")
        return
    print(f"  fetch   {PHISHSTATS_API} ({PHISHSTATS_PAGES} pages)")
    urls = []
    for page in range(1, PHISHSTATS_PAGES + 1):
        r = requests.get(PHISHSTATS_API, params={"_sort": "-id", "_size": 100, "_p": page, "_fields": "url"},
                         headers=HEADERS, timeout=60)
        if r.status_code == 429 and urls:
            break
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        urls += [row["url"] for row in rows if row.get("url")]
    dest.write_text("\n".join(urls) + "\n")


def _unzip(zip_path, dest_dir) -> None:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    for name, handle in KAGGLE_DATASETS.items():
        try:
            _kaggle_get(name, handle, args.force)
        except Exception as e:
            failures.append((handle, e))

    for fname, url in HTTP_SOURCES.items():
        try:
            _http_get(url, RAW_DIR / fname, args.force)
        except Exception as e:
            failures.append((url, e))

    for name, fetch in (("phreshphish", _phreshphish_get), ("phishstats", _phishstats_get),
                        ("deepurlbench", _deepurlbench_get)):
        try:
            fetch(args.force)
        except Exception as e:
            failures.append((name, e))

    for zname in ("uci_phiusiil", "tranco_top1m"):
        zp = RAW_DIR / f"{zname}.zip"
        if zp.exists():
            _unzip(zp, RAW_DIR / zname)

    if failures:
        print("\nSome sources failed (dataset will be built from the rest):")
        for src, e in failures:
            print(f"  {src}: {e}")


if __name__ == "__main__":
    main()
