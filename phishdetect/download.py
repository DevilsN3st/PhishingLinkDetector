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
}

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
