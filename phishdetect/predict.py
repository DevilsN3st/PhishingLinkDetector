"""Score URLs with the trained phishing detector.

Usage:
  python -m phishdetect.predict https://example.com "paypal.com.secure-login.xyz/verify"
  python -m phishdetect.predict --file urls.txt [--json]
"""

import argparse
import json
import sys
from pathlib import Path

from .model import DEFAULT_MODEL_PATH, PhishingDetector


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("urls", nargs="*", help="URLs to score")
    p.add_argument("--file", type=Path, help="text file with one URL per line")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--threshold", type=float, help="override the decision threshold tuned on validation")
    p.add_argument("--json", action="store_true", help="emit JSON lines")
    args = p.parse_args()

    urls = list(args.urls)
    if args.file:
        urls += [ln.strip() for ln in args.file.read_text().splitlines() if ln.strip()]
    if not urls:
        urls = [ln.strip() for ln in sys.stdin if ln.strip()]
    if not urls:
        p.error("no URLs given")

    detector = PhishingDetector.load(args.model)
    threshold = args.threshold if args.threshold is not None else detector.threshold
    for url, score in zip(urls, detector.predict_proba(urls)):
        verdict = "PHISHING" if score >= threshold else "legit"
        if args.json:
            print(json.dumps({"url": url, "phishing_probability": round(float(score), 4), "verdict": verdict}))
        else:
            print(f"{score:6.3f}  {verdict:8s}  {url}")


if __name__ == "__main__":
    main()
