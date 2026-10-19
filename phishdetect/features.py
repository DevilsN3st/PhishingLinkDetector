"""URL normalization and lexical feature extraction.

Features are computed from the URL string only (no network lookups), so the
model can score a link before anyone visits it.
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import tldextract
from joblib import Parallel, delayed

from .brands import load_brands

# Offline extractor: uses the public-suffix snapshot bundled with tldextract.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

_BRANDS: frozenset = frozenset()
_BRAND_DELETES: frozenset = frozenset()
_LEET = str.maketrans("013457@$", "oleastas")
_ALNUM_SPLIT = re.compile(r"[^a-z0-9]+")
_ALPHA_SPLIT = re.compile(r"[^a-z]+")


def _deletes(word: str) -> set:
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def set_brands(brands) -> None:
    """Brand vocabulary used by the impersonation features (shared by train and predict)."""
    global _BRANDS, _BRAND_DELETES
    _BRANDS = frozenset(brands)
    _BRAND_DELETES = frozenset(d for b in _BRANDS if len(b) >= 6 for d in _deletes(b))


def get_brands() -> frozenset:
    return _BRANDS


set_brands(load_brands())

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_SPLIT_RE = re.compile(r"^([^/?#]*)(.*)$", re.S)
_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_HEX_IP_RE = re.compile(r"^(0x[0-9a-f]+|\d{8,10})$")
_HEX_RUN_RE = re.compile(r"[0-9a-f]{16,}")
_TOKEN_RE = re.compile(r"[^a-zA-Z0-9]+")

SUSPICIOUS_WORDS = (
    "login", "log-in", "signin", "sign-in", "logon", "verify", "verification",
    "account", "update", "secure", "security", "banking", "confirm", "password",
    "passwd", "webscr", "cmd=", "ebayisapi", "wallet", "unlock", "suspend",
    "billing", "invoice", "recover", "authenticate", "validate", "session",
    "token", "bonus", "gift", "prize", "lucky", "free", "claim", "reward",
    "support", "service", "customer", "alert", "limited", "restore", "seed",
)
BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "hotmail",
    "google", "gmail", "facebook", "instagram", "whatsapp", "amazon", "netflix",
    "ebay", "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "dhl",
    "fedex", "usps", "linkedin", "dropbox", "docusign", "adobe", "yahoo", "aol",
    "steam", "coinbase", "binance", "metamask", "blockchain", "americanexpress",
    "mastercard", "itau", "santander", "bradesco", "hdfc", "icici", "telegram",
    "roblox", "twitter", "spotify", "wetransfer", "onedrive", "sharepoint",
)
SHORTENERS = frozenset((
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc",
    "rb.gy", "s.id", "t.ly", "lnkd.in", "qrco.de", "v.gd", "x.co", "u.to",
))
FREE_HOSTS = (
    "000webhostapp", "weebly", "wixsite", "firebaseapp", "web.app", "github.io",
    "herokuapp", "netlify", "vercel.app", "blogspot", "sites.google", "glitch.me",
    "repl.co", "pages.dev", "workers.dev", "ngrok", "duckdns", "azurewebsites",
    "godaddysites", "square.site", "ipfs", "r2.dev", "webflow.io", "framer",
    "wordpress.com", "jimdo", "yolasite", "mybluehost", "appspot", "translate.goog",
)
RISKY_EXT = (".php", ".exe", ".zip", ".apk", ".scr", ".js", ".bin", ".sh", ".asp", ".cgi")
VOWELS = frozenset("aeiou")


def normalize_url(url: str) -> str:
    """Canonical form shared by all sources: no scheme, no leading www.,
    lowercase host, no trailing slash. Different datasets store URLs with or
    without a scheme, so keeping it would let the model learn the source
    rather than the URL."""
    u = str(url).strip().strip("\"'")
    u = _SCHEME_RE.sub("", u)
    m = _SPLIT_RE.match(u)
    authority, rest = (m.group(1), m.group(2)) if m else (u, "")
    authority = authority.lower().rstrip(".")
    if authority.startswith("www."):
        authority = authority[4:]
    return (authority + rest).rstrip("/")


def split_url(norm: str):
    m = _SPLIT_RE.match(norm)
    authority, rest = (m.group(1), m.group(2)) if m else (norm, "")
    userinfo = ""
    if "@" in authority:
        userinfo, authority = authority.rsplit("@", 1)
    host, port = authority, ""
    if ":" in authority and not authority.startswith("["):
        host, port = authority.rsplit(":", 1)
    path, query = rest, ""
    if "?" in rest:
        path, query = rest.split("?", 1)
    return userinfo, host, port, path, query


def registered_domain(norm: str) -> str:
    _, host, _, _, _ = split_url(norm)
    ext = _EXTRACT(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return host


def registered_label(host: str) -> str:
    """'carousell' for 'help.carousell.com.sg'."""
    return _EXTRACT(host).domain


def _is_brand_typo(label: str) -> bool:
    """One edit (insert/delete/substitute) or digit-for-letter swap away from a brand."""
    if len(label) < 5 or label in _BRANDS:
        return False
    if label.translate(_LEET) in _BRANDS or label in _BRAND_DELETES:
        return True
    dels = _deletes(label)
    if len(label) >= 7 and any(d in _BRANDS for d in dels):
        return True
    return any(d in _BRAND_DELETES for d in dels)


def _brand_embedded(label: str) -> bool:
    """A brand of 6+ chars appears inside a longer domain label, e.g. 'paypalsecure'."""
    n = len(label)
    if n < 7 or label in _BRANDS:
        return False
    return any(label[i:j] in _BRANDS for i in range(n) for j in range(i + 6, min(n, i + 20) + 1)
               if j - i < n)


def _brand_features(domain: str, sub: str, path: str, query: str, free_host: bool) -> dict:
    domain = domain.lower()
    domain_is_brand = domain in _BRANDS
    sub_tokens = [t for t in _ALNUM_SPLIT.split(sub.lower()) if len(t) >= 3]
    dom_tokens = [t for t in _ALPHA_SPLIT.split(domain) if len(t) >= 3 and t != domain]
    path_tokens = {t for t in _ALNUM_SPLIT.split((path + " " + query).lower()) if len(t) >= 4}
    return {
        # The site is the brand itself (e.g. paypal.com), unless it is a free host
        # whose name happens to be popular (firebaseapp, weebly, ...).
        "domain_is_brand": int(domain_is_brand and not free_host),
        "brand_in_subdomain": int(any(t in _BRANDS and t != domain for t in sub_tokens)),
        "brand_token_in_domain": int(not domain_is_brand and any(t in _BRANDS for t in dom_tokens)),
        "brand_embedded_in_domain": int(_brand_embedded(domain)),
        "brand_typo_domain": int(_is_brand_typo(domain) or any(_is_brand_typo(t) for t in dom_tokens)),
        "brand_typo_subdomain": int(any(_is_brand_typo(t) for t in sub_tokens)),
        "brand_in_path": int(any(t in _BRANDS and t != domain for t in path_tokens)),
    }


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _features_one(norm: str) -> dict:
    userinfo, host, port, path, query = split_url(norm)
    ext = _EXTRACT(host)
    domain, suffix, sub = ext.domain, ext.suffix, ext.subdomain
    reg = f"{domain}.{suffix}" if domain and suffix else host
    is_ip = bool(_IPV4_RE.match(host) or _HEX_IP_RE.match(host))
    lower = norm.lower()
    path_l = path.lower()
    tokens = [t for t in _TOKEN_RE.split(lower) if t]
    letters = sum(ch.isalpha() for ch in norm)
    digits = sum(ch.isdigit() for ch in norm)
    dom_letters = [ch for ch in domain if ch.isalpha()]
    non_reg = (sub + " " + path_l + " " + query.lower())

    brand_in_reg = any(b in domain for b in BRANDS)
    brand_elsewhere = any(b in non_reg for b in BRANDS)
    free_host = any(f in host for f in FREE_HOSTS)

    return {
        **_brand_features(domain, sub, path, query, free_host),
        "url_len": len(norm),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "domain_len": len(domain),
        "subdomain_len": len(sub),
        "n_subdomains": len(sub.split(".")) if sub else 0,
        "n_dots": norm.count("."),
        "n_host_dots": host.count("."),
        "n_hyphens": norm.count("-"),
        "n_host_hyphens": host.count("-"),
        "n_underscores": norm.count("_"),
        "n_slashes": norm.count("/"),
        "n_qmarks": norm.count("?"),
        "n_equals": norm.count("="),
        "n_amps": norm.count("&"),
        "n_at": norm.count("@"),
        "n_percent": norm.count("%"),
        "n_tilde": norm.count("~"),
        "n_digits": digits,
        "n_host_digits": sum(ch.isdigit() for ch in host),
        "digit_ratio": digits / max(len(norm), 1),
        "letter_ratio": letters / max(len(norm), 1),
        "upper_ratio": sum(ch.isupper() for ch in norm) / max(len(norm), 1),
        "non_ascii": sum(ord(ch) > 127 for ch in norm),
        "path_depth": path.count("/"),
        "n_params": query.count("&") + 1 if query else 0,
        "has_userinfo": int(bool(userinfo)),
        "has_port": int(bool(port)),
        "is_ip": int(is_ip),
        "is_punycode": int("xn--" in host),
        "is_shortener": int(reg in SHORTENERS or host in SHORTENERS),
        "is_free_host": int(free_host),
        "double_slash_in_path": int("//" in path),
        "has_http_in_path": int("http" in path_l or "http" in query.lower()),
        "has_www_in_path": int("www" in path_l),
        "risky_ext": int(path_l.endswith(RISKY_EXT) or any(e + "?" in path_l for e in RISKY_EXT)),
        "has_hex_run": int(bool(_HEX_RUN_RE.search(lower))),
        "n_suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "brand_in_domain": int(brand_in_reg),
        "brand_elsewhere": int(brand_elsewhere and not brand_in_reg),
        "tld_in_subdomain": int(any(t in sub.split(".") for t in ("com", "net", "org"))),
        "host_entropy": _entropy(host),
        "url_entropy": _entropy(norm),
        "domain_entropy": _entropy(domain),
        "domain_vowel_ratio": (sum(ch in VOWELS for ch in dom_letters) / len(dom_letters)) if dom_letters else 0.0,
        "domain_digit_ratio": sum(ch.isdigit() for ch in domain) / max(len(domain), 1),
        "n_tokens": len(tokens),
        "longest_token": max((len(t) for t in tokens), default=0),
        "avg_token_len": (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0,
        "tld": suffix or ("ip" if is_ip else ""),
    }


def _features_chunk(urls, brands=None) -> list:
    if brands is not None and brands != _BRANDS:
        set_brands(brands)
    return [_features_one(u) for u in urls]


def extract_features(norm_urls, n_jobs: int = -1, chunk: int = 20000) -> pd.DataFrame:
    """Lexical features for already-normalized URLs."""
    norm_urls = list(norm_urls)
    if len(norm_urls) <= chunk:
        rows = _features_chunk(norm_urls)
    else:
        parts = Parallel(n_jobs=n_jobs)(
            delayed(_features_chunk)(norm_urls[i:i + chunk], _BRANDS)
            for i in range(0, len(norm_urls), chunk)
        )
        rows = [r for p in parts for r in p]
    return pd.DataFrame(rows)


def encode_tld(tld: pd.Series, vocab: dict) -> np.ndarray:
    """Map TLD strings to integer codes; unseen TLDs share code 0."""
    return tld.map(vocab).fillna(0).astype(np.int32).to_numpy()
