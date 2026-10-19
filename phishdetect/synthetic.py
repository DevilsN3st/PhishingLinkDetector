"""Synthetic brand-impersonation URLs (and matched legitimate brand URLs).

Real feeds contain few examples of lures such as
"carousell.order-distribution.com/?I-sEwkdawKvin" for any one brand. These
templates teach the pattern "brand name on a domain the brand does not own".
Each phishing sample is paired with a legitimate URL on the brand's real domain
that uses the same vocabulary (login, order, random query strings), so the
model cannot shortcut on those words. Used for training only.
"""

import string

import numpy as np
import pandas as pd

LURE_WORDS = (
    "order", "orders", "distribution", "delivery", "deliver", "payment", "pay", "payout", "checkout",
    "secure", "verify", "verification", "support", "service", "services", "help", "helpdesk", "claim",
    "refund", "receive", "receipt", "buyer", "seller", "sellers", "transfer", "wallet", "account",
    "confirm", "center", "centre", "portal", "express", "logistics", "shipping", "ship", "parcel",
    "track", "tracking", "invoice", "billing", "update", "auth", "member", "members", "reward",
    "rewards", "bonus", "promo", "deal", "deals", "sale", "shop", "store", "mall", "market", "trade",
    "escrow", "safe", "safety", "protect", "guard", "notice", "alert", "info", "online", "global",
    "asia", "intl", "official", "client", "customer", "care", "cs", "id", "login", "signin", "web",
    "app", "pro", "hub", "link", "go", "get", "my", "now", "fast", "quick", "cash", "fund", "funds",
    "release", "settle", "settlement", "process", "processing", "approve", "approval", "status",
)
SCAM_TLDS = (
    "com", "com", "com", "net", "org", "top", "xyz", "shop", "online", "site", "info", "cc", "vip",
    "icu", "live", "buzz", "cfd", "sbs", "pw", "store", "click", "link", "help", "support", "app",
    "co", "me", "life", "work", "club", "fun", "space", "website", "tech", "asia", "biz", "us",
    "cyou", "bond", "rest", "monster", "best", "lat", "mobi", "ink", "world", "today", "net.cn",
)
FREE_HOSTS = (
    "web.app", "firebaseapp.com", "netlify.app", "vercel.app", "pages.dev", "workers.dev",
    "weebly.com", "wixsite.com", "github.io", "herokuapp.com", "glitch.me", "000webhostapp.com",
    "godaddysites.com", "square.site", "blogspot.com", "webflow.io", "repl.co", "framer.website",
)
COUNTRY_CODES = ("sg", "my", "id", "ph", "th", "vn", "hk", "tw", "in", "au", "uk", "us", "br", "mx")
LEGIT_SUBS = (
    "www", "help", "support", "m", "shop", "order", "orders", "account", "accounts", "pay", "secure",
    "login", "signin", "auth", "id", "mail", "my", "app", "api", "docs", "blog", "careers", "seller",
    "sellers", "checkout", "business", "developer", "community", "news", "store", "cs", "static",
)
PHISH_PATHS = (
    "", "", "login", "verify", "account/verify", "order/{rand}", "get/{RAND}", "personal/{digits}",
    "checkout", "payment/confirm", "secure/update", "{rand}", "delivery/{rand}", "item/{digits}",
    "receive", "claim", "index.html", "index.php", "signin", "{word}/{rand}", "{rand}/{rand}",
    "p/{digits}", "order-confirm", "seller/receive-payment", "transfer/{digits}",
)
LEGIT_PATHS = (
    "", "", "help", "help/article/{digits}", "p/{word}-{word}-{digits}", "account", "login",
    "checkout", "orders", "search", "products/{word}-{word}", "en/{word}", "about", "careers",
    "support/contact", "cart", "u/{rand}", "item/{digits}", "{word}/{word}", "categories/{word}",
    "promotions/{word}", "seller/dashboard", "order/{digits}", "track/{RAND}", "signin", "sign-in",
    "myaccount", "myaccount/{word}", "account/login", "accounts/login", "user/profile",
    "user/account/{word}", "customer/account", "inbox", "settings", "dashboard", "auth/login",
    "personal/{word}", "{word}/login", "mail", "wallet", "payments", "security", "verify-email",
)
QUERIES = (
    "", "", "", "?{rand}", "?{Rand}", "?id={digits}", "?ref={word}", "?token={rand}",
    "?utm_source={word}&utm_campaign={word}", "?q={word}", "?lang=en", "?s={rand}",
)
PHISH_QUERIES = QUERIES + ("?ref=sms&bank={word}", "?refer=mail", "?order={digits}&confirm=1")

_LOWER = string.ascii_lowercase
_MIXED = string.ascii_letters + string.digits
_VOWELS = "aeiou"
_CONS = "bcdfghjklmnprstvwxz"


class _Gen:
    def __init__(self, rng):
        self.rng = rng

    def choice(self, seq):
        return seq[self.rng.integers(len(seq))]

    def rand(self, lo=5, hi=14, chars=_LOWER + string.digits):
        return "".join(self.choice(chars) for _ in range(self.rng.integers(lo, hi)))

    def pseudo_word(self):
        n = self.rng.integers(2, 5)
        return "".join(self.choice(_CONS) + self.choice(_VOWELS) for _ in range(n)) + (
            self.choice(_CONS) if self.rng.random() < 0.5 else "")

    def fill(self, template):
        out = template
        while "{" in out:
            out = (out.replace("{rand}", self.rand(), 1)
                      .replace("{RAND}", self.rand(8, 16, string.ascii_uppercase + string.digits), 1)
                      .replace("{Rand}", "".join(self.choice(_MIXED + "-") for _ in range(self.rng.integers(8, 16))), 1)
                      .replace("{digits}", str(self.rng.integers(10**4, 10**10)), 1)
                      .replace("{word}", self.choice(LURE_WORDS), 1))
        return out

    def scheme(self):
        return self.choice(("https://", "https://", "http://", ""))

    def lure_label(self):
        w1, w2 = self.choice(LURE_WORDS), self.choice(LURE_WORDS)
        style = self.rng.integers(5)
        if style == 0:
            return f"{w1}-{w2}"
        if style == 1:
            return f"{w1}{w2}"
        if style == 2:
            return f"{w1}{self.rng.integers(1, 999)}"
        if style == 3:
            return self.pseudo_word()
        return f"{w1}-{self.pseudo_word()}"

    def typo(self, brand):
        i = self.rng.integers(len(brand))
        op = self.rng.integers(4)
        if op == 0:
            return brand[:i] + brand[i + 1:]
        if op == 1:
            return brand[:i] + brand[i] + brand[i:]
        if op == 2:
            return brand[:i] + self.choice(_LOWER) + brand[i + 1:]
        return brand.replace("o", "0", 1).replace("l", "1", 1) if ("o" in brand or "l" in brand) else brand + "s"


def _phish_url(g: _Gen, brand: str, brand_domain: str) -> str:
    tld = g.choice(SCAM_TLDS)
    lure = g.lure_label()
    w = g.choice(LURE_WORDS)
    cc = g.choice(COUNTRY_CODES)
    t = g.rng.integers(9)
    if t == 0:
        host = f"{brand}.{lure}.{tld}"
    elif t == 1:
        host = f"{brand}-{g.choice((w, cc))}.{tld}"
    elif t == 2:
        host = f"{g.choice((w, cc))}-{brand}.{tld}"
    elif t == 3:
        host = f"{brand}{g.choice(('-', '--', ''))}{w}.{g.choice(FREE_HOSTS)}"
    elif t == 4:
        host = f"{brand_domain}-{g.rand(4, 10)}.{tld}"
    elif t == 5:
        host = f"{g.choice((w, cc, 'www'))}.{brand}{g.choice(('', '-' + cc, str(g.rng.integers(1, 99))))}.{tld}"
    elif t == 6:
        host = f"{brand}-{cc}.{lure}.{tld}"
    elif t == 7:
        host = f"{g.typo(brand)}.{tld}"
    else:
        return f"{g.scheme()}{lure}.{tld}/{brand}/{g.fill(g.choice(PHISH_PATHS))}{g.fill(g.choice(PHISH_QUERIES))}"
    path = g.fill(g.choice(PHISH_PATHS))
    query = g.fill(g.choice(PHISH_QUERIES))
    return f"{g.scheme()}{host}/{path}{query}"


def _legit_url(g: _Gen, brand_domain: str) -> str:
    host = brand_domain if g.rng.random() < 0.4 else f"{g.choice(LEGIT_SUBS)}.{brand_domain}"
    return f"{g.scheme()}{host}/{g.fill(g.choice(LEGIT_PATHS))}{g.fill(g.choice(QUERIES))}"


def generate(brand_domains: dict, n_pairs: int, seed: int = 42, exclude_domains=()) -> pd.DataFrame:
    """brand_domains maps brand label -> list of the brand's real registered domains."""
    g = _Gen(np.random.default_rng(seed))
    brands = sorted(brand_domains)
    exclude = tuple(exclude_domains)
    phish, legit = [], []
    while len(phish) < n_pairs:
        b = g.choice(brands)
        url = _phish_url(g, b, g.choice(brand_domains[b]))
        if any(d in url for d in exclude):
            continue
        phish.append(url)
        legit.append(_legit_url(g, g.choice(brand_domains[b])))
    return pd.DataFrame({
        "url": phish + legit,
        "label": [1] * len(phish) + [0] * len(legit),
        "source": "synthetic",
    })
