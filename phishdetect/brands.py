"""Brand names commonly impersonated in phishing.

The list is the curated set below plus second-level labels of the most popular
Tranco domains (dictionary words excluded, since "weather.example.com" is not
impersonation). It is written to resources/brands.txt by build_dataset and
read by the feature extractor, so the training and scoring code see the same list.
"""

import re
from pathlib import Path

import pandas as pd

BRANDS_FILE = Path(__file__).resolve().parent / "resources" / "brands.txt"
DICT_FILE = Path("/usr/share/dict/words")

CURATED_BRANDS = (
    # global tech / email / cloud
    "google", "gmail", "youtube", "apple", "icloud", "itunes", "microsoft", "office365", "outlook",
    "hotmail", "onedrive", "sharepoint", "azure", "xbox", "yahoo", "aol", "zoho",
    "facebook", "instagram", "whatsapp", "messenger", "twitter", "tiktok", "snapchat",
    "linkedin", "telegram", "discord", "wechat", "kakao", "viber", "skype",
    "dropbox", "docusign", "adobe", "wetransfer", "github", "gitlab", "cloudflare", "godaddy",
    "namecheap", "wix", "squarespace", "shopify", "zoom", "webex", "slack", "notion", "canva",
    # streaming / gaming
    "netflix", "spotify", "disney", "disneyplus", "hulu", "primevideo", "hbomax", "steam",
    "steamcommunity", "epicgames", "roblox", "playstation", "nintendo", "battlenet", "blizzard",
    "riotgames", "garena", "minecraft", "twitch",
    # marketplaces / e-commerce
    "amazon", "ebay", "etsy", "walmart", "target", "bestbuy", "costco", "alibaba", "aliexpress",
    "taobao", "temu", "shein", "rakuten", "mercari", "carousell", "shopee", "lazada",
    "tokopedia", "bukalapak", "blibli", "qoo10", "zalora", "flipkart", "myntra", "meesho",
    "snapdeal", "olx", "allegro", "vinted", "leboncoin", "kleinanzeigen", "marktplaats", "gumtree",
    "craigslist", "mercadolibre", "mercadolivre", "mercadopago", "avito", "jumia", "trendyol",
    "hepsiburada", "coupang", "gmarket", "depop", "poshmark", "offerup", "facebookmarketplace",
    # payments / fintech / crypto
    "paypal", "venmo", "cashapp", "zelle", "stripe", "revolut", "payoneer",
    "skrill", "neteller", "alipay", "paytm", "phonepe", "gpay", "googlepay", "applepay", "razorpay",
    "gcash", "paymaya", "grabpay", "touchngo", "shopeepay", "ovo", "gopay",
    "linkaja", "truemoney", "promptpay", "momo", "zalopay", "paylah", "paynow", "klarna", "afterpay",
    "affirm", "sezzle", "coinbase", "binance", "kraken", "bybit", "okx", "kucoin", "bitfinex",
    "metamask", "trustwallet", "phantom", "ledger", "trezor", "blockchain", "uniswap",
    "opensea", "exodus",
    # banks
    "chase", "wellsfargo", "bankofamerica", "citibank", "citi", "usbank", "capitalone", "pnc",
    "truist", "tdbank", "hsbc", "barclays", "lloyds", "natwest", "santander", "halifax", "monzo",
    "starling", "nationwide", "rbc", "scotiabank", "bmo", "cibc", "desjardins", "commbank", "westpac",
    "anz", "nab", "dbs", "posb", "ocbc", "uob", "maybank", "cimb", "publicbank", "rhb", "hongleong",
    "bca", "mandiri", "bni", "bri", "bdo", "bpi", "metrobank", "kasikorn", "kbank", "scb", "krungsri",
    "vietcombank", "techcombank", "sbi", "onlinesbi", "hdfc", "hdfcbank", "icici", "icicibank",
    "axisbank", "kotak", "yesbank", "pnbindia", "itau", "bradesco", "caixa", "nubank", "bancodobrasil",
    "bbva", "caixabank", "sabadell", "rabobank", "abnamro", "deutschebank", "commerzbank",
    "sparkasse", "postbank", "bnpparibas", "creditagricole", "societegenerale", "unicredit",
    "intesasanpaolo", "pkobp", "mbank", "ubs", "credit-suisse", "nordea", "swedbank", "danskebank",
    "mufg", "smbc", "mizuho", "resona", "aeon", "jcb", "amex", "americanexpress", "visa", "mastercard",
    "emirates", "emiratesnbd", "alrajhi", "qnb", "absa", "fnb", "standardbank", "nedbank",
    # delivery / logistics
    "dhl", "fedex", "ups", "usps", "royalmail", "evri", "hermes", "dpd", "gls", "parcelforce",
    "canadapost", "auspost", "australiapost", "nzpost", "singpost", "ninjavan", "jtexpress", "jnt",
    "lalamove", "grab", "gojek", "foodpanda", "deliveroo", "ubereats", "uber", "lyft", "doordash",
    "postnl", "bpost", "laposte", "colissimo", "chronopost", "correos", "ctt", "poste", "posteitaliane",
    "deutschepost", "inpost", "japanpost", "yamato", "sagawa", "kuroneko", "aramex", "bluedart",
    "delhivery", "indiapost", "thailandpost", "posindonesia", "lbc",
    "poslaju", "correios", "sfexpress", "cainiao", "yodel", "maersk",
    # telcos / utilities
    "verizon", "att", "tmobile", "sprint", "xfinity", "comcast", "spectrum", "vodafone", "orange",
    "telstra", "optus", "singtel", "starhub", "celcom",
    "maxis", "telkomsel", "indosat", "airtel", "jio", "docomo", "softbank",
    "rogers", "telus", "movistar", "claro", "etisalat", "stc",
    # travel / services
    "booking", "airbnb", "expedia", "agoda", "tripadvisor", "traveloka", "klook",
    "singaporeair", "airasia", "ryanair", "easyjet", "lufthansa", "qantas",
    "americanairlines", "marriott", "hilton",
    # government / tax
    "irs", "hmrc", "govuk", "ato", "mygov", "medicare", "dvla", "ezpass",
    "sunpass", "fastrak", "toll", "etoll", "singpass", "iras", "lhdn", "incometax", "uidai", "aadhaar",
    "epfo", "receita", "ameli", "impots", "elster", "agenziaentrate", "nhs",
)

_ALPHA = re.compile(r"^[a-z]+$")


def _dictionary_words() -> set:
    if not DICT_FILE.exists():
        return set()
    return {w.strip().lower() for w in DICT_FILE.read_text(errors="ignore").splitlines()}


def build_brand_list(tranco_csv: Path, top_n: int = 20_000) -> list:
    from .features import registered_label

    words = _dictionary_words()
    domains = pd.read_csv(tranco_csv, header=None, names=["rank", "domain"], nrows=top_n)["domain"]
    labels = {registered_label(str(d).strip().lower()) for d in domains}
    popular = {lb for lb in labels if lb and len(lb) >= 5 and _ALPHA.match(lb) and lb not in words}
    curated = {b.replace("-", "") for b in CURATED_BRANDS}
    return sorted(popular | curated)


def write_brand_list(brands: list) -> None:
    BRANDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BRANDS_FILE.write_text("\n".join(brands) + "\n")


def load_brands() -> frozenset:
    if BRANDS_FILE.exists():
        return frozenset(b for b in BRANDS_FILE.read_text().split() if b)
    return frozenset(b.replace("-", "") for b in CURATED_BRANDS)
