OLLAMA_MODEL    = "llama3.1:8b"
# OLLAMA_MODEL = "qwen2.5:3b"
# OLLAMA_MODEL = "gemma2:2b"
# OLLAMA_MODEL = "mistral:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

DATA_FILE         = "collected_data.json"
MAX_CONTEXT_CHARS = 3000

COMPANIES = {
    "Tesla":    {"industry": "Electric Vehicles & Clean Energy", "ticker": "TSLA"},
    "NVIDIA":   {"industry": "Semiconductors & AI Hardware",     "ticker": "NVDA"},
    "SAP":      {"industry": "Enterprise Software",              "ticker": "SAP"},
    "BMW":      {"industry": "Automotive",                       "ticker": "BMWYY"},
    "Airbus":   {"industry": "Aerospace & Defence",              "ticker": "EADSY"},
    "Siemens":  {"industry": "Industrial Automation",            "ticker": "SIEGY"},
    "Lufthansa":{"industry": "Aviation & Travel",                "ticker": "DLAKY"},
    "DHL":      {"industry": "Logistics & Supply Chain",         "ticker": "DHLGY"},
    "ASML":     {"industry": "Semiconductor Equipment",          "ticker": "ASML"},
    "Shopify":  {"industry": "E-Commerce Technology",            "ticker": "SHOP"},
}


def build_sources(company_name: str, ticker: str) -> tuple[dict, dict]:
    q = company_name.replace(" ", "+")
    rss_sources = {
        "Google News": (
            f"https://news.google.com/rss/search?q={q}"
            f"+stock+OR+{q}+business&hl=en-US&gl=US&ceid=US:en"
        ),
        "Yahoo Finance": (
            f"https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={ticker}&region=US&lang=en-US"
        ),
        "Reddit": (
            f"https://www.reddit.com/r/investing+stocks+wallstreetbets"
            f"/search.rss?q={q}&sort=new&limit=50"
        ),
    }
    extra_sources = {
        "Google News - Industry": (
            f"https://news.google.com/rss/search"
            f"?q={q}+industry+market+2024&hl=en-US&gl=US&ceid=US:en"
        ),
        "Google News - Competitors": (
            f"https://news.google.com/rss/search"
            f"?q={q}+competitor+OR+rival&hl=en-US&gl=US&ceid=US:en"
        ),
    }
    return rss_sources, extra_sources


# Defaults — overwritten at runtime when user selects a company
COMPANY_NAME = "Tesla"
INDUSTRY     = COMPANIES["Tesla"]["industry"]
RSS_SOURCES, EXTRA_SOURCES = build_sources("Tesla", COMPANIES["Tesla"]["ticker"])