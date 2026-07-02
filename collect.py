import json
import time
import hashlib
import re
import feedparser
import requests
from datetime import datetime


def _article_id(title, link):
    return hashlib.md5((title + link).encode()).hexdigest()


def _clean(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_feed(name, url, max_items=50):
    articles = []
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        feed = feedparser.parse(resp.content)
        for entry in feed.entries[:max_items]:
            title   = _clean(getattr(entry, "title",   ""))
            summary = _clean(getattr(entry, "summary", ""))
            link    = getattr(entry, "link",      "")
            pub     = getattr(entry, "published",  datetime.now().isoformat())
            if not title:
                continue
            articles.append({
                "id":        _article_id(title, link),
                "source":    name,
                "title":     title,
                "summary":   summary,
                "link":      link,
                "published": pub,
                "collected": datetime.now().isoformat(),
            })
        print(f"  {name}: {len(articles)} articles")
    except Exception as e:
        print(f"  {name}: failed — {e}")
    return articles


def collect_all(company_name=None, ticker=None, industry=None):
    from config import COMPANIES, build_sources, DATA_FILE

    if company_name is None:
        from config import COMPANY_NAME, RSS_SOURCES, EXTRA_SOURCES
        company_name  = COMPANY_NAME
        rss_sources   = RSS_SOURCES
        extra_sources = EXTRA_SOURCES
        industry      = industry or "Unknown"
    else:
        meta          = COMPANIES.get(company_name, {})
        ticker        = ticker   or meta.get("ticker", company_name[:4].upper())
        industry      = industry or meta.get("industry", "Unknown")
        rss_sources, extra_sources = build_sources(company_name, ticker)

    print(f"Collecting data for: {company_name}")
    raw, sources_used = [], []

    for name, url in {**rss_sources, **extra_sources}.items():
        arts = _parse_feed(name, url)
        raw.extend(arts)
        if arts:
            sources_used.append(name)
        time.sleep(0.5)

    seen, unique = set(), []
    for a in raw:
        if a["id"] not in seen:
            seen.add(a["id"])
            unique.append(a)

    data = {
        "company":         company_name,
        "industry":        industry,
        "collected_at":    datetime.now().isoformat(),
        "total_documents": len(unique),
        "num_sources":     len(sources_used),
        "sources":         sources_used,
        "articles":        unique,
    }

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Collected {len(unique)} articles from {len(sources_used)} sources")

    # Index into the vector store for semantic retrieval. Wrapped so a Chroma/model
    # hiccup never blocks collection — the JSON file is still written above.
    try:
        from knowledge import index_articles
        index_articles(unique, company_name)
    except Exception as e:
        print(f"  Chroma indexing skipped: {e}")

    return data


def load_data():
    from config import DATA_FILE
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("No data file found. Run collect_all() first.")
        return {}


def build_context(data, max_chars=3000):
    lines, total = [], 0
    for i, a in enumerate(data.get("articles", []), 1):
        block = (
            f"[{i}] SOURCE: {a['source']}\n"
            f"TITLE: {a['title']}\n"
            f"DATE: {a['published'][:10]}\n"
            f"SUMMARY: {a['summary'][:200]}\n---"
        )
        if total + len(block) > max_chars:
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines)


if __name__ == "__main__":
    collect_all()
