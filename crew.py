import json
import re
from collections import defaultdict, deque

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

from config import COMPANY_NAME, OLLAMA_MODEL, OLLAMA_BASE_URL
from collect import load_data


# Real rivals per company, so competitor detection isn't hardcoded to one sector.
COMPETITOR_HINTS = {
    "Tesla":     ["waymo", "rivian", "lucid", "byd", "ford", "gm", "volkswagen", "mercedes", "porsche", "toyota", "nio"],
    "NVIDIA":    ["amd", "intel", "broadcom", "qualcomm", "tpu", "cerebras", "groq", "huawei"],
    "SAP":       ["oracle", "salesforce", "workday", "microsoft", "servicenow"],
    "BMW":       ["mercedes", "audi", "volkswagen", "tesla", "porsche", "toyota"],
    "Airbus":    ["boeing", "embraer", "lockheed", "comac"],
    "Siemens":   ["abb", "schneider", "ge", "honeywell", "rockwell"],
    "Lufthansa": ["air france", "klm", "ryanair", "iag", "easyjet", "united", "delta"],
    "DHL":       ["fedex", "ups", "kuehne", "maersk", "amazon logistics"],
    "ASML":      ["applied materials", "lam research", "tokyo electron", "canon", "nikon"],
    "Shopify":   ["amazon", "etsy", "woocommerce", "bigcommerce", "squarespace", "wix"],
}
GENERIC_COMPETITION = ["competitor", "rival", "compete", "market share", "challenger", "versus"]


def competitor_keywords(company):
    return COMPETITOR_HINTS.get(company, []) + GENERIC_COMPETITION


# Toggle the search tool, and pick which agents get it. Keep it on one agent until
# you've confirmed the model actually calls it in the verbose logs.
ENABLE_TOOLS = True
TOOL_AGENTS = {"market"}


def make_news_search_tool(articles, company):
    """Semantic search tool (RAG). The agent queries the ChromaDB vector store by
    meaning; if the store is unavailable it falls back to simple keyword matching."""
    try:
        from crewai.tools import BaseTool
    except ImportError:
        from crewai_tools import BaseTool

    def keyword_fallback(query):
        words = [w for w in re.findall(r"\w+", (query or "").lower()) if len(w) > 2]
        matches = []
        for a in articles:
            text = f"{a.get('title','')} {a.get('summary','')}".lower()
            overlap = sum(1 for w in set(words) if w in text)
            if overlap:
                matches.append((overlap, a))
        matches.sort(key=lambda m: m[0], reverse=True)
        return [a for _, a in matches[:5]]

    class NewsSearchTool(BaseTool):
        name: str = "search_news"
        description: str = (
            "Semantic search over the collected company news. Give a topic, question, or "
            "phrase and it returns the most relevant articles by meaning. Use when you need "
            "more detail or examples than the news already provided in your task."
        )

        def _run(self, query: str) -> str:
            if not query or not query.strip():
                return "Provide a search query."

            hits = []
            try:
                from knowledge import search
                hits = search(query, company=company, k=5)
            except Exception:
                hits = []
            if not hits:                       # store empty/unavailable -> keyword fallback
                hits = keyword_fallback(query)

            if not hits:
                return f"No collected articles found for '{query}'."
            return "\n".join(
                f"- {h.get('title','')} ({h.get('source','')}): {h.get('summary','')[:160]}"
                for h in hits
            )

    return NewsSearchTool()


def run_vader(articles):
    """Deterministic sentiment baseline. The LLM interprets this, it never invents the numbers."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(f"{a.get('title','')} {a.get('summary','')}")["compound"] for a in articles]
    if not scores:
        return {"avg_score": 0.0, "label": "Neutral", "positive": 0, "negative": 0, "neutral": 0, "total": 0}
    avg = round(sum(scores) / len(scores), 4)
    return {
        "avg_score": avg,
        "label":     "Positive" if avg > 0.05 else "Negative" if avg < -0.05 else "Neutral",
        "positive":  sum(1 for s in scores if s > 0.05),
        "negative":  sum(1 for s in scores if s < -0.05),
        "neutral":   sum(1 for s in scores if -0.05 <= s <= 0.05),
        "total":     len(scores),
    }


def diversify(articles, limit=40):
    """Round-robin across sources so one feed doesn't dominate the context window."""
    buckets = defaultdict(deque)
    for a in articles:
        buckets[a.get("source", "")].append(a)

    out = []
    sources = list(buckets.keys())
    while len(out) < limit and any(buckets[s] for s in sources):
        for s in sources:
            if buckets[s]:
                out.append(buckets[s].popleft())
                if len(out) >= limit:
                    break
    return out


def filter_articles(articles, keywords):
    keywords = [k.lower() for k in keywords]
    out = []
    for a in articles:
        text = f"{a.get('title','')} {a.get('summary','')}".lower()
        if any(k in text for k in keywords):
            out.append(a)
    return out


def render_blocks(articles, limit=6, summary_chars=220):
    """Render articles with a short context snippet so agents reason from body text, not just titles."""
    blocks = []
    for a in articles[:limit]:
        title = (a.get("title") or "").strip()
        summary = re.sub(r"\s+", " ", a.get("summary") or "").strip()[:summary_chars]
        block = f"- {title} ({a.get('source', '')})"
        if summary:
            block += f"\n  Context: {summary}"
        blocks.append(block)
    return "\n".join(blocks)


def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    return []


def dedupe_evidence(cards, key="evidence"):
    """Drop evidence already used by an earlier card so two cards don't cite the same line."""
    seen = set()
    for card in cards:
        if not isinstance(card, dict):
            continue
        kept = []
        for ev in card.get(key, []) or []:
            norm = re.sub(r"\s+", " ", str(ev)).strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                kept.append(ev)
        card[key] = kept
    return cards


def dedupe_market(market):
    if not isinstance(market, dict):
        return {}
    seen = set()
    for key in ("competitor_activities", "emerging_technologies", "company_announcements"):
        kept = []
        for item in market.get(key, []) or []:
            text = item if isinstance(item, str) else (item.get("text") if isinstance(item, dict) else str(item))
            norm = re.sub(r"\s+", " ", str(text)).strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                kept.append(item)
        market[key] = kept
    return market


def coerce_sentiment(sentiment):
    """Force the scores to floats so the dashboard's number formatting can't crash."""
    if not isinstance(sentiment, dict):
        return {}
    for key in ("news_score", "public_score"):
        try:
            sentiment[key] = float(sentiment.get(key, 0.0))
        except (TypeError, ValueError):
            sentiment[key] = 0.0
    return sentiment


def best_source(text, articles, min_overlap=3):
    """Find the collected article whose title+summary shares the most meaningful words
    with an evidence sentence. Returns None if the best match is too weak to trust."""
    words = {w for w in re.findall(r"\w+", text.lower()) if len(w) > 3}
    if not words:
        return None
    best, best_score = None, 0
    for a in articles:
        a_words = set(re.findall(r"\w+", f"{a.get('title','')} {a.get('summary','')}".lower()))
        score = len(words & a_words)
        if score > best_score:
            best, best_score = a, score
    return best if best_score >= min_overlap else None


def attach_sources(cards, articles):
    """Turn each plain evidence sentence into {text, title, link, matched} by linking it
    to its most likely source article. Unmatched sentences keep matched=False."""
    for card in cards:
        if not isinstance(card, dict):
            continue
        linked = []
        for ev in card.get("evidence", []) or []:
            text = ev if isinstance(ev, str) else str(ev.get("text", ev))
            match = best_source(text, articles)
            linked.append({
                "text":    text,
                "title":   match.get("title", "") if match else "",
                "link":    match.get("link", "") if match else "",
                "matched": bool(match),
            })
        card["evidence"] = linked
    return cards


def get_llm():
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.4,
        max_tokens=1200,
    )


def build_agents(llm, company, tools=None):
    base = dict(llm=llm, verbose=True, allow_delegation=False)
    paraphrase = "You explain what each signal means in your own words and never copy a headline verbatim."

    def agent_tools(name):
        return (tools or []) if (tools and name in TOOL_AGENTS) else []

    return {
        "market": Agent(
            role="Market Intelligence Analyst",
            goal=f"Analyse the latest news about {company} and produce a market intelligence report.",
            backstory=f"Senior analyst covering {company}. Extracts market signals from news. {paraphrase}",
            tools=agent_tools("market"), **base,
        ),
        "opportunity": Agent(
            role="Opportunity Scout",
            goal=f"Identify 3 distinct strategic opportunities for {company} from the news.",
            backstory=f"Business development expert for {company}. Writes original opportunity insights. {paraphrase}",
            tools=agent_tools("opportunity"), **base,
        ),
        "risk": Agent(
            role="Risk Assessment Officer",
            goal=f"Identify 3 distinct strategic risks for {company} from the news.",
            backstory=f"Risk analyst for {company}. Identifies distinct business threats. {paraphrase}",
            tools=agent_tools("risk"), **base,
        ),
        "sentiment": Agent(
            role="Sentiment Analysis Specialist",
            goal=f"Interpret sentiment around {company} using the provided VADER scores and headlines.",
            backstory=f"Brand analyst for {company}. Turns sentiment data into a short narrative. {paraphrase}",
            tools=agent_tools("sentiment"), **base,
        ),
        "trend": Agent(
            role="Trend Analyst",
            goal=f"Identify 4 distinct emerging trends relevant to {company}.",
            backstory=f"Strategic foresight analyst for {company}. {paraphrase}",
            tools=agent_tools("trend"), **base,
        ),
        "ceo": Agent(
            role="AI CEO Strategic Advisor",
            goal=f"Write 3 strategic recommendations and a CEO briefing for {company}.",
            backstory=f"Senior strategy consultant advising the CEO of {company}. {paraphrase}",
            tools=agent_tools("ceo"), **base,
        ),
    }


def build_tasks(agents, articles, company):
    vader = run_vader(articles)

    # Route each agent only the articles relevant to its job.
    competitor = filter_articles(articles, competitor_keywords(company))
    technology = filter_articles(articles, ["ai", "technology", "electric", "battery", "software",
                                            "innovation", "robot", "digital", "autonomous", "chip"])
    risk = filter_articles(articles, ["decline", "warning", "probe", "recall", "tariff", "loss",
                                      "slump", "cut", "fall", "ban", "lawsuit", "investigation", "delay"])
    opportunity = filter_articles(articles, ["invest", "launch", "deal", "contract", "growth", "expansion",
                                             "partnership", "fund", "record", "order", "approval", "demand"])
    trend = filter_articles(articles, ["market", "trend", "regulation", "demand", "strategy",
                                       "sustainability", "supply", "policy", "shift", "outlook"])

    general = render_blocks(articles, 8)
    competitor_news = render_blocks(competitor, 5) or general
    technology_news = render_blocks(technology, 5) or general
    risk_news = render_blocks(risk, 6) or general
    opportunity_news = render_blocks(opportunity, 6) or general
    trend_news = render_blocks(trend, 6) or general

    evidence_rule = ("Each evidence item is ONE original sentence in your own words that explains "
                     "the signal — do NOT paste a headline. Each card must cover a DIFFERENT topic.")

    return [
        Task(
            description=(
                f"Market intelligence report for {company}. Read the context under each item.\n\n"
                f"COMPETITOR NEWS:\n{competitor_news}\n\n"
                f"TECHNOLOGY NEWS:\n{technology_news}\n\n"
                "recent_news may use the real headlines. All OTHER fields must be your own "
                "analytical sentences (not headline copies), and must not repeat each other.\n"
                "Return ONLY this JSON:\n"
                '{"recent_news":[{"headline":"...","source":"...","date":"..."}],'
                '"competitor_activities":["original sentence about a competitor move"],'
                '"emerging_technologies":["original sentence about a technology shift"],'
                '"company_announcements":["original sentence about a company announcement"]}'
            ),
            expected_output="JSON: recent_news, competitor_activities, emerging_technologies, company_announcements",
            agent=agents["market"],
        ),
        Task(
            description=(
                f"3 strategic opportunities for {company}. Read the context under each item.\n\n"
                f"OPPORTUNITY NEWS:\n{opportunity_news}\n\n"
                f"{evidence_rule} Title = a strategic insight, not a headline.\n"
                "Return ONLY a JSON array (exactly 3 items):\n"
                '[{"title":"strategic insight","impact":"High|Medium|Low",'
                '"evidence":["original explanatory sentence","original explanatory sentence"],"confidence":80}]'
            ),
            expected_output="JSON array: 3 opportunities",
            agent=agents["opportunity"],
        ),
        Task(
            description=(
                f"3 strategic risks for {company}. Read the context under each item.\n\n"
                f"RISK NEWS:\n{risk_news}\n\n"
                f"{evidence_rule} Each risk must have a DIFFERENT category, and evidence must not "
                "be reused between risks. Title = a specific risk description, not a headline.\n"
                "Return ONLY a JSON array (exactly 3 items):\n"
                '[{"title":"risk description","category":"Financial|Competitive|Regulatory|Sentiment|Supply Chain",'
                '"severity":"High|Medium|Low",'
                '"evidence":["original explanatory sentence","original explanatory sentence"],"confidence":75}]'
            ),
            expected_output="JSON array: 3 risks",
            agent=agents["risk"],
        ),
        Task(
            description=(
                f"Sentiment analysis for {company}.\n\n"
                f"VADER RESULTS: avg={vader['avg_score']} ({vader['label']}), "
                f"positive={vader['positive']}, negative={vader['negative']}, neutral={vader['neutral']}\n\n"
                f"HEADLINES:\n{general}\n\n"
                "key_drivers: 3 SHORT original phrases (5-8 words) explaining the mood — not headlines.\n"
                "news_score and public_score must be numbers between -1 and 1.\n"
                "Return ONLY this JSON:\n"
                '{"news_sentiment":"Positive|Neutral|Negative","news_score":0.0,'
                '"public_sentiment":"Positive|Neutral|Negative","public_score":0.0,'
                '"trend":"Improving|Stable|Declining","key_drivers":["phrase 1","phrase 2","phrase 3"]}'
            ),
            expected_output="JSON: sentiment analysis",
            agent=agents["sentiment"],
        ),
        Task(
            description=(
                f"4 emerging trends for {company}. Read the context under each item.\n\n"
                f"TREND NEWS:\n{trend_news}\n\n"
                "Each trend must have a DIFFERENT type. implication = one original sentence on "
                f"the impact for {company}.\n"
                "Return ONLY a JSON array (exactly 4 items):\n"
                '[{"trend":"short name","type":"Technology|Market|Regulatory|Customer Behaviour",'
                '"implication":"one original sentence"}]'
            ),
            expected_output="JSON array: 4 trends",
            agent=agents["trend"],
        ),
        Task(
            description=(
                f"Strategic brief for the CEO of {company}.\n\n"
                f"KEY NEWS:\n{general}\n\n"
                f"Write 3 recommendations, each addressing a DIFFERENT priority. {evidence_rule} "
                "CEO briefing fields must be plain strings (not lists).\n"
                "Return ONLY this JSON:\n"
                '{"recommendations":[{"recommendation":"action sentence","priority":"High|Medium|Low",'
                '"evidence":["original explanatory sentence","original explanatory sentence"],'
                '"expected_impact":"one sentence","risk_level":"High|Medium|Low"}],'
                '"ceo_briefing":{"what_happened":"2 sentences about ' + company + '.",'
                '"why_it_matters":"2 sentences about business impact.",'
                '"what_to_do_next":"2 sentences of specific actions."}}'
            ),
            expected_output="JSON: recommendations and ceo_briefing",
            agent=agents["ceo"],
        ),
    ]


def extract_json(text):
    """Local models often wrap JSON in prose or code fences; pull the JSON out reliably."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    try:
        return json.loads(text)
    except Exception:
        pass
    for pattern in [r"(\[.*\])", r"(\{.*\})"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    return None


def run_analysis(data=None):
    if data is None:
        data = load_data()
    if not data:
        raise RuntimeError("No data found. Run collect first.")

    company = data.get("company", COMPANY_NAME)

    # Cap the working set: more articles means longer prompts, and past ~40 the
    # heaviest agent (CEO) overflows the context window and returns broken JSON.
    articles = diversify(data.get("articles", []), limit=40)
    print(f"Starting analysis: {company} — {len(articles)} articles (diversified)")

    llm = get_llm()
    tools = [make_news_search_tool(articles, company)] if ENABLE_TOOLS else []
    agents = build_agents(llm, company, tools=tools)
    tasks = build_tasks(agents, articles, company)

    Crew(agents=list(agents.values()), tasks=tasks,
         process=Process.sequential, verbose=True).kickoff()

    outputs = [t.output.raw if t.output else "" for t in tasks]

    market = dedupe_market(extract_json(outputs[0]) or {})
    opportunities = dedupe_evidence(as_list(extract_json(outputs[1])))
    risks = dedupe_evidence(as_list(extract_json(outputs[2])))
    sentiment = coerce_sentiment(extract_json(outputs[3]) or {})
    trends = as_list(extract_json(outputs[4]))

    # The CEO task is the one most likely to return the wrong shape; normalise it.
    strategic = extract_json(outputs[5])
    if isinstance(strategic, list):
        strategic = {"recommendations": strategic}
    elif not isinstance(strategic, dict):
        strategic = {}
    strategic["recommendations"] = dedupe_evidence(as_list(strategic.get("recommendations")))
    if not isinstance(strategic.get("ceo_briefing"), dict):
        strategic["ceo_briefing"] = {}

    # Link each evidence sentence to the source article it most likely came from,
    # matching against the full collected corpus (not just the 40 the agents saw).
    corpus = data.get("articles", [])
    opportunities = attach_sources(opportunities, corpus)
    risks = attach_sources(risks, corpus)
    strategic["recommendations"] = attach_sources(strategic["recommendations"], corpus)

    results = {
        "company":         company,
        "industry":        data.get("industry", ""),
        "collected_at":    data.get("collected_at", ""),
        "total_documents": data.get("total_documents", 0),
        "num_sources":     data.get("num_sources", 0),
        "sources":         data.get("sources", []),
        "market":          market,
        "opportunities":   opportunities,
        "risks":           risks,
        "sentiment":       sentiment,
        "trends":          trends,
        "strategic":       strategic,
    }

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Analysis complete. Saved to analysis_results.json")
    return results


if __name__ == "__main__":
    run_analysis()