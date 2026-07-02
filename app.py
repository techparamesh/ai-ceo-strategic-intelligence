import json
import os
from email.utils import parsedate_to_datetime
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import COMPANIES

st.set_page_config(page_title="NLP Project Dashboard", layout="wide")


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        return {}


def safe(val, fallback="—"):
    if not val or "|" in str(val) or "..." in str(val):
        return fallback
    return str(val)


def extract_text(item):
    if isinstance(item, dict):
        return item.get("headline", item.get("title", str(item)))
    return str(item)


def render_evidence(evidence):
    """Show each evidence item as the source headline (clickable), with the agent's
    sentence as a caption. Falls back gracefully for older string-only evidence."""
    for n, ev in enumerate(evidence or [], 1):
        if isinstance(ev, dict):
            text  = ev.get("text", "")
            title = ev.get("title", "")
            link  = ev.get("link", "")
            if title and link:
                st.markdown(f"**Evidence {n}:** [{title}]({link})")
                st.caption(text)
            elif title:
                st.markdown(f"**Evidence {n}:** {title}")
                st.caption(text)
            else:
                st.markdown(f"**Evidence {n}:** {text}")
                st.caption("source: collected news")
        else:
            st.markdown(f"**Evidence {n}:** {ev}")


# Sidebar first — need to know company before loading data
with st.sidebar:
    st.markdown("### Strategic Intelligence Agent")
    # st.markdown("---")

    preset_options = list(COMPANIES.keys()) + ["Other"]
    selection = st.selectbox("Select Company", preset_options, index=0)

    if selection == "Other":
        custom_name     = st.text_input("Company Name", placeholder="e.g. Apple")
        custom_ticker   = st.text_input("Stock Ticker (optional)", placeholder="e.g. AAPL")
        chosen_name     = custom_name.strip()
        chosen_ticker   = custom_ticker.strip() or chosen_name[:4].upper()
        chosen_industry = ""
    else:
        chosen_name     = selection
        chosen_ticker   = COMPANIES[selection]["ticker"]
        chosen_industry = COMPANIES[selection]["industry"]

    # st.markdown("---")

    if st.button("Run AgentAnalysis", use_container_width=True, type="primary"):
        if not chosen_name:
            st.warning("Please enter a company name.")
        else:
            with st.spinner(f"Collecting data for {chosen_name}..."):
                try:
                    from collect import collect_all
                    collect_all(company_name=chosen_name, ticker=chosen_ticker, industry=chosen_industry)
                except Exception as e:
                    st.error(f"Collection failed: {e}")
                    st.stop()

            with st.spinner("Running AI agents..."):
                try:
                    from crew import run_analysis
                    from collect import load_data
                    run_analysis(load_data())
                    st.success("Done.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # st.markdown("---")
    # st.caption("Model: llama3.1:8b via Ollama")
    # st.caption("Framework: CrewAI")

# Clear stale data on new session
if "session_started" not in st.session_state:
    for f in ["collected_data.json", "analysis_results.json"]:
        if os.path.exists(f):
            os.remove(f)
    st.session_state["session_started"] = True


# Load data — only show if it matches selected company
collected    = load("collected_data.json")
analysis     = load("analysis_results.json")
data_company = collected.get("company", "")
data_matches = data_company.lower() == chosen_name.lower() if chosen_name else False

company      = chosen_name
industry     = collected.get("industry", chosen_industry) if data_matches else chosen_industry
total_docs   = collected.get("total_documents", 0) if data_matches else 0
num_sources  = collected.get("num_sources",     0) if data_matches else 0
sources      = collected.get("sources",        []) if data_matches else []
collected_at = collected.get("collected_at",   "") if data_matches else ""
articles     = collected.get("articles",       []) if data_matches else []
analysis     = analysis                             if data_matches else {}


st.title("AI CEO — Strategic Intelligence Dashboard")
st.caption(f"Company: {company or '—'}  |  Industry: {industry or '—'}")
st.markdown("---")


# Section 1: Company Overview
st.subheader("Company Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Company",     company or "—")
c2.metric("Documents",   total_docs)
c3.metric("Sources",     num_sources)
c4.metric("Last Update", collected_at[:16].replace("T", " ") if collected_at else "—")
if sources:
    st.caption("Sources: " + " | ".join(sources))
st.markdown("---")


# Section 2: Market Intelligence
st.subheader("Market Intelligence")
market = analysis.get("market", {})
if not isinstance(market, dict):
    market = {}
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Recent News**")
    news = market.get("recent_news", [])
    if news:
        for item in news[:6]:
            st.markdown(f"- {extract_text(item)}")
    elif articles:
        for a in articles[:6]:
            st.markdown(f"- {a['title'][:90]}  *{a['source']}*")
    else:
        st.info("Select a company and click Run Analysis.")

with col2:
    for label, key in [
        ("Competitor Activities", "competitor_activities"),
        ("Emerging Technologies", "emerging_technologies"),
        ("Company Announcements", "company_announcements"),
    ]:
        items = market.get(key, [])
        if items:
            st.markdown(f"**{label}**")
            for item in items[:3]:
                text = extract_text(item)
                if text:
                    st.markdown(f"- {text}")

st.markdown("---")


# Section 3: Opportunity Monitor
st.subheader("Opportunity Monitor")
opportunities = analysis.get("opportunities", [])
if opportunities:
    for opp in opportunities:
        if not isinstance(opp, dict):
            continue
        with st.expander(opp.get("title", "Opportunity"), expanded=True):
            st.markdown(f"**Impact:** {opp.get('impact', 'Medium')}  |  **Confidence:** {opp.get('confidence', 70)}%")
            render_evidence(opp.get("evidence", []))
else:
    st.info("Run Analysis to identify opportunities.")
st.markdown("---")


# Section 4: Risk Monitor
st.subheader("Risk Monitor")
risks = analysis.get("risks", [])
if risks:
    for risk in risks:
        if not isinstance(risk, dict):
            continue
        with st.expander(risk.get("title", "Risk"), expanded=True):
            st.markdown(f"**Category:** {risk.get('category', '')}  |  **Severity:** {risk.get('severity', 'Medium')}  |  **Confidence:** {risk.get('confidence', 75)}%")
            render_evidence(risk.get("evidence", []))
else:
    st.info("Run Analysis to identify risks.")
st.markdown("---")


# Section 5: Sentiment Analysis
st.subheader("Sentiment Analysis")
sentiment = analysis.get("sentiment", {})

if articles:
    sia    = SentimentIntensityAnalyzer()
    scored = [(sia.polarity_scores(f"{a.get('title','')} {a.get('summary','')}")["compound"], a)
              for a in articles]
    scores    = [s for s, _ in scored]
    total     = len(scores)
    vader_avg = round(sum(scores) / total, 3)
    pos = sum(1 for s in scores if s >  0.05)
    neg = sum(1 for s in scores if s < -0.05)
    neu = total - pos - neg
else:
    scored, scores, total, vader_avg, pos, neg, neu = [], [], 0, 0.0, 0, 0, 0

overall = "Positive" if vader_avg > 0.05 else "Negative" if vader_avg < -0.05 else "Neutral"


def _real_trend():
    """Compare mean sentiment of the newest third of articles vs the oldest third,
    using publish dates. Falls back to the model's trend if dates are unusable."""
    dated = []
    for comp, a in scored:
        try:
            dated.append((parsedate_to_datetime(a.get("published", "")).timestamp(), comp))
        except Exception:
            continue
    if len(dated) >= 6:
        dated.sort(key=lambda x: x[0])
        k = max(1, len(dated) // 3)
        diff = (sum(c for _, c in dated[-k:]) / k) - (sum(c for _, c in dated[:k]) / k)
        if diff >  0.05: return "Improving", diff
        if diff < -0.05: return "Declining", diff
        return "Stable", diff
    return safe(sentiment.get("trend"), "Stable"), None


trend_label, trend_diff = _real_trend()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Overall",  overall, f"{vader_avg:+.2f}" if total else "—")
m2.metric("Positive", pos, f"{round(pos/total*100)}%" if total else "—", delta_color="off")
m3.metric("Neutral",  neu, f"{round(neu/total*100)}%" if total else "—", delta_color="off")
m4.metric("Negative", neg, f"{round(neg/total*100)}%" if total else "—", delta_color="off")
m5.metric("Trend",    trend_label, f"{trend_diff:+.2f}" if trend_diff is not None else None)

# Slim proportion strip
if total:
    p, nu, ng = pos / total * 100, neu / total * 100, neg / total * 100
    st.markdown(
        f'<div style="display:flex;height:10px;border-radius:5px;overflow:hidden;margin:10px 0 4px;">'
        f'<div style="width:{p}%;background:#4caf50;"></div>'
        f'<div style="width:{nu}%;background:#607d8b;"></div>'
        f'<div style="width:{ng}%;background:#ef5350;"></div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{pos} positive · {neu} neutral · {neg} negative across {total} articles")

# The meaningful part: what is actually driving sentiment each way
if scored:
    top_pos = [(c, a) for c, a in sorted(scored, key=lambda x: x[0], reverse=True) if c >  0.05][:3]
    top_neg = [(c, a) for c, a in sorted(scored, key=lambda x: x[0])              if c < -0.05][:3]
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**What's lifting sentiment**")
        if top_pos:
            for c, a in top_pos:
                st.markdown(f"- {a['title'][:95]}  `{c:+.2f}`")
        else:
            st.caption("No strongly positive coverage.")
    with cB:
        st.markdown("**What's dragging sentiment**")
        if top_neg:
            for c, a in top_neg:
                st.markdown(f"- {a['title'][:95]}  `{c:+.2f}`")
        else:
            st.caption("No strongly negative coverage.")

# Themes from the model — kept only if they're short phrases, not pasted headlines
drivers = sentiment.get("key_drivers", [])
themes  = []
for d in drivers:
    txt = d if isinstance(d, str) else d.get("driver", "")
    if txt and len(txt.split()) <= 12:
        themes.append(txt)
if themes:
    st.markdown("**Themes:** " + "  ·  ".join(themes))
st.markdown("---")


# Section 6: Strategic Recommendations
st.subheader("Strategic Recommendations")
strategic = analysis.get("strategic", {})
if not isinstance(strategic, dict):
    strategic = {}
recs      = strategic.get("recommendations", [])
trends    = analysis.get("trends", [])

valid_trends = [tr for tr in trends if isinstance(tr, dict) and tr.get("trend")]
if valid_trends:
    st.markdown("**Emerging Trends**")
    cols = st.columns(min(len(valid_trends), 4))
    for i, tr in enumerate(valid_trends[:4]):
        with cols[i]:
            st.markdown(f"**{tr.get('trend', '')}**")
            st.caption(tr.get("type", ""))
            st.write(tr.get("implication", ""))

if recs:
    st.markdown("**Recommendations**")
    for i, rec in enumerate(recs, 1):
        if not isinstance(rec, dict):
            continue
        with st.expander(f"#{i}  {rec.get('recommendation', '')}", expanded=(i <= 2)):
            st.markdown(f"**Priority:** {rec.get('priority', 'Medium')}  |  **Risk Level:** {rec.get('risk_level', 'Medium')}")
            st.markdown(f"**Expected Impact:** {rec.get('expected_impact', '')}")
            render_evidence(rec.get("evidence", []))
else:
    st.info("Run Analysis to generate recommendations.")
st.markdown("---")


# Section 7: CEO Briefing
st.subheader("Section 7: CEO Briefing")
briefing = strategic.get("ceo_briefing", {})
if not isinstance(briefing, dict):
    briefing = {}

def _as_text(val):
    if isinstance(val, list):
        return " ".join(str(v) for v in val if v)
    return str(val) if val else ""

if briefing:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**What Happened?**")
        st.write(_as_text(briefing.get("what_happened", "")))
    with col2:
        st.markdown("**Why Does It Matter?**")
        st.write(_as_text(briefing.get("why_it_matters", "")))
    with col3:
        st.markdown("**What Should Management Do Next?**")
        st.write(_as_text(briefing.get("what_to_do_next", "")))

    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Articles",        total_docs)
    k2.metric("Opportunities",   len(opportunities))
    k3.metric("Risks",           len(risks))
    k4.metric("Trends",          len(trends))
    k5.metric("Recommendations", len(recs))
else:
    st.info("Select a company and click Run Analysis to generate the CEO briefing.")

st.markdown("---")
# st.caption(f"Strategic Intelligence Agent  |  {company}  |  CrewAI + Ollama")