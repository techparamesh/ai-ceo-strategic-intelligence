# AI CEO — Strategic Intelligence Agent

A multi-agent system that turns live company news into a structured strategic briefing: market signals, opportunities, risks, sentiment, emerging trends, and CEO-level recommendations — each claim backed by a link to the actual source article.

It runs entirely on a local LLM (Ollama), with a RAG layer on top so the agents can pull specific evidence instead of just summarizing whatever fits in the prompt.

## Why I built this

I wanted to see how far a small local model (Llama 3.1 8B) could go on a task that normally assumes GPT-4-class reasoning: reading a pile of unstructured news and producing something an actual CEO could skim in two minutes. The interesting problems weren't the LLM calls themselves — they were everything around them: getting six agents to stay on-topic without repeating each other, keeping local models from hallucinating headlines, tracing every claim back to a real source, and keeping the whole thing usable when a small model occasionally returns broken JSON.

## How it works

```
RSS feeds (Google News, Yahoo Finance, Reddit)
        │
        ▼
   collect.py  ──►  collected_data.json (raw articles, deduped)
        │
        ▼
   knowledge.py ──► ChromaDB (all-MiniLM-L6-v2 embeddings)
        │
        ▼
   crew.py  ──►  6 CrewAI agents, each scoped to a filtered slice of the articles
        │        (market / opportunity / risk / sentiment / trend / CEO advisor)
        ▼
   analysis_results.json
        │
        ▼
   app.py  ──►  Streamlit dashboard
```

**Collection** — `collect.py` pulls RSS feeds per company (news, finance, Reddit discussion), strips HTML, dedupes by a hash of title+link, and writes a single JSON snapshot.

**Retrieval** — `knowledge.py` embeds every collected article into ChromaDB. Agents don't just get a static news dump; the market agent has a semantic search tool it can call mid-task to pull additional context by meaning, not keyword.

**Analysis** — `crew.py` is the core of the project. Six agents each get a different, pre-filtered slice of the same article set (competitor-tagged articles for the market analyst, risk-keyword articles for the risk officer, and so on) so their outputs don't overlap. A few things I had to solve here that aren't obvious from a first pass:
- **Evidence grounding** — every opportunity, risk, and recommendation has to cite evidence in the model's own words, then that sentence is matched back against the original corpus by keyword overlap (`best_source`) and linked to a real headline and URL. If nothing matches well enough, it's marked unmatched rather than silently faked.
- **Deduplication across agents** — `dedupe_evidence` and `dedupe_market` strip repeated evidence lines so the dashboard doesn't show the same signal three times under different headings.
- **JSON recovery** — local models frequently wrap valid JSON in prose or markdown fences; `extract_json` handles that instead of crashing the pipeline.
- **Context budget** — articles are round-robin diversified across sources and capped at 40; past that, the CEO briefing task (which sees the most context) started overflowing and returning malformed output.

**Presentation** — `app.py` is a Streamlit dashboard with a live VADER sentiment breakdown (computed directly from the collected articles, not just trusted from the LLM), a company selector with sensible presets, and evidence rendered as clickable source links.

## Tech stack

- **CrewAI** — multi-agent orchestration (sequential process, 6 role-based agents)
- **Ollama** (Llama 3.1 8B) — local inference, no API costs
- **ChromaDB + sentence-transformers** (`all-MiniLM-L6-v2`) — vector store for semantic retrieval
- **Streamlit** — dashboard
- **VADER** — deterministic sentiment baseline the LLM interprets rather than invents
- **feedparser** — RSS ingestion

## Setup

Requires [Ollama](https://ollama.com) running locally with a model pulled:

```bash
ollama pull llama3.1:8b
ollama serve
```

Then:

```bash
git clone https://github.com/<your-username>/ai-ceo-strategic-intelligence.git
cd ai-ceo-strategic-intelligence
pip install -r requirements.txt
streamlit run app.py
```

Pick a company from the sidebar (or enter a custom one) and click **Run Analysis**. First run per company takes a couple of minutes — it's collecting live RSS data, embedding it, and running six sequential agent calls against a local model.

## Project structure

```
.
├── app.py          # Streamlit dashboard
├── collect.py       # RSS collection, cleaning, deduplication
├── knowledge.py      # ChromaDB embedding + semantic search
├── crew.py          # Agent definitions, task prompts, orchestration, evidence linking
├── config.py         # Company presets, feed URL builders, model config
└── requirements.txt
```

## Known limitations

- RSS feeds (especially Reddit) can be sparse or rate-limited depending on the company and time of day, which directly affects how much evidence the agents have to work with.
- Running on an 8B local model is a deliberate tradeoff — it's noticeably weaker at structured JSON output than a hosted frontier model, which is most of why `extract_json` and the evidence-matching logic exist in the first place.
- Evidence linking (`best_source`) is keyword-overlap based, not a second embedding lookup — good enough to catch clear mismatches, not a guarantee of a perfect match.

## Possible next steps

- Swap the keyword-overlap evidence matcher for a second ChromaDB lookup
- Add a config option to run against a hosted model (OpenAI/Azure) for comparison against the local model's output quality
- Cache collected data per company with a TTL instead of re-collecting on every run

## License

MIT
