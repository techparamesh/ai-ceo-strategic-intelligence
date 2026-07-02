"""
Vector knowledge repository (ChromaDB).

Stores the collected articles as embeddings using a local sentence-transformers
model (all-MiniLM-L6-v2) and serves semantic retrieval to the analysis agents.
This is the RAG layer: agents query by meaning, not just keywords.
"""

import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "chroma_store"
COLLECTION  = "company_news"
EMBED_MODEL = "all-MiniLM-L6-v2"

_client = None
_embed_fn = None


def _collection():
    """Lazy singleton: build the client and embedding function once, reuse after."""
    global _client, _embed_fn
    if _client is None:
        _client = chromadb.PersistentClient(path=PERSIST_DIR)
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return _client.get_or_create_collection(name=COLLECTION, embedding_function=_embed_fn)


def index_articles(articles, company):
    """Embed and store articles. Each is tagged with its company so one store can
    hold several companies and we filter at query time. Upsert keeps it idempotent."""
    if not articles:
        return 0

    coll = _collection()
    ids, docs, metas = [], [], []
    for a in articles:
        title = (a.get("title") or "").strip()
        summary = (a.get("summary") or "").strip()
        if not title and not summary:
            continue
        ids.append(a.get("id") or str(abs(hash(title))))
        docs.append(f"{title}. {summary}".strip())          # text that gets embedded
        metas.append({
            "company":   company,
            "source":    a.get("source", ""),
            "title":     title,
            "summary":   summary[:300],
            "link":      a.get("link", ""),
            "published": a.get("published", ""),
        })

    if ids:
        coll.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"  Chroma: indexed {len(ids)} articles for {company}")
    return len(ids)


def search(query, company=None, k=5):
    """Return the k most semantically similar articles, optionally scoped to a company."""
    coll = _collection()
    where = {"company": company} if company else None
    try:
        res = coll.query(query_texts=[query], n_results=k, where=where)
    except Exception:
        return []

    metas = (res.get("metadatas") or [[]])[0]
    return [
        {
            "title":     m.get("title", ""),
            "source":    m.get("source", ""),
            "summary":   m.get("summary", ""),
            "link":      m.get("link", ""),
            "published": m.get("published", ""),
        }
        for m in metas
    ]
