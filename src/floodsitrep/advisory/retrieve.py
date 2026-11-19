"""bge-m3 retrieval of guidance passages for the advisory step. Owner: member 5.

Guidance documents are split into passages (passages.csv on the shared Drive, never in git).
Every passage and the query become vectors with BAAI/bge-m3 (English and Arabic); the passages
closest to the query are returned. The advisory may state only what a returned passage says.

Nothing is loaded at import time. The model, vectors and passages are arguments.
sentence-transformers is imported inside ``load_model`` only.

    passages = load_passages("guidance/passages.csv")
    model = load_model()                                   # BAAI/bge-m3, GPU if present
    vectors = load_or_embed(model, passages, "guidance/vectors.npy")
    hits = retrieve(verified_claims, model, vectors, passages, k=3, language="ar")
"""

import csv
from pathlib import Path

import numpy as np

MODEL_NAME = "BAAI/bge-m3"
FALLBACK_QUERY = "flood safety advice for residents and what to do after a flood"
DAMAGE_WORDS = {"minor": "minor damage", "major": "major damage", "destroyed": "destroyed buildings"}


def load_passages(path):
    """passages.csv -> list of dicts. Needs columns id, language, text; others are kept."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for need in ("id", "language", "text"):
        if rows and need not in rows[0]:
            raise ValueError(f"passages file has no '{need}' column")
    return rows


def load_model(name=MODEL_NAME, device=None):
    """Load the embedding model. The only place sentence-transformers is imported."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


def embed(model, texts, batch_size=16):
    """texts -> float32 matrix, one unit-length row per text."""
    vecs = model.encode(list(texts), normalize_embeddings=True, batch_size=batch_size,
                        show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def load_or_embed(model, passages, cache_path=None):
    """Vectors for the passages. Reuses cache_path (.npy) when it exists and has the right
    number of rows, otherwise embeds and saves. Delete the file after changing passages."""
    if cache_path is not None and Path(cache_path).exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(passages):
            return cached
    vectors = embed(model, [p["text"] for p in passages])
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, vectors)
    return vectors


def claims_to_query(claims):
    """Supported claims -> one English query, e.g. 'many buildings with major damage'.

    Only claims with verdict 'supported' are used. Claims about 'none' damage or without a
    damage class add nothing. With no usable claim a general flood-safety query is returned."""
    parts = []
    for c in claims or []:
        if getattr(c, "verdict", None) != "supported":
            continue
        damage = DAMAGE_WORDS.get(getattr(c, "damage_class", None))
        if damage is None or getattr(c, "quantity", None) in (None, "none"):
            continue
        parts.append(f"{c.quantity} buildings with {damage}")
    parts = list(dict.fromkeys(parts))  # same phrase once, order kept
    return "; ".join(parts) if parts else FALLBACK_QUERY


def search(query, model, vectors, passages, k=3, language=None):
    """Top-k passages for a text query, best first. language 'en' or 'ar' restricts results.

    Returns dicts: rank, score, plus every column of the passage. Cosine similarity, since
    both sides are unit length."""
    q = embed(model, [query])[0]
    scores = vectors @ q
    order = np.argsort(-scores)
    hits = []
    for i in order:
        if language is not None and passages[i].get("language") != language:
            continue
        hits.append({"rank": len(hits) + 1, "score": float(scores[i]), **passages[i]})
        if len(hits) == k:
            break
    return hits


def retrieve(claims, model, vectors, passages, k=3, language=None):
    """Verified claims -> top-k guidance passages (the function the advisory step calls)."""
    return search(claims_to_query(claims), model, vectors, passages, k=k, language=language)
