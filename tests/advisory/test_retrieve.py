import numpy as np
import pytest

from floodsitrep.advisory.retrieve import (
    FALLBACK_QUERY,
    claims_to_query,
    load_or_embed,
    load_passages,
    retrieve,
    search,
)
from floodsitrep.nlp.schema import Claim

PASSAGES = [
    {"id": "p1", "language": "en", "text": "do not drive through flooded roads"},
    {"id": "p2", "language": "en", "text": "switch off electricity before leaving"},
    {"id": "p3", "language": "ar", "text": "لا تقد السيارة في الطرق المغمورة"},
]
VECS = {
    "do not drive through flooded roads": [1.0, 0.0, 0.0],
    "switch off electricity before leaving": [0.0, 1.0, 0.0],
    "لا تقد السيارة في الطرق المغمورة": [0.9, 0.0, 0.436],
    "roads": [1.0, 0.0, 0.0],
    "power": [0.0, 1.0, 0.0],
}


class FakeModel:
    """Stand-in for bge-m3: fixed vectors, same encode signature."""

    calls = 0

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        FakeModel.calls += 1
        out = np.array([VECS.get(t, [0.0, 0.0, 1.0]) for t in texts], dtype="float32")
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def vectors():
    return FakeModel().encode([p["text"] for p in PASSAGES])


def test_top_k_order():
    hits = search("roads", FakeModel(), vectors(), PASSAGES, k=3)
    assert [h["id"] for h in hits] == ["p1", "p3", "p2"]
    assert hits[0]["rank"] == 1 and hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


def test_language_filter():
    hits = search("roads", FakeModel(), vectors(), PASSAGES, k=3, language="ar")
    assert [h["id"] for h in hits] == ["p3"]
    assert search("power", FakeModel(), vectors(), PASSAGES, k=1, language="en")[0]["id"] == "p2"


def test_claims_to_query():
    supported = Claim("x", "r0c0", "many", "major", verdict="supported")
    wrong = Claim("y", "r0c1", "few", "destroyed", verdict="contradicted")
    assert claims_to_query([supported, wrong]) == "many buildings with major damage"
    assert claims_to_query([supported, supported]) == "many buildings with major damage"
    assert claims_to_query([]) == FALLBACK_QUERY
    assert claims_to_query([wrong]) == FALLBACK_QUERY
    assert claims_to_query(None).strip()


def test_retrieve_uses_supported_claims_only():
    seen = []

    class Spy(FakeModel):
        def encode(self, texts, **kw):
            seen.extend(texts)
            return super().encode(texts, **kw)

    retrieve([Claim("y", "r0c1", "few", "destroyed", verdict="contradicted")], Spy(), vectors(),
             PASSAGES, k=1)
    assert seen == [FALLBACK_QUERY]


def test_cache_is_reused(tmp_path):
    cache = tmp_path / "v.npy"
    FakeModel.calls = 0
    first = load_or_embed(FakeModel(), PASSAGES, cache)
    second = load_or_embed(FakeModel(), PASSAGES, cache)
    assert FakeModel.calls == 1 and np.allclose(first, second)
    assert load_or_embed(FakeModel(), PASSAGES[:2], cache).shape[0] == 2  # stale cache re-built


def test_load_passages(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("id,language,text\np1,en,hello\n", encoding="utf-8")
    assert load_passages(f)[0]["text"] == "hello"
    bad = tmp_path / "b.csv"
    bad.write_text("id,text\np1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_passages(bad)
