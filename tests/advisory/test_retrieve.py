import pytest

np = pytest.importorskip("numpy")  # CI installs no numpy; the test runs locally and in Colab

from floodsitrep.advisory.retrieve import (
    FALLBACK_QUERY,
    claims_to_query,
    embed_text,
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


def test_min_score_drops_weak_hits():
    all_hits = search("roads", FakeModel(), vectors(), PASSAGES, k=3)
    assert len(all_hits) == 3
    strong = search("roads", FakeModel(), vectors(), PASSAGES, k=3, min_score=0.6)
    assert [h["id"] for h in strong] == ["p1", "p3"]  # p2 scores 0 for this query
    assert search("roads", FakeModel(), vectors(), PASSAGES, k=3, min_score=1.1) == []
    assert search("roads", FakeModel(), vectors(), PASSAGES, k=3, language="en", min_score=0.6)[0]["id"] == "p1"


def test_retrieve_returns_nothing_below_cutoff():
    # fallback query points at an unrelated direction, so no passage reaches the default cutoff
    assert retrieve([], FakeModel(), vectors(), PASSAGES, k=3) == []
    assert len(retrieve([], FakeModel(), vectors(), PASSAGES, k=3, min_score=None)) == 3


def test_claims_to_query():
    supported = Claim("x", "r0c0", "many", "major", verdict="supported")
    wrong = Claim("y", "r0c1", "few", "destroyed", verdict="contradicted")
    major_query = claims_to_query([supported, wrong])
    assert "damaged buildings" in major_query and "evacuation" in major_query
    assert claims_to_query([supported, supported]) == major_query
    destroyed = Claim("z", "r1c1", "some", "destroyed", verdict="supported")
    assert claims_to_query([supported, destroyed]) == major_query  # same advice, said once
    minor = Claim("m", "r2c2", "few", "minor", verdict="supported")
    assert claims_to_query([minor]) != major_query
    assert claims_to_query([Claim("n", "r0c0", "none", "damaged", verdict="supported")]) == FALLBACK_QUERY
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


def test_embed_text_is_used_when_present_and_text_is_still_returned():
    assert embed_text({"text": "plain"}) == "plain"
    assert embed_text({"text": "plain", "embed_text": ""}) == "plain"  # empty column: fall back
    assert embed_text({"text": "plain", "embed_text": "Source: plain"}) == "Source: plain"
    passages = [{"id": "p1", "language": "en", "text": "do not drive through flooded roads",
                 "embed_text": "switch off electricity before leaving"}]
    seen = []

    class Spy(FakeModel):
        def encode(self, texts, **kw):
            seen.extend(texts)
            return super().encode(texts, **kw)

    vecs = load_or_embed(Spy(), passages)
    assert seen == ["switch off electricity before leaving"]  # the embed_text, not the shown text
    hit = search("power", FakeModel(), vecs, passages, k=1)[0]
    assert hit["text"] == "do not drive through flooded roads"  # reader still sees the plain text
