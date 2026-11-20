import pytest

np = pytest.importorskip("numpy")  # CI installs no numpy; this test runs locally and in Colab

from floodsitrep.advisory import retrieve as R
from floodsitrep.advisory.generate import advise, make_retriever

PASSAGES = [
    {"id": "a1", "language": "en", "text": "stay away from damaged buildings"},
    {"id": "a2", "language": "en", "text": "do not drive through flooded roads"},
    {"id": "b1", "language": "ar", "text": "ابتعد عن المباني المتضررة"},
]
MAJOR = R.ADVICE_FOR["major"]
VECS = {
    "stay away from damaged buildings": [1.0, 0.0, 0.0],
    "do not drive through flooded roads": [0.0, 1.0, 0.0],
    "ابتعد عن المباني المتضررة": [0.9, 0.0, 0.436],
    MAJOR: [1.0, 0.0, 0.0],
    R.FALLBACK_QUERY: [0.0, 0.0, 1.0],
}


class FakeModel:
    def encode(self, texts, normalize_embeddings=True, **kwargs):
        out = np.array([VECS.get(t, [0.0, 0.0, 1.0]) for t in texts], dtype="float32")
        return out / np.linalg.norm(out, axis=1, keepdims=True)


SUPPORTED_DICT = {"text": "many of the buildings show major damage.", "cell": "r1c1", "quantity": "many",
                  "damage_class": "major", "verdict": "supported"}


def setup():
    model = FakeModel()
    return model, model.encode([p["text"] for p in PASSAGES])


def test_dict_claims_reach_the_advice_query_and_do_not_fall_back():
    model, vectors = setup()
    fn = make_retriever(model, vectors, PASSAGES, k=2, min_score=0.5)
    out = advise([SUPPORTED_DICT], "en", retrieve_fn=fn)
    assert "[a1]" in out and "[a2]" not in out


def test_fallback_query_with_nothing_above_the_cutoff_says_no_guidance():
    model, vectors = setup()
    fn = make_retriever(model, vectors, PASSAGES, k=2, min_score=0.5)
    out = advise([], "en", retrieve_fn=fn)
    assert "No guidance found" in out


def test_language_is_respected():
    model, vectors = setup()
    fn = make_retriever(model, vectors, PASSAGES, k=3, min_score=0.5)
    out = advise([SUPPORTED_DICT], "ar", retrieve_fn=fn)
    assert "[b1]" in out and "[a1]" not in out
