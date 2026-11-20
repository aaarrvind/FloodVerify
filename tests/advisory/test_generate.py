import pytest

from floodsitrep.advisory import generate as G
from floodsitrep.advisory.generate import (
    NO_GUIDANCE,
    NO_SITUATION,
    advise,
    as_claim,
    build_messages,
    parse_advice,
    set_backends,
    situation_lines,
    supported_claims,
)
from floodsitrep.nlp.schema import Claim

HITS = [
    {"id": "READY-D06", "language": "en", "text": "Do not drive through flood waters.", "score": 0.8},
    {"id": "NCEMA-07", "language": "en", "text": "Stay away from power lines.", "score": 0.75},
]
CLAIMS = [
    {"text": "many of the buildings show major damage.", "cell": "r1c1", "quantity": "many",
     "damage_class": "major", "verdict": "supported"},
    {"text": "few of the buildings show minor damage.", "cell": "r0c0", "quantity": "few",
     "damage_class": "minor", "verdict": "contradicted"},
    {"text": "some of the buildings are destroyed.", "cell": "r2c2", "quantity": "some",
     "damage_class": "destroyed", "verdict": "not_verifiable"},
]


@pytest.fixture(autouse=True)
def reset_backends():
    set_backends()
    yield
    set_backends()


def test_dicts_from_the_checker_become_claims():
    c = as_claim(CLAIMS[0])
    assert isinstance(c, Claim) and c.verdict == "supported" and c.cell == "r1c1"
    assert as_claim(c) is c


def test_only_supported_claims_are_used():
    assert [c.cell for c in supported_claims(CLAIMS)] == ["r1c1"]
    assert situation_lines(CLAIMS) == ["r1c1: many of the buildings show major damage."]
    assert supported_claims(None) == []


def test_empty_retrieval_gives_no_guidance_and_never_calls_the_writer():
    calls = []
    out = advise(CLAIMS, "en", retrieve_fn=lambda claims, lang: [], write_fn=lambda m: calls.append(m) or "x")
    assert calls == []
    assert NO_GUIDANCE["en"] in out and out.count("- ") == 1
    assert "r1c1: many of the buildings show major damage." in out


def test_arabic_no_guidance_line_and_unknown_language():
    out = advise([], "ar", retrieve_fn=lambda claims, lang: [])
    assert NO_GUIDANCE["ar"] in out and NO_SITUATION in out
    with pytest.raises(ValueError):
        advise([], "fr", retrieve_fn=lambda claims, lang: [])


def test_without_a_writer_the_advice_is_the_passages_with_ids():
    out = advise(CLAIMS, "en", retrieve_fn=lambda claims, lang: HITS)
    assert "- Do not drive through flood waters. [READY-D06]" in out
    assert "- Stay away from power lines. [NCEMA-07]" in out


def test_writer_bullets_need_a_valid_citation():
    text = ("SITUATION:\nr1c1: many of the buildings show major damage.\n\nADVICE:\n"
            "- Avoid driving through flood water. [P1]\n"
            "- Everyone should buy sandbags.\n"
            "- Keep away from power lines. [P2]\n"
            "- Something invented. [P9]\n")
    out = advise(CLAIMS, "en", retrieve_fn=lambda c, l: HITS, write_fn=lambda m: text)
    assert "- Avoid driving through flood water. [READY-D06]" in out
    assert "- Keep away from power lines. [NCEMA-07]" in out
    assert "sandbags" not in out and "invented" not in out


def test_writer_with_no_valid_bullet_falls_back_to_no_guidance():
    out = advise(CLAIMS, "en", retrieve_fn=lambda c, l: HITS, write_fn=lambda m: "ADVICE:\n- no tags here")
    assert NO_GUIDANCE["en"] in out


def test_parse_advice_ignores_the_situation_section_and_handles_numbering():
    text = "SITUATION:\n- r1c1: not a bullet to keep [P1]\n\nADVICE:\n1. Stay calm. [P1]\n2) Stay dry. [P2]"
    assert parse_advice(text, HITS) == ["Stay calm. [READY-D06]", "Stay dry. [NCEMA-07]"]


def test_prompt_contains_only_given_passages_and_statements():
    msgs = build_messages(["r1c1: many of the buildings show major damage."], HITS, "ar")
    user = msgs[1]["content"]
    assert "[P1] Do not drive through flood waters." in user and "[P2]" in user
    assert "Arabic" in user and "r1c1: many of the buildings show major damage." in user


def test_backends_are_used_when_not_passed():
    assert NO_GUIDANCE["en"] in advise(CLAIMS, "en")  # nothing configured: no guidance, no error
    set_backends(retrieve_fn=lambda c, l: HITS)
    assert "[READY-D06]" in advise(CLAIMS, "en")
    assert G._BACKENDS["write"] is None


def test_retriever_gets_claims_as_objects_whatever_the_input():
    seen = []
    advise(CLAIMS, "en", retrieve_fn=lambda claims, lang: seen.append((claims, lang)) or [])
    claims, lang = seen[0]
    assert lang == "en" and all(isinstance(c, Claim) for c in claims)
