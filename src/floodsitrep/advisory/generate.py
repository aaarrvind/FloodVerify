"""Advisory text from verified claims and retrieved guidance. Owner: member 5.

Contract (common/interfaces.py): advise(verified_claims, lang) -> str.

Rules kept here, in code and not only in a prompt:
  * Advice comes only from retrieved passages. If retrieve returns nothing, the advisory says
    that no guidance was found and adds nothing. The language model is not called in that case.
  * A model-written bullet is kept only when it cites a passage that was given. Others are dropped.
  * The situation block lists only claims the checker marked "supported".

Nothing heavy is imported here (no numpy, no model), so the pipeline can import this module in CI.
Until a retriever is set (CI, tests) every advisory says no guidance was found.
Plug in the retriever and, optionally, a writer once:

    from floodsitrep.advisory.generate import advise, make_retriever, set_backends
    set_backends(retrieve_fn=make_retriever(model, vectors, passages))   # extractive advisory
    set_backends(retrieve_fn=..., write_fn=lambda messages: generator(messages))   # model-written advice
    advise(claims, "en")

Without a writer the advice is the retrieved passages themselves, each followed by its id. This
needs no language model and cannot add anything. Passage text must not be shown publicly until
the licence questions in the retrieval handoff are answered.
"""

import re

from floodsitrep.nlp.schema import Claim

NO_GUIDANCE = {
    "en": "No guidance found for this situation. No advice is given.",
    # Arabic wording written by the team, not published guidance: an Arabic reader should check it.
    "ar": "لم يتم العثور على إرشادات مناسبة لهذه الحالة، لذلك لا تُقدَّم أي نصيحة.",
}
NO_SITUATION = "No damage statement could be verified from the satellite evidence."
LANG_NAME = {"en": "English", "ar": "Arabic"}
SYSTEM_PROMPT = ("You write short advisories for residents after a flash flood. "
                 "You follow the output format exactly and never invent facts or numbers.")

CITE = re.compile(r"\[\s*P\s*(\d+)\s*\]")
BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s*(.+?)\s*$")
ADVICE_HEADER = re.compile(r"^\W*advice\b", re.IGNORECASE)

_BACKENDS = {"retrieve": None, "write": None}


def set_backends(retrieve_fn=None, write_fn=None):
    """Set the retriever (claims, lang -> list of passage dicts) and the optional writer
    (chat messages -> text) that advise uses when they are not passed as arguments."""
    _BACKENDS["retrieve"], _BACKENDS["write"] = retrieve_fn, write_fn


def as_claim(c):
    """A Claim, or the dict that nlp.checker.verify returns, as a Claim."""
    if isinstance(c, Claim):
        return c
    get = c.get if isinstance(c, dict) else lambda k, d=None: getattr(c, k, d)
    return Claim(text=get("text", "") or "", cell=get("cell"), quantity=get("quantity"),
                 damage_class=get("damage_class"), verdict=get("verdict", "not_verifiable"))


def supported_claims(claims):
    """Only the claims the checker supported, as Claim objects."""
    return [c for c in (as_claim(x) for x in claims or []) if c.verdict == "supported"]


def make_retriever(model, vectors, passages, k=3, min_score=None):
    """Retriever for set_backends: claims (Claim or dict) and a language -> passages at or above
    the cutoff, in that language. retrieve is imported here because it needs numpy."""
    from floodsitrep.advisory import retrieve as R

    cutoff = R.MIN_SCORE if min_score is None else min_score

    def retrieve_fn(claims, lang):
        return R.retrieve([as_claim(c) for c in claims], model, vectors, passages, k=k,
                          language=lang, min_score=cutoff)

    return retrieve_fn


def situation_lines(claims):
    """Supported claims as 'r1c1: few of the buildings show minor damage.', each once."""
    lines = []
    for c in supported_claims(claims):
        line = f"{c.cell}: {c.text}" if c.cell else c.text
        if line not in lines:
            lines.append(line)
    return lines


def build_messages(statements, passages, lang):
    """Chat messages for one advisory. passages must not be empty (the caller handles that case)."""
    stm = "\n".join(statements) if statements else "none"
    psg = "\n".join(f"[P{i}] {p['text']}" for i, p in enumerate(passages, 1))
    user = (f"VERIFIED DAMAGE STATEMENTS:\n{stm}\n\nGUIDANCE PASSAGES:\n{psg}\n\n"
            f"Write 3 to 5 short bullet points of practical advice for residents, one line each, "
            f"in {LANG_NAME[lang]}. Use only the guidance passages above. End every bullet with the "
            "tag of the passage it comes from, for example [P2]. If no passage supports a point, "
            "leave that point out. Do not describe any damage that is not in the statements above.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_advice(text, passages):
    """Bullets of model text that cite only passages that were given, with [P2] replaced by the
    passage id. A bullet without a tag, or with a tag that was not given, is dropped."""
    lines = (text or "").splitlines()
    start = next((i + 1 for i, line in enumerate(lines) if ADVICE_HEADER.match(line)), 0)
    bullets = []
    for line in lines[start:]:
        m = BULLET.match(line)
        if not m:
            continue
        tags = [int(t) for t in CITE.findall(m.group(1))]
        if tags and all(1 <= t <= len(passages) for t in tags):
            bullets.append(CITE.sub(lambda t: f"[{passages[int(t.group(1)) - 1]['id']}]", m.group(1)))
    return bullets


def format_advisory(situation, advice):
    """Situation lines and advice bullets as one text."""
    return ("SITUATION:\n" + "\n".join(situation or [NO_SITUATION])
            + "\n\nADVICE:\n" + "\n".join(f"- {a}" for a in advice))


def advise(verified_claims, lang, retrieve_fn=None, write_fn=None):
    """Verified claims -> advisory text in lang ('en' or 'ar').

    Retrieval returns passages in lang at or above the cutoff, or nothing. With nothing, the
    advice is the fixed no-guidance line. With a writer, the model rewrites the passages as
    bullets and unsupported bullets are dropped; without one, the passages are the advice."""
    if lang not in NO_GUIDANCE:
        raise ValueError(f"lang must be one of {sorted(NO_GUIDANCE)}, got {lang!r}")
    retrieve_fn = retrieve_fn or _BACKENDS["retrieve"]
    write_fn = write_fn or _BACKENDS["write"]
    claims = [as_claim(c) for c in verified_claims or []]
    situation = situation_lines(claims)
    hits = retrieve_fn(claims, lang) if retrieve_fn else []  # no retriever set: no guidance
    if not hits:
        advice = [NO_GUIDANCE[lang]]
    elif write_fn is None:
        advice = [f"{h['text']} [{h['id']}]" for h in hits]
    else:
        written = parse_advice(write_fn(build_messages(situation, hits, lang)), hits)
        advice = written or [NO_GUIDANCE[lang]]
    return format_advisory(situation, advice)
