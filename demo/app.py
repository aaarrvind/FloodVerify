"""Gradio demo for the Hugging Face Space. Owner: member 5.

Runs on CPU with no model. For one tile it shows the imagery with the damage and the 9 areas, checks a caption
against the evidence with the project's real checker (nlp.checker.verify), marks each caption line and each area
with the verdict, and writes a verified situation report from the confirmed statements only.

Optional files in demo/data (the app works without them); build them with demo/build_bundle.py:
  evidence/<tile>.json   evidence for the tiles that have captions
  captions.json          {tile_id: "r0c0: few of the buildings show minor damage.\\n..."}
  images/<tile>_{pre,post,damage}.jpg   imagery (xBD is CC BY-NC-SA: local or private Space only)
  advisories.json        {tile_id: {"en": "...", "ar": "..."}}
  tweet_stats.json       counts per event, day and class (from demo/export_tweet_stats.py, no text)
  tweet_examples.json    {event: [{text, class, score, day}]} example tweets (private Space only)
  linked_tweets.csv      tile_id, event, tweet_id, score, text (text only if the licence allows)
"""

import csv
import html
import inspect
import json
import math
import os
import re
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from floodsitrep.common.interfaces import CLASSES
from floodsitrep.nlp.checker import MIN_VALID, bin_of, class_fraction, verify
from floodsitrep.nlp.claims import CELL_LINE, parse_line

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMG_DIR = DATA / "images"
EVIDENCE_DIR = Path(
    os.environ.get("FV_EVIDENCE_DIR")
    or (
        DATA / "evidence"
        if any((DATA / "evidence").glob("*.json"))
        else ROOT.parent / "evidence" / "samples"
    )
)

NAMES = {
    "r0c0": "Top left",
    "r0c1": "Top middle",
    "r0c2": "Top right",
    "r1c0": "Middle left",
    "r1c1": "Centre",
    "r1c2": "Middle right",
    "r2c0": "Bottom left",
    "r2c1": "Bottom middle",
    "r2c2": "Bottom right",
}
GRID = [f"r{r}c{c}" for r in range(3) for c in range(3)]
WORDS = {
    "supported": ("✓", "Matches"),
    "contradicted": ("✕", "Does not match"),
    "not_verifiable": ("–", "Cannot check"),
}
SOURCES = {"labels": "ground-truth building labels", "predicted": "model-predicted buildings"}
XBD_TO_HUMAID = {
    "hurricane-harvey": "hurricane_harvey_2017",
    "hurricane-florence": "hurricane_florence_2018",
    "hurricane-matthew": "hurricane_matthew_2016",
    "midwest-flooding": "midwestern_us_floods_2019",
}
# Mask codes written by build_bundle.py: 0 not a building, 1 undamaged, 2 minor, 3 major, 4 destroyed.
LAYERS = {
    1: ("Undamaged", (122, 184, 240)),
    2: ("Minor", (255, 209, 102)),
    3: ("Major", (242, 140, 40)),
    4: ("Destroyed", (181, 64, 14)),
}
STATE_RGB = {
    "supported": (31, 122, 90),
    "contradicted": (178, 58, 31),
    "not_verifiable": (136, 150, 160),
}
STATE_TAG = {"supported": "OK", "contradicted": "WRONG", "not_verifiable": "?"}
SEVERITY = {"contradicted": 0, "supported": 1, "not_verifiable": 2}


def _json(path):
    return json.loads(path.read_text()) if path.exists() else {}


TILES = {p.stem: json.loads(p.read_text()) for p in sorted(EVIDENCE_DIR.glob("*.json"))}
CAPTIONS = _json(DATA / "captions.json")
ADVISORIES = _json(DATA / "advisories.json")
TWEET_STATS = _json(DATA / "tweet_stats.json")
TWEET_EXAMPLES = _json(DATA / "tweet_examples.json")


def _load_tweets():
    path = DATA / "linked_tweets.csv"
    out = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out.setdefault(row["tile_id"], []).append(row)
    return out


TWEETS = _load_tweets()


def esc(x):
    return html.escape(str(x))


_TAG = re.compile(r"(#\w+|@\w+)")


def _tweet_text(t):
    """Escaped tweet text with #hashtags and @names in the link colour, like a real post."""
    return _TAG.sub(r"<span class='tg'>\1</span>", esc(html.unescape(str(t))))


def tweet_list(he, n=None, skip=0):
    """Example tweets for an event drawn as post cards, or '' when none were added."""
    rows = TWEET_EXAMPLES.get(he) or []
    rows = rows[skip : (skip + n) if n else None]
    if not rows:
        return ""
    out = []
    for r in rows:
        cls = esc(str(r.get("class", "")).replace("_", " "))
        day = esc(str(r.get("day", "")))
        out.append(
            "<li class='tweet'><div class='tw-av' aria-hidden='true'>"
            "<svg viewBox='0 0 24 24' width='22' height='22'><circle cx='12' cy='9' r='4.2' fill='currentColor'/>"
            "<path d='M3.5 21c.7-4.6 4.2-6.7 8.5-6.7s7.8 2.1 8.5 6.7z' fill='currentColor'/></svg></div>"
            "<div class='tw-body'><div class='tw-head'><b>Public post</b><span class='tw-h'>@user</span>"
            + (f"<span class='tw-d'>· {day}</span>" if day else "")
            + "</div>"
            f"<p class='tx'>{_tweet_text(r['text'])}</p>"
            + (f"<span class='tw-c'>{cls}</span>" if cls else "")
            + "</div></li>"
        )
    return "<ul class='tweets' aria-label='Example public posts'>" + "".join(out) + "</ul>"


# ---------- tiles and words ----------


def tile_event(tile_id):
    return tile_id.rsplit("_", 2)[0]


def tile_title(tile_id):
    event, _, rest = tile_id.partition("_")
    num, _, part = rest.partition("_")
    part = part[1:] if part[:1] == "p" and part[1:].isdigit() else part
    return f"{event_name(event)}  ·  Place {int(num) if num.isdigit() else num}, part {part}"


def event_name(event):
    """'hurricane-florence' -> 'Hurricane Florence'."""
    return event.replace("-", " ").replace("_", " ").title()


def _tile_key(t):
    event, _, rest = t.partition("_")
    num, _, part = rest.partition("_")
    return (event, int(num) if num.isdigit() else 0, part)


def tiles_of(event):
    """Tile ids of one event, in a fixed order."""
    return sorted((t for t in TILES if tile_event(t) == event), key=_tile_key)


def event_choices():
    """(label, event) for the event dropdown, with the number of places it covers."""
    events = sorted({tile_event(t) for t in TILES})
    return [
        (f"{event_name(e)}  ·  {len(tiles_of(e))} place{'s' if len(tiles_of(e)) != 1 else ''}", e)
        for e in events
    ]


def default_tile(event):
    """The place shown first for an event: the first one that has a caption, else the first one."""
    ids = tiles_of(event)
    return next((t for t in ids if t in CAPTIONS), ids[0] if ids else None)


# Events the vision model never saw in training. Per notebook 04, Nepal is the held-out test event.
HELD_OUT = {"nepal-flooding"}


def place_note(tile_id):
    """'Showing place 3 of 8 in Hurricane Florence.' plus a note when the event was held out of training."""
    if not tile_id:
        return ""
    ids = tiles_of(tile_event(tile_id))
    i = ids.index(tile_id) + 1 if tile_id in ids else 1
    held = (
        " <b>Unseen event:</b> the damage model was never trained on it, so the evidence below is the model's "
        "own prediction, not ground truth."
        if tile_event(tile_id) in HELD_OUT
        else ""
    )
    return (
        f"<p class='note place-note'>Showing place {i} of {len(ids)} for <b>{esc(event_name(tile_event(tile_id)))}</b>. "
        f"Each place is one satellite view of an affected area.{held}</p>"
    )


def tile_choices():
    def key(t):
        event, _, rest = t.partition("_")
        num, _, part = rest.partition("_")
        return (event, int(num) if num.isdigit() else 0, part)

    return [
        (tile_title(t) + ("" if t in CAPTIONS else "  ·  no caption"), t)
        for t in sorted(TILES, key=key)
    ]


def step_tile(tile_id, step):
    """Next or previous place within the same event."""
    ids = tiles_of(tile_event(tile_id)) if tile_id else []
    if tile_id not in ids:
        return ids[0] if ids else tile_id
    return ids[(ids.index(tile_id) + step) % len(ids)]


def reachable_words(n):
    """Quantity words that some whole number of buildings out of n can produce."""
    return {bin_of(k / n) for k in range(0, n + 1)} if n > 0 else set()


def min_buildings(word):
    """Fewest buildings in an area at which the word can be true."""
    return next((n for n in range(1, 400) if word in reachable_words(n)), None)


def reason(claim, evidence):
    """Plain-language reason for one verdict."""
    cell = next((c for c in evidence["cells"] if c["cell"] == claim["cell"]), None)
    if cell is None:
        return "This area is not in the evidence."
    n, q = cell["buildings"], claim["quantity"]
    plural = "s" if n != 1 else ""
    if claim["verdict"] == "not_verifiable":
        if cell.get("valid_fraction", 1.0) < MIN_VALID:
            return "Too little usable imagery in this area to count buildings."
        if n == 0:
            return "There are no buildings in this area, so there is nothing to compare."
        if q in ("few", "some", "many") and n < (min_buildings(q) or 0):
            return (
                f"Too few buildings here ({n}) to judge “{q}”; we need at least {min_buildings(q)}."
            )
        return "The statement could not be turned into something the evidence can answer."
    share = class_fraction([cell], claim["damage_class"])
    word = "none" if share == 0 else bin_of(share)
    if claim["damage_class"] == "damaged":
        base = f"{share:.0%} of the {n} building{plural} are damaged"
    else:
        base = f"{share:.0%} of the {n} building{plural} have {claim['damage_class']} damage"
    return f"{base}, which counts as “{word}”."


def sample_report(evidence):
    """Report written from the evidence itself, used when a tile has no caption."""
    lines = []
    for c in evidence["cells"]:
        if c["buildings"] == 0:
            lines.append(f"{c['cell']}: no buildings.")
        elif class_fraction([c], "damaged") == 0:
            lines.append(f"{c['cell']}: no damage.")
        else:
            cls = max(CLASSES[1:], key=lambda k: class_fraction([c], k))
            lines.append(
                f"{c['cell']}: {bin_of(class_fraction([c], cls))} of the buildings show {cls} damage."
            )
    return "\n".join(lines)


def default_report(tile_id):
    return CAPTIONS.get(tile_id) or sample_report(TILES[tile_id])


def pick_examples():
    """Three captioned tiles that tell the story: a right caption, a wrong claim caught, claims that cannot be checked."""
    stats = {}
    for t in CAPTIONS:
        if t in TILES:
            cl = verify(CAPTIONS[t], TILES[t])
            k = {v: sum(c["verdict"] == v for c in cl) for v in WORDS}
            stats[t] = (k, len(cl))
    if not stats:
        return {}
    right = max(
        stats,
        key=lambda t: (
            stats[t][0]["supported"] - 2 * stats[t][0]["contradicted"],
            stats[t][0]["supported"],
        ),
    )
    wrong = max(
        (t for t in stats if t != right), key=lambda t: stats[t][0]["contradicted"], default=None
    )
    cannot = max(
        (t for t in stats if t not in (right, wrong)),
        key=lambda t: stats[t][0]["not_verifiable"] / max(stats[t][1], 1),
        default=None,
    )
    ex = {
        "A caption that is right": right if stats[right][0]["supported"] else None,
        "A wrong claim, caught": wrong if wrong and stats[wrong][0]["contradicted"] else None,
        "Claims that cannot be checked": cannot
        if cannot and stats[cannot][0]["not_verifiable"]
        else None,
    }
    return {k: v for k, v in ex.items() if v}


EXAMPLES = pick_examples()


# ---------- imagery ----------


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def cell_states(claims):
    states = {}
    for c in claims:
        cur = states.get(c["cell"])
        if cur is None or SEVERITY[c["verdict"]] < SEVERITY[cur]:
            states[c["cell"]] = c["verdict"]
    return states


def _open(tile_id, name):
    path = IMG_DIR / f"{tile_id}_{name}"
    return Image.open(path).convert("RGB") if path.exists() else None


def _grid_lines(img):
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for i in (1, 2):
        d.line([(i * w / 3, 0), (i * w / 3, h)], fill=(255, 255, 255, 140), width=1)
        d.line([(0, i * h / 3), (w, i * h / 3)], fill=(255, 255, 255, 140), width=1)
    return img


def damage_view(post, mask, alpha, keep):
    """The after-flood photo with every building coloured by its damage. Strength sets how solid the colour is.
    With a spotlight, everything else is greyed out so the chosen classes stand out."""
    arr = np.asarray(post, dtype=np.float32)
    m = np.asarray(mask)
    if keep:
        grey = np.asarray(post.convert("L").convert("RGB"), dtype=np.float32) * 0.6 + 60
        out, strength = grey, 1.0
    else:
        out, strength = arr.copy(), float(alpha)
    for code, (name, rgb) in LAYERS.items():
        if keep and name not in keep:
            continue
        sel = m == code
        out[sel] = (arr if not keep else arr)[sel] * (1 - strength) + np.array(
            rgb, dtype=np.float32
        ) * strength
    return _grid_lines(Image.fromarray(out.astype(np.uint8)))


def schematic(evidence, size=512):
    """Used when no imagery is bundled: each area filled by its damaged share, with the building count."""
    img = Image.new("RGB", (size, size), (20, 34, 47))
    d = ImageDraw.Draw(img)
    cw = size / 3
    big, small = _font(size // 21), _font(size // 28)
    for c in evidence["cells"]:
        r, col = int(c["cell"][1]), int(c["cell"][3])
        x0, y0 = col * cw, r * cw
        n, valid = c["buildings"], c.get("valid_fraction", 1.0)
        share = class_fraction([c], "damaged") if n else None
        if valid < MIN_VALID:
            fill, lines = (58, 70, 82), ["no usable", "imagery"]
        elif not n:
            fill, lines = (27, 44, 60), ["no buildings"]
        else:
            t = share or 0.0
            fill = tuple(int(a + (b - a) * t) for a, b in zip((52, 84, 110), (242, 140, 40)))
            lines = [f"{n} building{'s' if n != 1 else ''}", f"{share:.0%} damaged"]
        d.rectangle([x0 + 2, y0 + 2, x0 + cw - 2, y0 + cw - 2], fill=fill)
        ink = (15, 36, 56) if sum(fill) > 420 else (240, 246, 250)
        for i, line in enumerate(lines):
            d.text(
                (x0 + 10, y0 + 12 + i * size // 15),
                line,
                fill=ink,
                font=big if i == 0 and n and valid >= MIN_VALID else small,
            )
    return img


def short_reason(claim, evidence):
    """A few words saying why an area cannot be checked, for the label on the map."""
    cell = (
        next((c for c in evidence["cells"] if c["cell"] == claim["cell"]), None)
        if evidence
        else None
    )
    if cell is None:
        return "Not in the evidence"
    if cell.get("valid_fraction", 1.0) < MIN_VALID:
        return "Too little imagery"
    if cell["buildings"] == 0:
        return "No buildings here"
    if claim["quantity"] in ("few", "some", "many"):
        return "Too few buildings"
    return "Can't be measured"


def _dashed(d, box, rgb, width=3, dash=14, gap=9):
    x0, y0, x1, y1 = box
    for ax, ay, bx, by in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = math.hypot(bx - ax, by - ay)
        t = 0.0
        while t < length:
            e = min(t + dash, length)
            d.line(
                [
                    (ax + (bx - ax) * t / length, ay + (by - ay) * t / length),
                    (ax + (bx - ax) * e / length, ay + (by - ay) * e / length),
                ],
                fill=(*rgb, 255),
                width=width,
            )
            t += dash + gap


def verdict_view(base, claims, evidence=None):
    """The photo dimmed, with each area marked by what the check found: green matches, red disagrees,
    dashed grey cannot be checked (with the reason), untouched means the caption does not mention it."""
    img = Image.fromarray((np.asarray(base, dtype=np.float32) * 0.5).astype(np.uint8)).convert(
        "RGBA"
    )
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    cw, ch = w / 3, h / 3
    font = _font(max(13, w // 28))
    small = _font(max(12, w // 33))
    alpha = {"contradicted": 150, "supported": 55, "not_verifiable": 0}
    first_nv = {}
    for c in claims:
        if c["verdict"] == "not_verifiable":
            first_nv.setdefault(c["cell"], c)
    for cid, state in cell_states(claims).items():
        r, c = int(cid[1]), int(cid[3])
        x0, y0, x1, y1 = c * cw + 3, r * ch + 3, (c + 1) * cw - 3, (r + 1) * ch - 3
        rgb = STATE_RGB[state]
        if state == "not_verifiable":
            rgb = (205, 215, 223)
            _dashed(d, (x0 + 2, y0 + 2, x1 - 2, y1 - 2), rgb, width=3)
            tag = "CAN'T CHECK"
            fill = (60, 72, 82)
        else:
            d.rectangle([x0, y0, x1, y1], fill=(*rgb, alpha[state]), outline=(*rgb, 255), width=4)
            tag = STATE_TAG[state]
            fill = rgb
        th = (font.size if hasattr(font, "size") else 16) + 10
        tw = d.textlength(tag, font=font)
        ty = (
            y1 - th - 2
        )  # tags sit at the bottom-left so the panel's title bar and toolbar never cover them
        d.rectangle([x0 + 2, ty, x0 + tw + 16, ty + th], fill=(*fill, 245))
        d.text((x0 + 9, ty + 4), tag, fill=(255, 255, 255, 255), font=font)
        if state == "not_verifiable" and cid in first_nv:
            why = short_reason(first_nv[cid], evidence)
            sh = (small.size if hasattr(small, "size") else 13) + 10
            sw = d.textlength(why, font=small)
            sy = ty - sh - 4
            d.rectangle([x0 + 2, sy, x0 + sw + 16, sy + sh], fill=(12, 22, 32, 235))
            d.text((x0 + 9, sy + 5), why, fill=(235, 241, 246, 255), font=small)
    return _grid_lines(img.convert("RGB"))


def why_html(claims, evidence):
    """Plain explanation, under the maps, of every area drawn as dashed grey."""
    rows, seen = [], set()
    for c in claims:
        if c["verdict"] != "not_verifiable" or c["cell"] in seen:
            continue
        if cell_states(claims).get(c["cell"]) != "not_verifiable":
            continue
        seen.add(c["cell"])
        rows.append(
            f"<li><b>{NAMES.get(c['cell'], c['cell'])}:</b> {esc(reason(c, evidence))}</li>"
        )
    if not rows:
        return ""
    return (
        '<div class="why"><b>Why are some areas dashed grey?</b>'
        "<p>The caption mentions these areas, but the building evidence cannot confirm or contradict what it says. "
        "Nothing here is marked as wrong.</p><ul>" + "".join(sorted(rows)) + "</ul></div>"
    )


def panels(tile_id, evidence, claims, alpha, keep):
    """(before, after with damage, where the caption is wrong, has imagery)."""
    pre, post, mask = (
        _open(tile_id, "pre.jpg"),
        _open(tile_id, "post.jpg"),
        IMG_DIR / f"{tile_id}_mask.png",
    )
    if post is None:
        base = schematic(evidence)
        return None, _grid_lines(base.copy()), verdict_view(base, claims, evidence), False
    if mask.exists():
        after = damage_view(post, Image.open(mask), alpha, keep)
    else:
        after = (
            _open(tile_id, "damage.jpg") or post
        )  # older bundle: outlines already drawn into the image
    return (
        _grid_lines(pre.copy()) if pre else None,
        after,
        verdict_view(post, claims, evidence),
        True,
    )


# ---------- HTML pieces ----------


def grid_html(evidence, claims):
    """Counts per area: buildings, damage mix and the verdict on each statement."""
    by_cell = {c["cell"]: c for c in evidence["cells"]}
    stmts = {}
    for cl in claims:
        stmts.setdefault(cl["cell"], []).append(cl)
    cells = []
    for cid in GRID:
        c = by_cell.get(cid)
        if c is None:
            continue
        n, valid, f = c["buildings"], c.get("valid_fraction", 1.0), c["class_fraction"]
        cls = "cell"
        if valid < MIN_VALID:
            body, cls = '<div class="big">No usable imagery</div>', cls + " nodata"
        elif n == 0:
            body, cls = '<div class="big">No buildings</div>', cls + " empty"
        else:
            damaged = class_fraction([c], "damaged")
            seg = "".join(
                f'<i class="d-{k}" style="width:{f[k] * 100:.1f}%"></i>' for k in CLASSES[1:]
            )
            body = (
                f'<div class="big">{n}<span> building{"s" if n != 1 else ""}</span></div>'
                f'<div class="bar" role="img" aria-label="{damaged:.0%} damaged">{seg}</div>'
                f'<div class="dmg">{damaged:.0%} damaged</div>'
            )
        chips = "".join(
            f'<details class="chip {s["verdict"]}"><summary><b>{WORDS[s["verdict"]][0]}</b> {esc(s["quantity"])} · '
            f'{esc(s["damage_class"])}<span class="sr-only"> ({WORDS[s["verdict"]][1]}), show reason</span></summary>'
            f'<div class="tip">{esc(reason(s, evidence))}</div></details>'
            for s in stmts.get(cid, [])
        )
        cells.append(
            f'<div class="{cls}"><div class="name">{NAMES[cid]}<small>{cid}</small></div>{body}'
            f'<div class="chips">{chips}</div></div>'
        )
    legend = (
        '<div class="legend"><span><i class="d-minor"></i>minor</span><span><i class="d-major"></i>major</span>'
        '<span><i class="d-destroyed"></i>destroyed</span></div>'
    )
    src = SOURCES.get(evidence["source"], evidence["source"])
    return (
        f'<div class="fv-grid" role="group" aria-label="Counts for the 3 by 3 areas of the place">{"".join(cells)}</div>'
        f'{legend}<p class="note">Evidence: {esc(src)}. Tap a tag to see its reason.</p>'
    )


def summary_html(claims):
    if not claims:
        return (
            '<div class="empty-state"><b>No statements found.</b> Write one line per area, like '
            "<code>r1c1: few of the buildings show minor damage.</code></div>"
        )
    k = {v: sum(c["verdict"] == v for c in claims) for v in WORDS}
    total = len(claims)
    bar = "".join(
        f'<i class="{v}" style="width:{k[v] / total * 100:.1f}%"></i>' for v in WORDS if k[v]
    )
    checked = k["supported"] + k["contradicted"]
    lead = (
        f"{checked} of {total} statements could be checked."
        if checked
        else f"None of the {total} statements could be checked."
    )
    counts = "".join(
        f'<li class="{v}"><b>{WORDS[v][0]}</b> {k[v]} {WORDS[v][1].lower()}</li>' for v in WORDS
    )
    return (
        f'<div class="result-head"><h3>{lead}</h3><div class="stack" role="img" '
        f'aria-label="{k["supported"]} match, {k["contradicted"]} do not match, {k["not_verifiable"]} cannot be checked">'
        f'{bar}</div><ul class="counts">{counts}</ul></div>'
    )


def confidence_html(claim, evidence):
    """How much to trust the evidence behind one verdict: the area's evidence confidence, its building count and
    how much of it has usable imagery. Shown next to the verdict so a match on thin evidence reads differently."""
    cell = next((c for c in evidence["cells"] if c["cell"] == claim["cell"]), None)
    if cell is None or cell.get("confidence") is None:
        return ""
    conf = max(0.0, min(1.0, float(cell["confidence"])))
    n, valid = cell["buildings"], cell.get("valid_fraction", 1.0)
    return (
        f'<div class="meta"><span class="conf" role="img" aria-label="evidence confidence {conf:.2f} out of 1">'
        f'<i style="width:{conf * 100:.0f}%"></i></span>'
        f"<span>Evidence confidence <b>{conf:.2f}</b> · {n} building{'s' if n != 1 else ''} · "
        f"{valid:.0%} usable imagery</span></div>"
    )


def cards_html(claims, evidence):
    out = []
    for c in sorted(claims, key=lambda c: (SEVERITY[c["verdict"]], c["cell"])):
        icon, word = WORDS[c["verdict"]]
        out.append(
            f'<li class="card {c["verdict"]}"><div class="v"><b>{icon}</b> {word}</div>'
            f'<div class="t"><span class="where">{NAMES.get(c["cell"], c["cell"])}</span> '
            f'“{esc(c["text"].rstrip("."))}”</div><div class="r">{esc(reason(c, evidence))}</div>'
            f"{confidence_html(c, evidence)}</li>"
        )
    return f'<ul class="cards">{"".join(out)}</ul>' if out else ""


def line_issues(report_text):
    """Caption lines the checker will not use, each with the reason, so nothing is dropped silently."""
    found = []
    for raw in report_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = CELL_LINE.match(line)
        if not m:
            bad_id = re.match(r"^\s*(r\d+c\d+)\s*:", line, re.IGNORECASE)
            why = (
                f"{bad_id.group(1)} is not an area. Areas run from r0c0 (top left) to r2c2 (bottom right)."
                if bad_id
                else "It does not start with an area such as r1c1:"
            )
            found.append((line, why))
            continue
        kind, pairs = parse_line(m.group(2))
        if kind == "claims" and not pairs:
            found.append(
                (
                    line,
                    "No damage statement found. Use “few / some / many / most of the buildings show "
                    "minor / major / destroyed damage” or “no damage”.",
                )
            )
    return found


def issues_html(report_text):
    found = line_issues(report_text)
    if not found:
        return ""
    n = len(found)
    items = "".join(
        f"<li><code>{esc(l[:90] + ('…' if len(l) > 90 else ''))}</code> {esc(w)}</li>"
        for l, w in found
    )
    return (
        f'<div class="warn" role="alert"><b>{n} line{"s were" if n != 1 else " was"} skipped and '
        f"not checked.</b><ul>{items}</ul></div>"
    )


def highlighted(report_text, claims):
    """Caption lines as (text, label) pieces for the highlighted-text component."""
    by_cell = cell_states(claims)
    pieces = []
    for raw in report_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        cid = line.split(":", 1)[0].strip().lower()
        shown = (
            f"{NAMES[cid]}: {line.split(':', 1)[1].strip()}"
            if cid in NAMES and ":" in line
            else line
        )
        st = by_cell.get(cid)
        pieces.append((shown, WORDS[st][1] if st else None))
        pieces.append(("\n", None))
    return pieces


def caption_summary(ok):
    """One plain sentence from the confirmed statements only (no model)."""
    if not ok:
        return ""
    dmg = [c for c in ok if c.get("quantity") != "none"]
    none = [c for c in ok if c.get("quantity") == "none"]

    def label(c):
        cls = c.get("damage_class", "damaged")
        return f"{NAMES.get(c['cell'], c['cell']).lower()} ({c['quantity']} {'damaged' if cls == 'damaged' else cls + ' damage'})"

    def join(items):
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    parts = []
    if dmg:
        parts.append(f"Damage is confirmed in the {join([label(c) for c in dmg])}.")
    if none:
        parts.append(
            f"No damage is confirmed in the {join([NAMES.get(c['cell'], c['cell']).lower() for c in none])}."
        )
    return " ".join(parts)


def tweet_topics(ev_stats, n=4):
    """Most common tweet topics for the event, as a short list with shares."""
    cls = sorted((ev_stats.get("classes") or {}).items(), key=lambda kv: -kv[1])[:n]
    tot = max(sum((ev_stats.get("classes") or {}).values()), 1)
    if not cls:
        return ""
    return (
        "<h4 class='ex-h'>Most common topics</h4><ul class='topics'>"
        + "".join(
            f"<li><span>{esc(k.replace('_', ' '))}</span><b>{v / tot:.0%}</b></li>" for k, v in cls
        )
        + "</ul>"
    )


def sitrep_html(tile_id, evidence, claims):
    """The verified situation report: only what the evidence supports, shown as the main result."""
    ok = sorted((c for c in claims if c["verdict"] == "supported"), key=lambda c: c["cell"])
    bad = sorted((c for c in claims if c["verdict"] == "contradicted"), key=lambda c: c["cell"])
    nv = sorted((c for c in claims if c["verdict"] == "not_verifiable"), key=lambda c: c["cell"])
    plural = lambda n: "s" if n != 1 else ""

    def item(c, mark, cls, note):
        return (
            f'<li class="{cls}"><span class="mk">{mark}</span><div><b>{NAMES.get(c["cell"], c["cell"])}</b> '
            f"{esc(c['text'].rstrip('.'))}.<small>{esc(note)}</small></div></li>"
        )

    keep_list = (
        (
            "<ul class='sr-list'>"
            + "".join(item(c, "✓", "yes", reason(c, evidence)) for c in ok)
            + "</ul>"
        )
        if ok
        else "<p class='sr-empty'>Nothing in the caption could be confirmed, so this report says nothing about damage.</p>"
    )
    out = "".join(item(c, "✕", "no", reason(c, evidence)) for c in bad) + "".join(
        item(c, "–", "na", f"Cannot check: {short_reason(c, evidence).lower()}.") for c in nv
    )
    out_block = (
        (f"<h3>Left out of the report</h3><ul class='sr-list small'>{out}</ul>")
        if out
        else "<h3>Left out of the report</h3><p class='sr-empty'>Nothing. Every statement was confirmed.</p>"
    )
    adv = ADVISORIES.get(tile_id, {}).get("en")
    if ok and adv and NO_GUIDANCE_MARK in adv:
        advice = (
            "<h3>What residents should do</h3><p class='sr-empty'>No guidance passage was close enough to these "
            "statements, so no advice is given.</p>"
        )
    elif ok and adv:
        advice = advice_panel(adv, "en", "What residents should do") or (
            f"<h3>What residents should do</h3><pre class='adv'>{esc(adv)}</pre>"
        )
    elif ok:
        advice = (
            f"<h3>What residents should do</h3><p class='sr-empty'>Advice is not switched on in this version of the demo. "
            f"It would be written from the {len(ok)} confirmed statement{plural(len(ok))} above.</p>"
        )
    else:
        advice = "<h3>What residents should do</h3><p class='sr-empty'>No advice is given without a confirmed statement.</p>"
    he = XBD_TO_HUMAID.get(tile_event(tile_id))
    ev_stats = TWEET_STATS.get("events", {}).get(he)
    social = ""
    if ev_stats:
        share = ev_stats["flagged"] / max(ev_stats["n"], 1)
        social = (
            f"<h3>What people were reporting</h3><p>{ev_stats['n']:,} tweets from this event were scored, and "
            f"{share:.0%} were flagged as useful for responders. The chart is on the Public posts tab.</p>"
        )
        social += tweet_topics(ev_stats)
        ex = tweet_list(he, 3)
        if ex:
            social += "<h4 class='ex-h'>Examples flagged as useful</h4>" + ex
    elif he is None:
        social = "<h3>What people were reporting</h3><p class='sr-empty'>This event has no matching tweet set.</p>"
    sm = caption_summary(ok)
    sum_line = f'<p class="sr-sum"><b>In short:</b> {esc(sm)}</p>' if sm else ""
    flow = (
        f'<div class="sr-flow"><span class="st all"><b>{len(claims)}</b> statement{plural(len(claims))} in the caption</span>'
        f'<span class="ar">→</span><span class="st yes"><b>{len(ok)}</b> confirmed, kept</span>'
        f'<span class="st no"><b>{len(bad)}</b> wrong, removed</span>'
        f'<span class="st na"><b>{len(nv)}</b> cannot check, left out</span></div>'
    )
    return (
        f'<article class="sitrep" id="fv-report"><div class="sr-head"><h2>Verified Situation Report</h2>'
        f'<p class="sub">{esc(tile_title(tile_id))} · Written only from statements the evidence supports · '
        f"Evidence: {esc(SOURCES.get(evidence['source'], evidence['source']))}</p></div>{flow}"
        f"{sum_line}"
        f'<div class="sr-cols"><div><h3>Confirmed damage</h3>{keep_list}</div><div>{out_block}</div></div>'
        f'<div class="sr-more"><div>{advice}</div><div>{social}</div></div></article>'
    )


def explainer_html(n):
    """Which quantity words are possible when an area holds n buildings."""
    n = int(n)
    ok = reachable_words(n)
    chips = []
    for w in ("none", "few", "some", "many", "most"):
        if w in ok:
            chips.append(f'<span class="wchip yes"><b>✓</b> {w}</span>')
        else:
            chips.append(
                f'<span class="wchip no"><b>–</b> {w}<small>needs {min_buildings(w)}+ buildings</small></span>'
            )
    cannot = [w for w in ("few", "some", "many") if w not in ok]
    plural = "s" if n != 1 else ""
    msg = (
        f"With {n} building{plural} in an area, a caption cannot say {', '.join(f'“{w}”' for w in cannot)} "
        "truthfully or falsely: no whole number of buildings falls in that range, so the checker does not judge it."
        if cannot
        else f"With {n} buildings, every quantity word can be checked."
    )
    return f'<div class="wrow">{"".join(chips)}</div><p class="note">{msg}</p>'


def stats_html(tile_id):
    """Tweet volume per day and class mix for the place's event, from counts only."""
    he = XBD_TO_HUMAID.get(tile_event(tile_id))
    st = TWEET_STATS.get("events", {}).get(he)
    if he is None:
        return "<p class='note'>This place's event has no matching tweet set, so there is nothing to show.</p>"
    if not st:
        return (
            "<p class='note'>No tweet statistics have been added to the demo. Run notebook 08, "
            "<code>export_tweet_stats</code>, then <code>build_bundle.py --tweet-stats</code>.</p>"
        )
    days = sorted(st["daily"])
    mx = max(v[0] for v in st["daily"].values())
    cols = "".join(
        f'<div class="col">'
        f'<div class="tot" style="height:{st["daily"][d][0] / mx * 100:.1f}%">'
        f'<div class="flag" style="height:{st["daily"][d][1] / max(st["daily"][d][0], 1) * 100:.1f}%"></div></div></div>'
        for d in days
    )
    peak = max(days, key=lambda d: st["daily"][d][0])
    act = set(TWEET_STATS.get("actionable", []))
    cls_items = sorted(st["classes"].items(), key=lambda kv: -kv[1])[:8]
    cmx = max(v for _, v in cls_items)
    rows = "".join(
        f'<div class="crow"><span class="cn">{esc(k.replace("_", " "))}</span>'
        f'<span class="cb"><i class="{"act" if k in act else ""}" style="width:{v / cmx * 100:.1f}%"></i></span>'
        f'<span class="cv">{v:,}</span></div>'
        for k, v in cls_items
    )
    ex = tweet_list(he, 12)
    ex_block = (
        (
            f'<h3 class="ex-title">Example tweets flagged as useful</h3>{ex}'
            '<p class="note">The highest-scoring tweets of this event. Links and account names are removed.</p>'
        )
        if ex
        else ""
    )
    return (
        f'<div class="tw"><div><h3>Tweets per day</h3><div class="daily" role="img" '
        f'aria-label="Tweets per day from {days[0]} to {days[-1]}, peak on {peak} with {st["daily"][peak][0]} tweets">{cols}</div>'
        f'<div class="axis"><span>{days[0]}</span><span>peak {peak}</span><span>{days[-1]}</span></div>'
        f'<details class="daily-nums"><summary>Show the numbers for each day</summary><ul>'
        + "".join(
            f"<li>{d}: {st['daily'][d][0]:,} tweets, {st['daily'][d][1]:,} flagged useful</li>"
            for d in days
        )
        + "</ul></details>"
        f'<div class="legend"><span><i class="flagk"></i>flagged useful (score ≥ {TWEET_STATS.get("theta", 0.75):.2f})</span>'
        f'<span><i class="otherk"></i>other</span></div></div>'
        f"<div><h3>What the tweets are about</h3>{rows}</div></div>"
        f'<p class="note">{st["n"]:,} test-split tweets of {esc(he.replace("_", " "))}. '
        "Linked to the place by event and time, not by place, and not checked against the imagery.</p>"
        + ex_block
    )


def tile_signature(evidence):
    """(buildings, share of none/minor/major/destroyed) over the areas with usable imagery, or None."""
    cells = [
        c
        for c in evidence["cells"]
        if c.get("valid_fraction", 1.0) >= MIN_VALID and c["buildings"] > 0
    ]
    n = sum(c["buildings"] for c in cells)
    if not n:
        return None
    counts = [sum(round(c["buildings"] * c["class_fraction"][k]) for c in cells) for k in CLASSES]
    total = sum(counts) or 1
    return n, [x / total for x in counts]


SIGNATURES = {t: tile_signature(e) for t, e in TILES.items()}


def similar_tiles(tile_id, k=3):
    """The k other places whose damage mix is closest, with the overlap of the two mixes (100% = identical)."""
    mine = SIGNATURES.get(tile_id)
    if mine is None:
        return None, []
    scored = []
    for t, sig in SIGNATURES.items():
        if t == tile_id or sig is None:
            continue
        gap = 0.5 * sum(abs(a - b) for a, b in zip(mine[1], sig[1]))
        scored.append((gap, abs(math.log(mine[0] / sig[0])), t))
    return mine, [(t, 1 - g) for g, _, t in sorted(scored)[:k]]


def similar_html(tile_id):
    mine, items = similar_tiles(tile_id)
    if mine is None:
        return (
            "<p class='note'>This place has no buildings with usable imagery, so there is nothing to compare.</p>",
            [],
        )
    rows = []
    for rank, (t, overlap) in enumerate(items, 1):
        n, mix = SIGNATURES[t]
        seg = "".join(
            f'<i class="d-{k}" style="width:{mix[i] * 100:.1f}%"></i>'
            for i, k in enumerate(CLASSES)
            if i
        )
        if t in CAPTIONS:
            cl = verify(CAPTIONS[t], TILES[t])
            k_ = {v: sum(c["verdict"] == v for c in cl) for v in WORDS}
            said = (
                f"Its caption: {k_['supported']} match, {k_['contradicted']} do not match, "
                f"{k_['not_verifiable']} cannot be checked."
            )
        else:
            said = "No caption exists for this place."
        same = (
            ' <span class="tag">same event</span>' if tile_event(t) == tile_event(tile_id) else ""
        )
        rows.append(
            f'<li class="card sim"><div class="rank">{rank}</div><div class="body">'
            f"<div><b>{esc(tile_title(t))}</b>{same}</div>"
            f'<div class="bar" role="img" aria-label="damage mix">{seg}</div>'
            f'<div class="dmg">{n} buildings · {1 - mix[0]:.0%} damaged · damage mix overlap {overlap:.0%}</div>'
            f'<div class="r">{esc(said)}</div></div></li>'
        )
    if not rows:
        return "<p class='note'>No other place has usable buildings to compare with.</p>", []
    note = (
        f"This place: {mine[0]} buildings, {1 - mine[1][0]:.0%} damaged. Overlap is 100% minus half of the total "
        "gap between two places' shares of undamaged, minor, major and destroyed buildings. It compares damage "
        "patterns only, not places."
    )
    return f'<ul class="simlist">{"".join(rows)}</ul><p class="note">{note}</p>', [
        t for t, _ in items
    ]


def similar_view(tile_id):
    page, ids = similar_html(tile_id)
    btns = [
        gr.update(value=f"Open: {tile_title(ids[i])}", visible=True)
        if i < len(ids)
        else gr.update(visible=False)
        for i in range(3)
    ]
    return (page, ids, *btns)


ADVICE_OFF = {
    "en": "Advice is not switched on in this version of the demo.",
    "ar": "لم يتم تفعيل الإرشادات في هذه النسخة من العرض التوضيحي.",
}
ADVICE_NONE = {
    "en": "No statement was confirmed for this place, so no advice is shown.",
    # Arabic wording written for this demo: an Arabic reader should check it.
    "ar": "لم يتم تأكيد أي عبارة لهذه المنطقة، لذلك لا تُعرض أي نصيحة.",
}
NO_GUIDANCE_MARK = "No guidance found"


def advisory(tile_id, lang, confirmed=1):
    adv = ADVISORIES.get(tile_id, {}).get(lang)
    if not ADVISORIES:
        return ADVICE_OFF[lang]
    if not confirmed:  # the advice module falls back to general advice here; the demo only advises on confirmed statements
        return ADVICE_NONE[lang]
    return adv or ADVICE_NONE[lang]


_ADV_SRC = re.compile(r"\s*\[([A-Za-z]+)-([A-Za-z0-9]+)-(?:en|ar)\]\s*$")
_ADV_SPLIT = re.compile(r"(?<=[.!?\u061f])\s+")
_ADV_CELL = re.compile(r"^(r\dc\d):\s*(.*)$")


def parse_advisory(text):
    """Split the stored advisory text into (situation lines, advice items). Both are empty for a plain message."""
    situation, items, mode = [], [], None
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "SITUATION:":
            mode = "s"
        elif line == "ADVICE:":
            mode = "a"
        elif mode == "s":
            situation.append(line)
        elif mode == "a" and line.startswith("- "):
            items.append(line[2:].strip())
    return situation, items


def _advice_item(raw):
    """One guidance passage: its first sentence is the instruction, the rest is detail, the tag is its source."""
    src = ""
    m = _ADV_SRC.search(raw)
    if m:
        src, raw = f"{m.group(1)} {m.group(2)}", raw[: m.start()]
    parts = _ADV_SPLIT.split(raw.strip(), maxsplit=1)
    rest = f'<p class="adv-rest">{esc(parts[1])}</p>' if len(parts) == 2 else ""
    tag = f'<span class="adv-src" title="Guidance passage {esc(src)}">{esc(src)}</span>' if src else ""
    return f'<li><div><p class="adv-head">{esc(parts[0])}</p>{rest}</div>{tag}</li>'


def advice_panel(text, lang, title, with_basis=False, extra_cls=""):
    """The advice as a panel: a title bar, one entry per guidance passage, the passage ID as a small tag.
    Returns None when the text is a plain message (advice off, no guidance), so callers keep their own wording."""
    situation, items = parse_advisory(text)
    if not items:
        return None
    d = "rtl" if lang == "ar" else "ltr"
    basis = ""
    if with_basis and situation:
        rows = []
        for line in situation:
            m = _ADV_CELL.match(line)
            rows.append(f"<li><b>{esc(NAMES.get(m.group(1), m.group(1)))}</b> {esc(m.group(2))}</li>" if m else f"<li>{esc(line)}</li>")
        basis = f'<div class="adv-basis" dir="ltr" lang="en"><b>Written from</b><ul>{"".join(rows)}</ul></div>'
    n = len(items)
    return (
        f'<section class="advisory {extra_cls}" dir="{d}" lang="{lang}">'
        f'<header class="adv-bar"><h3>{esc(title)}</h3><span dir="ltr" lang="en">{n} guidance passage{"s" if n != 1 else ""}</span></header>'
        f'{basis}<ul class="adv-list">{"".join(_advice_item(i) for i in items)}</ul></section>'
    )


def advice_box(text, lang):
    """The advice for the tab: a panel when the text has guidance, otherwise the plain message in a box."""
    title, d = ("English", "ltr") if lang == "en" else ("\u0627\u0644\u0639\u0631\u0628\u064a\u0629", "rtl")
    panel = advice_panel(text, lang, title, with_basis=True, extra_cls="advbox")
    if panel:
        return panel
    return (
        f'<section class="advbox" dir="{d}" lang="{lang}" aria-label="{title}"><h3>{title}</h3>'
        f'<div class="advtext">{esc(text)}</div></section>'
    )


def advice_why_html(tile_id, claims):
    """Shown when a tile has no advice: a three-step checklist that says where it stopped."""
    ok = sum(c["verdict"] == "supported" for c in claims)
    bad = sum(c["verdict"] == "contradicted" for c in claims)
    nv = sum(c["verdict"] == "not_verifiable" for c in claims)
    adv = ADVISORIES.get(tile_id, {}).get("en", "")
    no_guidance = bool(ADVISORIES) and NO_GUIDANCE_MARK in adv
    if ok and adv and not no_guidance:
        return ""
    on = bool(ADVISORIES)
    steps = [
        (
            ok > 0,
            "A statement was confirmed",
            f"{ok} statement{'s' if ok != 1 else ''} matched the evidence."
            if ok
            else f"None. {bad} did not match and {nv} could not be checked, so there is nothing safe to advise on.",
        ),
        (
            on,
            "The advice step is switched on",
            "Yes."
            if on
            else "No. This version of the demo shows the checking only, so advice is not written.",
        ),
        (
            False,
            "Guidance fits these statements",
            "No guidance passage was close enough to these statements, so nothing is added."
            if (ok and on)
            else "Advice is only written from confirmed statements, using the guidance library.",
        ),
    ]
    items = "".join(
        f'<li class="{"yes" if done else "no"}"><span class="mk">{"✓" if done else "✕"}</span>'
        f"<div><b>{t}</b><small>{d}</small></div></li>"
        for done, t, d in steps
    )
    return (
        '<div class="why"><b>Why is there no advice for this place?</b>'
        "<p>Advice is written only from statements the evidence confirms. This is where it stopped:</p>"
        f'<ol class="chk">{items}</ol></div>'
    )


def evidence_rows(evidence):
    rows = []
    for c in evidence["cells"]:
        share = class_fraction([c], "damaged")
        rows.append(
            [
                NAMES.get(c["cell"], c["cell"]),
                c["buildings"],
                "-" if share is None else f"{share:.0%}",
                f"{c.get('valid_fraction', 1.0):.2f}",
                f"{c.get('confidence', 1.0):.2f}",
            ]
        )
    return rows


def kpi_html(evidence, claims):
    """Five headline numbers for the place and its caption."""
    usable = [c for c in evidence["cells"] if c.get("valid_fraction", 1.0) >= MIN_VALID]
    n = sum(c["buildings"] for c in usable)
    dam = class_fraction(usable, "damaged") if usable else None
    k = {v: sum(c["verdict"] == v for c in claims) for v in WORDS}
    tiles = [
        ("", f"{n:,}", "buildings with usable imagery"),
        ("", "–" if dam is None else f"{dam:.0%}", "of those are damaged"),
        ("ok", f"✓ {k['supported']}", "statements match the evidence"),
        ("bad", f"✕ {k['contradicted']}", "do not match the evidence"),
        ("na", f"– {k['not_verifiable']}", "cannot be checked"),
    ]
    return (
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi {c}"><b>{v}</b><span>{esc(t)}</span></div>' for c, v, t in tiles
        )
        + "</div>"
    )


# ---------- events ----------


def legend_html(has_imagery, evidence):
    swatch = "".join(f'<span class="lchip l{c}">{n}</span>' for c, (n, _) in LAYERS.items())
    verdicts = (
        '<span class="chip supported"><b>✓</b> Matches</span><span class="chip contradicted"><b>✕</b> Does not match</span>'
        '<span class="chip not_verifiable"><b>–</b> Cannot check (dashed grey)</span><span class="chip nomention">Not mentioned in the caption</span>'
    )
    src = SOURCES.get(evidence["source"], evidence["source"])
    if has_imagery:
        cap = (
            "Fig. 1 · Building colours come from the dataset's damage labels. The right panel marks areas where "
            f"the caption disagrees with the evidence ({esc(src)}). Dashed grey areas could not be checked, and the reason is written on them."
        )
    else:
        cap = (
            "No satellite imagery is bundled with this demo (xBD is licensed for non-commercial use). "
            "The panels show the building evidence for each area instead."
        )
    return (
        f'<div class="chips-row"><span class="lead">Buildings</span>{swatch}<span class="lead">Caption check</span>{verdicts}</div>'
        f'<p class="caption">{cap}</p>'
    )


def map_alts(evidence, claims, has_imagery, keep):
    """Short descriptions of the three maps for screen readers, with the numbers the pictures show."""
    cells = evidence["cells"]
    total = sum(c["buildings"] for c in cells)
    mix = {
        k: sum(c["buildings"] * c["class_fraction"].get(k, 0.0) for c in cells)
        for k in ("minor", "major", "destroyed")
    }
    undamaged = max(total - sum(mix.values()), 0)
    parts = [
        f"{round(undamaged)} undamaged",
        f"{round(mix['minor'])} minor",
        f"{round(mix['major'])} major",
        f"{round(mix['destroyed'])} destroyed",
    ]
    shown = f" Only {', '.join(k.lower() for k in keep)} buildings are shown." if keep else ""
    n = f"{total} building{'s' if total != 1 else ''}"
    if has_imagery:
        pre = "Before the flood: satellite view of the place, divided into a 3 by 3 grid of areas."
        post = f"After the flood: {n}, coloured by damage level ({', '.join(parts)}).{shown}"
    else:
        pre = "No satellite image is available for this place."
        post = f"Evidence map: {n} in the 3 by 3 areas of the place ({', '.join(parts)})."
    st = cell_states(claims)
    k = {v: sum(x == v for x in st.values()) for v in WORDS}
    verdict = (
        f"Where the caption is wrong: {k['supported']} area{'s' if k['supported'] != 1 else ''} match the evidence, "
        f"{k['contradicted']} do not match, {k['not_verifiable']} cannot be checked, and the other areas are not "
        "mentioned in the caption."
    )
    return pre, post, verdict


def render(tile_id, report_text, alpha, keep):
    ev = TILES[tile_id]
    claims = verify(report_text, ev)
    before, after, verdict, has = panels(tile_id, ev, claims, alpha, keep)
    pre_alt, post_alt, verdict_alt = map_alts(ev, claims, has, keep)
    return (
        gr.update(value=before, visible=before is not None, alt_text=pre_alt),
        gr.update(value=after, alt_text=post_alt),
        gr.update(value=verdict, alt_text=verdict_alt),
        legend_html(has, ev) + why_html(claims, ev),
        highlighted(report_text, claims),
        grid_html(ev, claims),
        issues_html(report_text) + summary_html(claims),
        cards_html(claims, ev),
        sitrep_html(tile_id, ev, claims),
        kpi_html(ev, claims),
        explore_html(tile_id, ev, claims, alpha, keep),
    )


def on_check(tile_id, report_text, alpha, keep):
    return render(tile_id, report_text, alpha, keep)


def on_tile(tile_id, alpha, keep):
    report = default_report(tile_id)
    note = (
        ""
        if tile_id in CAPTIONS
        else (
            "<div class='warn'><b>No caption exists for this place.</b> The text on the right was written from the "
            "evidence, so it will match. Replace it with a real caption to see a real result.</div>"
        )
    )
    claims0 = verify(report, TILES[tile_id])
    confirmed = sum(c["verdict"] == "supported" for c in claims0)
    return (
        note,
        report,
        *render(tile_id, report, alpha, keep),
        advice_why_html(tile_id, claims0),
        advice_box(advisory(tile_id, "en", confirmed), "en"),
        advice_box(advisory(tile_id, "ar", confirmed), "ar"),
        stats_html(tile_id),
        evidence_rows(TILES[tile_id]),
        *similar_view(tile_id),
    )


# ---------- interactive viewer ----------

import base64
import io

XP_JS = r"""
const S = element.__xp = element.__xp || {z: 1, x: 0, y: 0, layer: 'after', cell: null};
const q = (s) => element.querySelector(s);
function apply() {
  const pan = q('.xp-pan'); if (!pan) return;
  const st = q('.xp-stage'), w = st.clientWidth, h = st.clientHeight;
  S.z = Math.min(6, Math.max(1, S.z));
  S.x = Math.min(0, Math.max(w - w * S.z, S.x)); S.y = Math.min(0, Math.max(h - h * S.z, S.y));
  pan.style.transform = 'translate(' + S.x + 'px,' + S.y + 'px) scale(' + S.z + ')';
  st.classList.toggle('zoomed', S.z > 1);
  const zl = q('.xp-zl'); if (zl) zl.textContent = Math.round(S.z * 100) + '%';
  let layer = S.layer; if (!q('.xp-img[data-layer="' + layer + '"]')) layer = S.layer = 'after';
  element.querySelectorAll('.xp-img').forEach(i => i.classList.toggle('on', i.dataset.layer === layer));
  element.querySelectorAll('.xp-tab').forEach(b => b.setAttribute('aria-pressed', b.dataset.layer === layer ? 'true' : 'false'));
  element.querySelectorAll('.xp-cell').forEach(c => c.classList.toggle('sel', c.dataset.cell === S.cell));
  const d = q('.xp-detail'), src = S.cell && q('.xp-cell[data-cell="' + S.cell + '"] .xp-tip');
  if (d) d.innerHTML = src ? src.innerHTML : d.dataset.hint;
}
function zoomAt(f, cx, cy) {
  const nz = Math.min(6, Math.max(1, S.z * f)), r = nz / S.z;
  S.x = cx - (cx - S.x) * r; S.y = cy - (cy - S.y) * r; S.z = nz; apply();
}
function stageCentre() { const st = q('.xp-stage'); return [st.clientWidth / 2, st.clientHeight / 2]; }
element.addEventListener('click', (e) => {
  const b = e.target.closest('.xp-btn'); 
  if (b) {
    const a = b.dataset.act, [cx, cy] = stageCentre();
    if (a === 'in') zoomAt(1.5, cx, cy); else if (a === 'out') zoomAt(1 / 1.5, cx, cy);
    else if (a === 'reset') { S.z = 1; S.x = 0; S.y = 0; apply(); }
    return;
  }
  const t = e.target.closest('.xp-tab'); if (t) { S.layer = t.dataset.layer; apply(); return; }
  if (S.moved) { S.moved = false; return; }
  const c = e.target.closest('.xp-cell'); if (c) { S.cell = S.cell === c.dataset.cell ? null : c.dataset.cell; apply(); }
});
element.addEventListener('keydown', (e) => {
  const c = e.target.closest('.xp-cell');
  if (c && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); S.cell = S.cell === c.dataset.cell ? null : c.dataset.cell; apply(); }
});
element.addEventListener('mouseover', (e) => {
  const c = e.target.closest('.xp-cell'); if (c && !S.dragging && S.cell !== c.dataset.cell && !S.pinned) { S.hover = c.dataset.cell; }
});
element.addEventListener('wheel', (e) => {
  if (!e.target.closest('.xp-stage') || !(e.ctrlKey || e.metaKey)) return;
  e.preventDefault(); const r = q('.xp-stage').getBoundingClientRect();
  zoomAt(e.deltaY < 0 ? 1.2 : 1 / 1.2, e.clientX - r.left, e.clientY - r.top);
}, {passive: false});
let start = null;
element.addEventListener('pointerdown', (e) => {
  const st = e.target.closest('.xp-stage'); if (!st || S.z <= 1) return;
  start = {px: e.clientX, py: e.clientY, x: S.x, y: S.y}; S.moved = false; st.setPointerCapture(e.pointerId);
});
element.addEventListener('pointermove', (e) => {
  if (!start) return;
  const dx = e.clientX - start.px, dy = e.clientY - start.py;
  if (Math.abs(dx) + Math.abs(dy) > 4) S.moved = true;
  S.x = start.x + dx; S.y = start.y + dy; apply();
});
const end = () => { start = null; };
element.addEventListener('pointerup', end); element.addEventListener('pointercancel', end);
new MutationObserver(apply).observe(element, {childList: true, subtree: false});
apply();
"""


def _data_uri(img, size=900):
    img = img.convert("RGB")
    if max(img.size) > size:
        img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _cell_tip(cid, c, stmts):
    name = f"<b>{NAMES[cid]}</b> <small>{cid}</small>"
    if c is None:
        return f"{name}<p>No data for this area.</p>"
    n, valid, f = c["buildings"], c.get("valid_fraction", 1.0), c["class_fraction"]
    if valid < MIN_VALID:
        body = "<p>No usable imagery in this area.</p>"
    elif n == 0:
        body = "<p>No buildings in this area.</p>"
    else:
        mix = " · ".join(f"{DMG_LABEL[k]} {f[k]:.0%}" for k in DMG_KEYS if f.get(k, 0) > 0)
        body = (
            f"<p>{n} building{'s' if n != 1 else ''}, {class_fraction([c], 'damaged'):.0%} damaged.</p>"
            f"<p class='mix'>{esc(mix)}</p>"
        )
    rows = "".join(
        f"<li><span class='v {s['verdict']}'>{WORDS[s['verdict']][1]}</span> "
        f"{esc(s['quantity'])} · {esc(s['damage_class'])}</li>"
        for s in stmts
    )
    return (
        name
        + body
        + (
            f"<ul>{rows}</ul>"
            if rows
            else "<p class='mix'>The caption does not mention this area.</p>"
        )
    )


def explore_html(tile_id, evidence, claims, alpha, keep):
    """Zoomable, pannable view of the place. Tap an area to read its counts and verdicts."""
    before, after, verdict, has = panels(tile_id, evidence, claims, alpha, keep)
    layers = []
    if before is not None:
        layers.append(("before", "Before", before))
    layers += [("after", "After, by damage", after), ("verdict", "Caption check", verdict)]
    imgs = "".join(
        f'<img class="xp-img" data-layer="{k}" src="{_data_uri(im)}" alt="" draggable="false">'
        for k, _, im in layers
    )
    tabs = "".join(
        f'<button type="button" class="xp-tab" data-layer="{k}" aria-pressed="false">{lab}</button>'
        for k, lab, _ in layers
    )
    by_cell = {c["cell"]: c for c in evidence["cells"]}
    stmts = {}
    for cl in claims:
        stmts.setdefault(cl["cell"], []).append(cl)
    cells = []
    for r in range(3):
        for k in range(3):
            cid = f"r{r}c{k}"
            vs = {s["verdict"] for s in stmts.get(cid, [])}
            state = (
                "contradicted"
                if "contradicted" in vs
                else "supported"
                if "supported" in vs
                else "not_verifiable"
                if vs
                else "none"
            )
            label = f"{NAMES[cid]}" + (f", {WORDS[state][1].lower()}" if state != "none" else "")
            cells.append(
                f'<div class="xp-cell s-{state}" data-cell="{cid}" role="button" tabindex="0" '
                f'aria-label="{esc(label)}" style="left:{k * 100 / 3:.4f}%;top:{r * 100 / 3:.4f}%">'
                f'<span class="xp-n">{NAMES[cid]}</span>'
                f'<div class="xp-tip">{_cell_tip(cid, by_cell.get(cid), stmts.get(cid, []))}</div></div>'
            )
    note = (
        ""
        if has
        else '<p class="note">No photo is bundled for this place, so the areas are drawn without one.</p>'
    )
    return (
        '<div class="xp">'
        f'<div class="xp-bar"><div class="xp-tabs" role="group" aria-label="Map layer">{tabs}</div>'
        '<div class="xp-zoom" role="group" aria-label="Zoom">'
        '<button type="button" class="xp-btn" data-act="out" aria-label="Zoom out">−</button>'
        '<span class="xp-zl" aria-live="polite">100%</span>'
        '<button type="button" class="xp-btn" data-act="in" aria-label="Zoom in">+</button>'
        '<button type="button" class="xp-btn" data-act="reset">Reset</button></div></div>'
        f'<div class="xp-stage"><div class="xp-pan">{imgs}{"".join(cells)}</div></div>'
        '<div class="xp-detail" aria-live="polite" data-hint="Tap or click an area to see its counts and the checks on it. '
        'Zoom with the buttons, or hold Ctrl or ⌘ and scroll; drag to move when zoomed in."></div>'
        f"{note}</div>"
    )


XP_CSS = """
.xp{display:flex;flex-direction:column;gap:12px}
.xp-bar{display:flex;flex-wrap:wrap;gap:12px;justify-content:space-between;align-items:center}
.xp-tabs,.xp-zoom{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.xp-tab,.xp-btn{font:600 .95rem 'Inter',system-ui,sans-serif;color:var(--ink);background:var(--card-solid);border:1px solid var(--line);border-radius:999px;padding:8px 16px;min-height:40px;cursor:pointer}
.xp-tab[aria-pressed=true]{background:var(--accent);color:var(--on-accent);border-color:var(--accent)}
.xp-btn{min-width:40px;padding:8px 12px}
.xp-zl{min-width:3.2rem;text-align:center;color:var(--muted);font-size:.9rem}
.xp-stage{position:relative;width:100%;max-width:760px;margin:0 auto;aspect-ratio:1/1;overflow:hidden;border-radius:var(--r-lg);border:1px solid var(--line);background:#14202c;touch-action:pan-y;user-select:none}
.xp-stage.zoomed{cursor:grab;touch-action:none}
.xp-pan{position:absolute;inset:0;transform-origin:0 0}
.xp-img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none}
.xp-img.on{display:block}
.xp-cell{position:absolute;width:33.3334%;height:33.3334%;box-sizing:border-box;border:1px dashed rgba(255,255,255,.35);cursor:pointer}
.xp-cell:hover,.xp-cell.sel,.xp-cell:focus-visible{background:rgba(255,255,255,.16);border:2px solid #fff;outline:none}
.xp-cell.s-supported{box-shadow:inset 0 0 0 3px #1f7a5a}.xp-cell.s-contradicted{box-shadow:inset 0 0 0 3px #b23a1f}.xp-cell.s-not_verifiable{box-shadow:inset 0 0 0 3px #7c8c98}
.xp-n{position:absolute;left:6px;top:4px;font:600 .8rem 'Inter',system-ui,sans-serif;color:#fff;background:rgba(10,20,30,.7);padding:2px 8px;border-radius:999px;pointer-events:none}
.xp-tip{display:none}
.xp-detail{background:var(--card-solid);border:1px solid var(--line);border-radius:var(--r);padding:14px 18px;color:var(--ink);min-height:84px}
.xp-detail p{margin:.2rem 0}.xp-detail .mix{color:var(--muted)}
.xp-detail ul{margin:.4rem 0 0;padding-left:1.1rem}
.xp-detail .v{font-weight:700}.xp-detail .v.supported{color:#1f7a5a}.xp-detail .v.contradicted{color:#b23a1f}.xp-detail .v.not_verifiable{color:var(--muted)}
.xp-detail[data-hint]:empty::before{content:attr(data-hint);color:var(--muted)}
"""


# ---------- globe ----------

LOCS = _json(
    DATA / "locations.json"
)  # {tile_id: [lat, lon]}, written by build_bundle.py when the xBD folder is given
LAND = json.loads((ROOT / "land.json").read_text()) if (ROOT / "land.json").exists() else []
# Rough centre of each xBD flood event, used when a place has no coordinates of its own.
EVENT_LOC = {
    "hurricane-florence": (34.7, -77.4),
    "hurricane-harvey": (29.8, -95.4),
    "hurricane-matthew": (18.6, -74.1),
    "midwest-flooding": (41.3, -96.0),
    "nepal-flooding": (27.0, 85.5),
}


def globe_value(tile_id=None):
    """JSON for the globe: one dot per event, plus the places of the selected event when they have coordinates."""
    sel = tile_event(tile_id) if tile_id else None
    events = []
    for e in sorted({tile_event(t) for t in TILES}):
        ids = tiles_of(e)
        pts = [(t, LOCS[t]) for t in ids if t in LOCS]
        if pts:
            lat, lon = sum(p[1][0] for p in pts) / len(pts), sum(p[1][1] for p in pts) / len(pts)
        elif e in EVENT_LOC:
            lat, lon = EVENT_LOC[e]
        else:
            lat = lon = None
        events.append(
            {
                "id": e,
                "label": event_name(e),
                "n": len(ids),
                "lat": lat,
                "lon": lon,
                "exact": bool(pts),
                "places": [
                    {"id": t, "i": ids.index(t) + 1, "lat": ll[0], "lon": ll[1]} for t, ll in pts
                ]
                if e == sel
                else [],
            }
        )
    return html.escape(json.dumps({"sel": sel, "tile": tile_id, "events": events}), quote=False)


def pick_target(value):
    """What the globe sent back: 'event:<id>' or 'tile:<id>'. Returns the tile to show."""
    kind, _, key = (value or "").partition(":")
    if kind == "tile" and key in TILES:
        return key, ""
    if kind == "event" and default_tile(key):
        return default_tile(key), ""
    return gr.skip(), ""


GL_JS = r"""
const G = element.__gl = element.__gl || {lon: -40, lat: 20, s: 1, anim: null, hover: null};
const q = (s) => element.querySelector(s);
const css = (n, f) => (getComputedStyle(q('.gl') || element).getPropertyValue(n).trim() || f);
let D = null, dots = [];
const rad = Math.PI / 180;
function load() { const t = q('.gl-data'); try { D = JSON.parse(t.value || t.textContent); } catch (e) { D = null; } }
function proj(lon, lat, R, cx, cy) {
  const l = (lon - G.lon) * rad, p = lat * rad, p0 = G.lat * rad;
  const c = Math.sin(p0) * Math.sin(p) + Math.cos(p0) * Math.cos(p) * Math.cos(l);
  return [cx + R * Math.cos(p) * Math.sin(l), cy - R * (Math.cos(p0) * Math.sin(p) - Math.sin(p0) * Math.cos(p) * Math.cos(l)), c];
}
function draw() {
  const cv = q('canvas'); if (!cv || !D) return;
  const w = Math.min(520, element.clientWidth || 520), dpr = window.devicePixelRatio || 1;
  if (w < 60) return;
  cv.style.width = w + 'px'; cv.style.height = w + 'px';
  if (cv.width !== Math.round(w * dpr)) { cv.width = cv.height = Math.round(w * dpr); }
  const x = cv.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, w, w);
  const R = (w / 2 - 6) * G.s, cx = w / 2, cy = w / 2;
  const sea = css('--gl-sea', '#cfe0ee'), land = css('--gl-land', '#f3ead6'), line = css('--line', '#cdbf9f'),
        acc = css('--accent', '#0a62a3'), ink = css('--ink', '#14263a'), hot = css('--gl-hot', '#b5400e');
  const g = x.createRadialGradient(cx - R * .3, cy - R * .35, R * .1, cx, cy, R);
  g.addColorStop(0, sea); g.addColorStop(1, css('--gl-sea2', '#a9c4da'));
  x.beginPath(); x.arc(cx, cy, R, 0, 7); x.fillStyle = g; x.fill();
  x.save(); x.beginPath(); x.arc(cx, cy, R, 0, 7); x.clip();
  x.strokeStyle = 'rgba(255,255,255,.35)'; x.lineWidth = 1;
  for (let lo = -180; lo < 180; lo += 30) { x.beginPath(); let pen = false;
    for (let la = -90; la <= 90; la += 5) { const p = proj(lo, la, R, cx, cy); if (p[2] > 0) { pen ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]); pen = true; } else pen = false; } x.stroke(); }
  for (let la = -60; la <= 60; la += 30) { x.beginPath(); let pen = false;
    for (let lo = -180; lo <= 180; lo += 5) { const p = proj(lo, la, R, cx, cy); if (p[2] > 0) { pen ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]); pen = true; } else pen = false; } x.stroke(); }
  x.fillStyle = land; x.strokeStyle = line; x.lineWidth = .8;
  for (const poly of LAND) for (const ring of poly) {
    x.beginPath(); let any = false;
    ring.forEach((pt, i) => { const p = proj(pt[0], pt[1], R, cx, cy); let X = p[0], Y = p[1];
      if (p[2] > 0) any = true; else { const dx = X - cx, dy = Y - cy, d = Math.hypot(dx, dy) || 1; X = cx + dx / d * R; Y = cy + dy / d * R; }
      i ? x.lineTo(X, Y) : x.moveTo(X, Y); });
    if (any) { x.closePath(); x.fill(); x.stroke(); }
  }
  x.restore();
  x.beginPath(); x.arc(cx, cy, R, 0, 7); x.strokeStyle = line; x.lineWidth = 1.5; x.stroke();
  dots = [];
  const put = (id, kind, lat, lon, r, col, label, sel) => {
    const p = proj(lon, lat, R, cx, cy); if (p[2] <= 0.05) return;
    dots.push({id, kind, x: p[0], y: p[1], label});
    if (sel) { x.beginPath(); x.arc(p[0], p[1], r + 7, 0, 7); x.strokeStyle = col; x.globalAlpha = .45; x.lineWidth = 3; x.stroke(); x.globalAlpha = 1; }
    x.beginPath(); x.arc(p[0], p[1], r, 0, 7); x.fillStyle = col; x.fill(); x.lineWidth = 2; x.strokeStyle = '#fff'; x.stroke();
  };
  for (const e of D.events) if (e.lat != null) {
    const isSel = e.id === D.sel;
    put(e.id, 'event', e.lat, e.lon, isSel ? 8 : 6.5, isSel ? hot : acc, e.label + ' · ' + e.n + ' place' + (e.n === 1 ? '' : 's'), isSel);
    for (const pl of e.places) put(pl.id, 'tile', pl.lat, pl.lon, pl.id === D.tile ? 6 : 4.5, pl.id === D.tile ? hot : ink, e.label + ', place ' + pl.i, pl.id === D.tile);
  }
  if (G.hover) { const d = dots.find(d => d.id === G.hover); if (d) {
    x.font = '600 14px Inter, system-ui, sans-serif'; const tw = x.measureText(d.label).width + 16;
    const bx = Math.min(w - tw - 4, Math.max(4, d.x - tw / 2)), by = d.y < 40 ? d.y + 14 : d.y - 38;
    x.fillStyle = '#0f2438'; x.beginPath(); x.roundRect(bx, by, tw, 26, 13); x.fill();
    x.fillStyle = '#fff'; x.fillText(d.label, bx + 8, by + 18); } }
}
function caption() {
  const c = q('.gl-cap'); if (!c || !D) return;
  const e = D.events.find(e => e.id === D.sel);
  c.textContent = e ? e.label + ': ' + e.n + ' place' + (e.n === 1 ? '' : 's') + '. ' + (e.exact ? 'Dots mark the places.' : 'The dot marks the event’s approximate centre.')
                    : 'Click a dot to choose a flood.';
}
function spinTo(lon, lat, scale) {
  const s = {lon: G.lon, lat: G.lat, k: G.s}; let d = lon - s.lon; d = ((d + 540) % 360) - 180;
  const t0 = performance.now(), T = 800; cancelAnimationFrame(G.anim);
  const step = (t) => { const k = Math.max(0, Math.min(1, (t - t0) / T)), e = 1 - Math.pow(1 - k, 3);
    G.lon = s.lon + d * e; G.lat = s.lat + (Math.max(-80, Math.min(80, lat)) - s.lat) * e; G.s = s.k + (scale - s.k) * e; draw();
    if (k < 1) G.anim = requestAnimationFrame(step); };
  G.anim = requestAnimationFrame(step);
}
function fitScale(e) {
  if (!e.places.length) return 1;
  const la = e.places.map(p => p.lat), lo = e.places.map(p => p.lon);
  const sp = Math.max(Math.max(...la) - Math.min(...la), (Math.max(...lo) - Math.min(...lo)) * Math.cos(e.lat * rad), .05);
  return Math.max(1, Math.min(3, 70 / (4.2 * sp)));   // capped: the land outlines are coarse, so deeper zoom shows only a flat colour
}
function pick(v) {
  const ta = document.querySelector('#fv-pick textarea, #fv-pick input'); if (!ta) return;
  ta.value = v; ta.dispatchEvent(new Event('input', {bubbles: true}));
}
function refresh() {
  load(); caption(); draw();
  const list = q('.gl-list');
  if (list && D) list.innerHTML = D.events.map(e => '<button type="button" class="gl-chip" data-event="' + e.id + '" aria-pressed="false">' + e.label + ' · ' + e.n + '</button>').join('');
  const key = D && D.sel;
  if (key && key !== G.last) { const e = D.events.find(e => e.id === key);
    if (e && e.lat != null) {
      const first = G.last === undefined;
      spinTo(e.lon, first ? e.lat * .5 : e.lat, first ? 1 : fitScale(e)); } }
  G.last = key;
  element.querySelectorAll('.gl-chip').forEach(b => b.setAttribute('aria-pressed', D && b.dataset.event === D.sel ? 'true' : 'false'));
}
function hit(e) {
  const r = q('canvas').getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  let best = null, bd = 16;
  for (const d of dots) { const k = Math.hypot(d.x - px, d.y - py) - (d.kind === 'tile' ? 2 : 0); if (k < bd) { bd = k; best = d; } }
  return best;
}
let drag = null;
element.addEventListener('pointerdown', (e) => { const cv = e.target.closest('canvas'); if (!cv) return;
  drag = {x: e.clientX, y: e.clientY, lon: G.lon, lat: G.lat, moved: false}; cv.setPointerCapture(e.pointerId); cancelAnimationFrame(G.anim); });
element.addEventListener('pointermove', (e) => {
  if (drag) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
    if (drag.moved) { G.lon = drag.lon - dx * .4; G.lat = Math.max(-80, Math.min(80, drag.lat + dy * .4)); draw(); } return; }
  if (!e.target.closest('canvas')) return;
  const h = hit(e), id = h ? h.id : null; if (id !== G.hover) { G.hover = id; draw(); q('canvas').style.cursor = id ? 'pointer' : 'grab'; }
});
element.addEventListener('pointerup', (e) => { if (!drag) return; const moved = drag.moved; drag = null;
  if (!moved) { const h = hit(e); if (h) pick(h.kind + ':' + h.id); } });
element.addEventListener('pointerleave', () => { if (G.hover) { G.hover = null; draw(); } });
element.addEventListener('click', (e) => { const b = e.target.closest('.gl-chip'); if (b) pick('event:' + b.dataset.event);
  const z = e.target.closest('.gl-btn'); if (!z || !D) return;
  const ev = D.events.find(v => v.id === D.sel);
  if (z.dataset.zoom === 'out') spinTo(G.lon, G.lat * .5, 1);
  else if (ev && ev.lat != null) spinTo(ev.lon, ev.lat, fitScale(ev)); });
new MutationObserver(() => { const t = q('.gl-data'), raw = t && (t.value || t.textContent);
  if (raw !== G.raw || !q('.gl-chip')) { G.raw = raw; refresh(); } }).observe(element, {childList: true, subtree: true, characterData: true});
new ResizeObserver(draw).observe(element);
new MutationObserver(draw).observe(document.body, {attributes: true, attributeFilter: ['class']});
new MutationObserver(draw).observe(document.documentElement, {attributes: true, attributeFilter: ['data-fv']});
refresh();
"""

GL_JS_FULL = "const LAND = " + json.dumps(LAND, separators=(",", ":")) + ";\n" + GL_JS

GL_TEMPLATE = (
    '<div class="gl"><canvas role="img" aria-label="Globe with the flood events marked. Use the buttons below to choose one."></canvas>'
    '<div class="gl-zoom" role="group" aria-label="Globe zoom"><button type="button" class="gl-btn" data-zoom="out">Zoom out</button><button type="button" class="gl-btn" data-zoom="in">Zoom to places</button></div>'
    '<p class="gl-cap note" aria-live="polite"></p><div class="gl-list" role="group" aria-label="Flood events"></div>'
    '<textarea class="gl-data" hidden>${value}</textarea></div>'
)

GL_CSS = """
.gl{display:flex;flex-direction:column;align-items:center;gap:10px;--gl-sea:#d4e4f1;--gl-sea2:#a8c3da;--gl-land:#f6eedb;--gl-hot:#b5400e}
.dark .gl,.gradio-container.dark .gl{--gl-sea:#27465f;--gl-sea2:#16293b;--gl-land:#46586b;--gl-hot:#ff9a5a}
.gl canvas{display:block;max-width:100%;touch-action:pan-y;cursor:grab}
.gl-cap{margin:0;text-align:center}
.gl-list{display:flex;flex-wrap:wrap;gap:8px;justify-content:center}
.gl-zoom{display:flex;gap:8px;justify-content:center;flex-wrap:wrap}
.gl-btn,.gl-chip{font:600 .95rem 'Inter',system-ui,sans-serif;color:var(--ink);background:var(--card-solid);border:1px solid var(--line);border-radius:999px;padding:8px 16px;min-height:40px;cursor:pointer}
.gl-zoom{align-items:center}
.gl-zoom .gl-btn{flex:0 0 auto;box-sizing:border-box;min-width:10rem;height:40px;margin:0;padding:0 18px;display:inline-flex;align-items:center;justify-content:center;line-height:1;white-space:nowrap;transform:none}
.gl-chip[aria-pressed=true]{background:var(--accent);color:var(--on-accent);border-color:var(--accent)}
.fv-offscreen{position:absolute!important;left:-9999px!important;width:1px!important;height:1px!important;overflow:hidden!important}
"""

# ---------- look ----------

# ---------- dashboard ----------

from collections import Counter

DMG_KEYS = ("none", "minor", "major", "destroyed")
DMG_LABEL = {"none": "Undamaged", "minor": "Minor", "major": "Major", "destroyed": "Destroyed"}
DMG_RGB = {
    "none": LAYERS[1][1],
    "minor": LAYERS[2][1],
    "major": LAYERS[3][1],
    "destroyed": LAYERS[4][1],
}
WORD_ORDER = ("none", "few", "some", "many", "most")
WHY_LABEL = {
    "imagery": "Too little usable imagery in the area",
    "empty": "No buildings in the area",
    "few": "Too few buildings to judge the word used",
    "other": "The statement could not be read as a count",
}


def _why(claim, evidence):
    """Short code for why a statement could not be checked (mirrors reason())."""
    cell = next((c for c in evidence["cells"] if c["cell"] == claim.get("cell")), None)
    if cell is None or cell.get("valid_fraction", 1.0) < MIN_VALID:
        return "imagery"
    n, q = cell["buildings"], claim.get("quantity")
    if n == 0:
        return "empty"
    if q in ("few", "some", "many") and n < (min_buildings(q) or 0):
        return "few"
    return "other"


def _event_stats():
    """Counts per event from the evidence, the captions and the tweet statistics."""
    out = {}
    for e in sorted({tile_event(t) for t in TILES}):
        d = {
            "places": 0,
            "dmg": dict.fromkeys(DMG_KEYS, 0),
            "caps": 0,
            "verd": Counter(),
            "byq": {},
            "byc": {},
            "why": Counter(),
            "img": 0,
            "adv": 0,
        }
        for t in tiles_of(e):
            ev = TILES[t]
            d["places"] += 1
            for c in ev["cells"]:
                if c.get("valid_fraction", 1.0) >= MIN_VALID:
                    for k in DMG_KEYS:
                        d["dmg"][k] += round(c["buildings"] * c["class_fraction"][k])
            d["img"] += (IMG_DIR / f"{t}_post.jpg").exists()
            d["adv"] += bool(ADVISORIES.get(t))
            text = CAPTIONS.get(t)
            if isinstance(text, str) and text.strip():
                d["caps"] += 1
                for cl in verify(text, ev):
                    v = cl["verdict"]
                    d["verd"][v] += 1
                    d["byq"].setdefault(cl.get("quantity") or "unclear", Counter())[v] += 1
                    d["byc"].setdefault(cl.get("damage_class") or "unclear", Counter())[v] += 1
                    if v == "not_verifiable":
                        d["why"][_why(cl, ev)] += 1
        d["buildings"] = sum(d["dmg"].values())
        he = XBD_TO_HUMAID.get(e)
        d["tw"] = TWEET_STATS.get("events", {}).get(he) if he else None
        d["he"] = he
        d["tw_ex"] = len(TWEET_EXAMPLES.get(he) or []) if he else 0
        out[e] = d
    return out


DASH = _event_stats()


def _rgb(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


def _key(items):
    return (
        "<div class='dkey'>"
        + "".join(f"<span><i style='background:{col}'></i>{esc(lbl)}</span>" for lbl, col in items)
        + "</div>"
    )


def _sbar(parts, label):
    """Stacked bar. parts = [(value, css colour)]. The text label carries the same numbers for screen readers."""
    tot = sum(v for v, _ in parts)
    if not tot:
        return "<div class='sbar empty' role='img' aria-label='" + esc(label) + "'></div>"
    segs = "".join(
        f"<i style='width:{v / tot * 100:.2f}%;background:{col}'></i>" for v, col in parts if v
    )
    return f"<div class='sbar' role='img' aria-label='{esc(label)}'>{segs}</div>"


def _pct(a, b):
    return f"{a / b:.0%}" if b else "–"


def dash_kpis():
    tot_b = sum(d["buildings"] for d in DASH.values())
    dam = sum(d["dmg"][k] for d in DASH.values() for k in DMG_KEYS[1:])
    stm = sum(sum(d["verd"].values()) for d in DASH.values())
    ok = sum(d["verd"]["supported"] for d in DASH.values())
    bad = sum(d["verd"]["contradicted"] for d in DASH.values())
    chk, nv = ok + bad, stm - ok - bad
    tw = sum((d["tw"] or {}).get("n", 0) for d in DASH.values())
    items = [
        ("", f"{len(DASH)}", "flood events"),
        ("", f"{sum(d['places'] for d in DASH.values()):,}", "places checked"),
        ("", f"{tot_b:,}", "buildings with usable imagery"),
        ("", _pct(ok, chk), f"of {chk:,} checkable caption statements match"),
        (
            "na",
            _pct(nv, stm),
            f"of {stm:,} statements cannot be checked from the building evidence",
        ),
        ("na", f"{tw:,}", "public posts analysed"),
    ]
    return (
        '<div class="kpis kpis6">'
        + "".join(
            f'<div class="kpi {c}"><b>{esc(v)}</b><span>{esc(t)}</span></div>' for c, v, t in items
        )
        + "</div>"
        + (
            f"<p class='note'>{_pct(dam, tot_b)} of the buildings are damaged (minor, major or destroyed).</p>"
            if tot_b
            else ""
        )
    )


def dash_overview():
    rows = []
    for e, d in DASH.items():
        tw = d["tw"]
        rows.append(
            f"<tr><th scope='row'>{esc(event_name(e))}</th><td>{d['places']:,}</td><td>{d['buildings']:,}</td>"
            f"<td>{_pct(sum(d['dmg'][k] for k in DMG_KEYS[1:]), d['buildings'])}</td><td>{d['caps']:,}</td>"
            f"<td>{tw['n']:,}</td><td>{_pct(tw['flagged'], tw['n'])}</td></tr>"
            if tw
            else f"<tr><th scope='row'>{esc(event_name(e))}</th><td>{d['places']:,}</td><td>{d['buildings']:,}</td>"
            f"<td>{_pct(sum(d['dmg'][k] for k in DMG_KEYS[1:]), d['buildings'])}</td><td>{d['caps']:,}</td>"
            "<td>–</td><td>–</td></tr>"
        )
    head = (
        "<tr><th scope='col'>Event</th><th scope='col'>Places</th><th scope='col'>Buildings</th>"
        "<th scope='col'>Damaged</th><th scope='col'>Captions</th><th scope='col'>Posts</th><th scope='col'>Flagged useful</th></tr>"
    )
    return f"<div class='dscroll'><table class='dtable'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"


def dash_damage():
    key = _key([(DMG_LABEL[k], _rgb(DMG_RGB[k])) for k in DMG_KEYS])
    rows = []
    for e, d in DASH.items():
        parts = [(d["dmg"][k], _rgb(DMG_RGB[k])) for k in DMG_KEYS]
        said = ", ".join(f"{DMG_LABEL[k].lower()} {d['dmg'][k]:,}" for k in DMG_KEYS)
        rows.append(
            f"<div class='drow'><span class='dn-name'>{esc(event_name(e))}</span>"
            f"{_sbar(parts, f'{event_name(e)}: {said}')}<span class='dn'>{d['buildings']:,} buildings</span></div>"
        )
    return (
        f"<div class='dcard'>{key}{''.join(rows)}"
        "<p class='note'>Only buildings in areas with usable imagery are counted. The bar is the share of each damage level.</p></div>"
    )


def dash_checks():
    cols = {
        "supported": _rgb(STATE_RGB["supported"]),
        "contradicted": _rgb(STATE_RGB["contradicted"]),
        "not_verifiable": _rgb(STATE_RGB["not_verifiable"]),
    }
    key = _key([(WORDS[v][1], cols[v]) for v in WORDS])
    rows = []
    for e, d in DASH.items():
        v = d["verd"]
        tot = sum(v.values())
        if not tot:
            rows.append(
                f"<div class='drow'><span class='dn-name'>{esc(event_name(e))}</span>"
                "<span class='note'>No captions added for this event.</span><span class='dn'></span></div>"
            )
            continue
        said = f"matches {v['supported']}, does not match {v['contradicted']}, cannot check {v['not_verifiable']}"
        rows.append(
            f"<div class='drow'><span class='dn-name'>{esc(event_name(e))}</span>"
            f"{_sbar([(v[k], cols[k]) for k in WORDS], f'{event_name(e)}: {said}')}"
            f"<span class='dn'>{tot:,} statements</span></div>"
        )
    return (
        f"<div class='dcard'>{key}{''.join(rows)}"
        "<p class='note'>Every statement in a caption is checked against the building evidence. "
        '"Cannot check" means the evidence is too thin to answer, not that the statement is wrong.</p></div>'
    )


def _fail_rows(groups, order, label_of):
    cols = {
        "supported": _rgb(STATE_RGB["supported"]),
        "contradicted": _rgb(STATE_RGB["contradicted"]),
        "not_verifiable": _rgb(STATE_RGB["not_verifiable"]),
    }
    rows = []
    for g in order:
        c = groups.get(g)
        if not c:
            continue
        tot = sum(c.values())
        said = f"matches {c['supported']}, does not match {c['contradicted']}, cannot check {c['not_verifiable']}"
        rows.append(
            f"<div class='drow'><span class='dn-name'>{esc(label_of(g))}</span>"
            f"{_sbar([(c[k], cols[k]) for k in WORDS], f'{label_of(g)}: {said}')}"
            f"<span class='dn'>{c['supported']}/{c['contradicted']}/{c['not_verifiable']}</span></div>"
        )
    return "".join(rows) or "<p class='note'>No statements to show.</p>"


def dash_fail():
    byq, byc, why = {}, {}, Counter()
    for d in DASH.values():
        for k, c in d["byq"].items():
            byq.setdefault(k, Counter()).update(c)
        for k, c in d["byc"].items():
            byc.setdefault(k, Counter()).update(c)
        why.update(d["why"])
    cols = {
        "supported": _rgb(STATE_RGB["supported"]),
        "contradicted": _rgb(STATE_RGB["contradicted"]),
        "not_verifiable": _rgb(STATE_RGB["not_verifiable"]),
    }
    key = _key([(WORDS[v][1], cols[v]) for v in WORDS])
    qlabel = lambda w: {"none": "“none”", "unclear": "no word"}.get(w, f"“{w}”")
    clabel = lambda c: {
        "none": "No damage",
        "damaged": "Damaged (any)",
        "unclear": "Not stated",
    }.get(c, DMG_LABEL.get(c, c).title() + " damage")
    corder_c = ("damaged", "minor", "major", "destroyed", "none", "unclear")
    wtot = sum(why.values())
    wrows = "".join(
        f"<li><span>{esc(WHY_LABEL[k])}</span><b>{why[k]:,}</b></li>" for k in WHY_LABEL if why[k]
    )
    return (
        f"{key}<div class='dgrid'>"
        f"<div class='dcard'><h3>By the quantity word used</h3>{_fail_rows(byq, WORD_ORDER + ('unclear',), qlabel)}</div>"
        f"<div class='dcard'><h3>By the damage level named</h3>{_fail_rows(byc, corder_c, clabel)}</div></div>"
        "<p class='note'>Numbers on the right are matches / does not match / cannot check, summed over all events.</p>"
        + (
            f"<div class='dcard'><h3>Why {wtot:,} statements could not be checked</h3><ul class='topics'>{wrows}</ul></div>"
            if wtot
            else ""
        )
    )


def dash_tweets():
    rows = []
    for e, d in DASH.items():
        st = d["tw"]
        if not st:
            rows.append(
                f"<div class='drow'><span class='dn-name'>{esc(event_name(e))}</span>"
                "<span class='note'>No tweet statistics for this event.</span><span class='dn'></span></div>"
            )
            continue
        days = sorted(st["daily"])
        mx = max(v[0] for v in st["daily"].values()) or 1
        cols = "".join(
            f'<div class="col"><div class="tot" style="height:{st["daily"][x][0] / mx * 100:.1f}%">'
            f'<div class="flag" style="height:{st["daily"][x][1] / max(st["daily"][x][0], 1) * 100:.1f}%"></div></div></div>'
            for x in days
        )
        peak = max(days, key=lambda x: st["daily"][x][0])
        said = (
            f"{event_name(e)}: {st['n']:,} posts from {days[0]} to {days[-1]}, peak {peak} with {st['daily'][peak][0]:,}, "
            f"{_pct(st['flagged'], st['n'])} flagged useful"
        )
        rows.append(
            f"<div class='drow tw-row'><span class='dn-name'>{esc(event_name(e))}</span>"
            f"<div><div class='daily mini' role='img' aria-label='{esc(said)}'>{cols}</div>"
            f"<div class='axis'><span>{days[0]}</span><span>peak {peak}</span><span>{days[-1]}</span></div></div>"
            f"<span class='dn'>{st['n']:,} posts<br>{_pct(st['flagged'], st['n'])} useful</span></div>"
        )
    key = _key([("Flagged useful", "var(--accent)"), ("Other posts", "var(--bar-other)")])
    return f"<div class='dcard'>{key}{''.join(rows)}<p class='note'>Each bar is one day, scaled within its own event. Counts are from the test split of the tweet data.</p></div>"


def dash_topics():
    act = set(TWEET_STATS.get("actionable", []))
    cards = []
    for e, d in DASH.items():
        st = d["tw"]
        if not st or not st.get("classes"):
            continue
        cls = sorted(st["classes"].items(), key=lambda kv: -kv[1])[:5]
        tot = max(sum(st["classes"].values()), 1)
        mx = max(v for _, v in cls)
        rows = "".join(
            f'<div class="crow"><span class="cn">{esc(k.replace("_", " "))}</span>'
            f'<span class="cb"><i class="{"act" if k in act else ""}" style="width:{v / mx * 100:.1f}%"></i></span>'
            f'<span class="cv">{v / tot:.0%}</span></div>'
            for k, v in cls
        )
        cards.append(f"<div class='dcard'><h3>{esc(event_name(e))}</h3>{rows}</div>")
    if not cards:
        return "<p class='note'>No tweet statistics have been added to the demo.</p>"
    return (
        "<div class='dgrid'>" + "".join(cards) + "</div>"
        "<p class='note'>Share of each event's posts by topic. Blue bars are the topics counted as useful for responders.</p>"
    )


def dash_coverage():
    rows, tp = [], dict.fromkeys(("places", "caps", "img", "adv"), 0)
    for e, d in DASH.items():
        for k in tp:
            tp[k] += d[k]
        rows.append(
            f"<tr><th scope='row'>{esc(event_name(e))}</th><td>{d['places']}</td><td>{d['caps']}</td><td>{d['img']}</td>"
            f"<td>{d['adv']}</td><td>{'yes' if d['tw'] else 'no'}</td><td>{d['tw_ex'] or '–'}</td></tr>"
        )
    rows.append(
        f"<tr class='tot'><th scope='row'>All events</th><td>{tp['places']}</td><td>{tp['caps']}</td><td>{tp['img']}</td>"
        f"<td>{tp['adv']}</td><td></td><td></td></tr>"
    )
    head = (
        "<tr><th scope='col'>Event</th><th scope='col'>Places</th><th scope='col'>With a caption</th>"
        "<th scope='col'>With imagery</th><th scope='col'>With advice</th><th scope='col'>Tweet counts</th>"
        "<th scope='col'>Example tweets</th></tr>"
    )
    return (
        f"<div class='dscroll'><table class='dtable'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"
        "<p class='note'>What this version of the demo contains. Anything missing is simply not loaded, not a model failure.</p>"
    )


def dashboard_blocks():
    """(heading, html) pairs for the Dashboard tab."""
    if not DASH:
        return [
            (
                "Dashboard",
                "<p class='note'>No evidence files are loaded, so there is nothing to summarise.</p>",
            )
        ]
    return [
        ("Across all events", dash_kpis()),
        ("Events at a glance", dash_overview()),
        ("How much damage each event shows", dash_damage()),
        ("How the caption checks came out", dash_checks()),
        ("Which statements fail", dash_fail()),
        ("Tweet activity", dash_tweets()),
        ("What people posted about", dash_topics()),
        ("What the demo covers", dash_coverage()),
    ]


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,700;1,9..144,400&family=Inter:wght@400..700&display=swap');
:root{--paper:#EFE6D2;--card:rgba(255,255,255,.58);--card-solid:#FFFFFF;--glass-edge:rgba(255,255,255,.78);--glass-hi:rgba(255,255,255,.85);--glass-shadow:0 8px 24px rgba(15,36,56,.10);--blobs:radial-gradient(900px 620px at 8% 4%,rgba(10,98,163,.30),transparent 60%),radial-gradient(820px 620px at 96% 92%,rgba(31,122,90,.24),transparent 60%);--ink:#0F2438;--ink2:#3A4F63;--muted:#526679;--line:#CFD9E3;--accent:#0A62A3;--on-accent:#FFFFFF;
--ok:#1F7A5A;--ok-bg:#E3F3EC;--bad:#B23A1F;--bad-bg:#FBE9E3;--na:#55666F;--na-bg:#E8EDF1;
--warn-bg:#FFF6DD;--warn-line:#E8D28A;--warn-ink:#5A4600;--track:#E3E9EE;--bar-other:#C5D0DA;--bar-mid:#8FA1B0;--stripe:#E3E9EE;
--d-minor:#F3CFA8;--d-major:#D9803C;--d-destroyed:#8A3511;--on-solid:#FFFFFF;--outline:transparent;--focus:#B87A00;
--page:#EFE6D2}
:root[data-fv="dark"]{--paper:#0C1620;--card:rgba(24,42,58,.62);--card-solid:#13222F;--glass-edge:rgba(255,255,255,.12);--glass-hi:rgba(255,255,255,.10);--glass-shadow:0 8px 24px rgba(0,0,0,.35);--blobs:radial-gradient(900px 620px at 8% 4%,rgba(90,180,240,.20),transparent 60%),radial-gradient(820px 620px at 96% 92%,rgba(46,170,220,.14),transparent 60%);--ink:#E7EFF6;--ink2:#B7C6D4;--muted:#94A7B8;--line:#2B4052;--accent:#5AB4F0;--on-accent:#06121C;
--ok:#5CC79B;--ok-bg:#12352A;--bad:#FF8F70;--bad-bg:#3B1E16;--na:#A9B7C1;--na-bg:#1F2E3A;
--warn-bg:#3A2E0B;--warn-line:#7A6320;--warn-ink:#F3DE9A;--track:#22364A;--bar-other:#35506A;--bar-mid:#6C8397;--stripe:#22364A;
--on-solid:#06121C;--outline:rgba(231,239,246,.45);--focus:#FFC83D;
--page:radial-gradient(900px 520px at 92% -6%,rgba(90,180,240,.16),transparent 62%),radial-gradient(820px 620px at -6% 104%,rgba(46,170,220,.12),transparent 60%),linear-gradient(160deg,#0C1620 0%,#13222F 100%)}
.gradio-container,.dark .gradio-container{background:transparent!important;color:var(--ink);max-width:1240px!important;margin:auto;padding-top:28px!important;font-family:'Inter',system-ui,sans-serif;
--body-background-fill:var(--paper);--body-text-color:var(--ink);--body-text-color-subdued:var(--ink2);
--background-fill-primary:var(--card);--background-fill-secondary:var(--paper);
--block-background-fill:var(--card);--block-border-color:var(--line);--border-color-primary:var(--line);--border-color-accent:var(--accent);
--block-label-background-fill:var(--card);--block-label-text-color:var(--ink2);--block-title-text-color:var(--ink);--block-info-text-color:var(--ink2);
--input-background-fill:var(--card);--input-background-fill-focus:var(--card);--input-border-color:var(--line);--input-border-color-focus:var(--accent);--input-placeholder-color:var(--muted);
--button-secondary-background-fill:var(--card);--button-secondary-background-fill-hover:var(--track);--button-secondary-text-color:var(--ink);--button-secondary-border-color:var(--line);
--button-primary-background-fill:var(--accent);--button-primary-background-fill-hover:var(--accent);--button-primary-text-color:var(--on-accent);--button-primary-border-color:var(--accent);
--checkbox-label-background-fill:var(--card);--checkbox-label-background-fill-hover:var(--track);--checkbox-label-background-fill-selected:var(--accent);
--checkbox-label-text-color:var(--ink);--checkbox-label-text-color-selected:var(--on-accent);--checkbox-label-border-color:var(--line);--checkbox-label-border-color-selected:var(--accent);
--panel-background-fill:var(--card);--table-even-background-fill:var(--card);--table-odd-background-fill:var(--paper);--table-text-color:var(--ink);--table-border-color:var(--line);
--slider-color:var(--accent);--color-accent:var(--accent);--link-text-color:var(--accent);--neutral-400:var(--muted);--neutral-500:var(--muted)}
.gradio-container label span,.gradio-container .block-info,.gradio-container [data-testid="block-info"]{color:var(--ink2)!important}
.gradio-container input,.gradio-container textarea,.gradio-container select{color:var(--ink)!important}
.gradio-container .label-wrap span{color:var(--ink)!important}
.fv-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.cell{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.5rem .55rem;min-height:118px;display:flex;flex-direction:column;gap:.25rem}
.cell .name{font-size:.8rem;font-weight:600;color:var(--ink2);display:flex;justify-content:space-between;gap:.4rem}
.cell .name small{font-weight:400;color:var(--muted)}
.cell .big{font-size:1.4rem;font-weight:700;line-height:1.1;color:var(--ink)}
.cell .big span{font-size:.78rem;font-weight:500;color:var(--ink2);margin-left:.25rem}
.cell.empty .big,.cell.nodata .big{font-size:.92rem;font-weight:600;color:var(--muted)}
.cell.empty{background:var(--paper)}
.cell.nodata{background:repeating-linear-gradient(135deg,var(--card) 0 6px,var(--stripe) 6px 8px)}
.bar{display:flex;height:8px;border-radius:2px;background:var(--track);overflow:hidden;gap:2px}.bar i{display:block;height:100%}
.d-minor{background:var(--d-minor)}.d-major{background:var(--d-major)}.d-destroyed{background:var(--d-destroyed)}
.bar i,.legend i{box-shadow:0 0 0 1px var(--outline)}
.dmg{font-size:.76rem;color:var(--ink2)}
.chips{margin-top:auto;display:flex;flex-direction:column;gap:3px}
.chip{font-size:.76rem;padding:2px 6px;border-radius:4px;font-weight:600;width:fit-content}
.chip.supported,.card.supported .v{background:var(--ok-bg);color:var(--ok)}
.chip.contradicted,.card.contradicted .v{background:var(--bad-bg);color:var(--bad)}
.chip.not_verifiable,.card.not_verifiable .v{background:var(--na-bg);color:var(--na)}
.chip.not_verifiable{background-image:repeating-linear-gradient(135deg,transparent 0 4px,var(--line) 4px 6px)}
.legend{display:flex;gap:1rem;flex-wrap:wrap;margin:.5rem 0 .1rem;font-size:.82rem;color:var(--ink2)}
.legend span{display:flex;align-items:center;gap:.35rem}.legend i{width:12px;height:8px;border-radius:2px;display:inline-block}
.legend .nodmg{background:var(--card);outline:1px solid var(--muted)}
.legend .t-ok{color:var(--ok)}.legend .t-bad{color:var(--bad)}.legend .t-na{color:var(--na)}
.note{font-size:.84rem;color:var(--muted);margin:.4rem 0 0}
.warn{background:var(--warn-bg);border:1px solid var(--warn-line);border-radius:8px;padding:.5rem .75rem;margin:.2rem 0 .5rem;font-size:.92rem;color:var(--warn-ink)}
.result-head h3{font-size:1.2rem;font-weight:700;margin:.1rem 0 .5rem;color:var(--ink)}
.stack{display:flex;height:10px;gap:2px;border-radius:3px;overflow:hidden;background:var(--track)}.stack i{display:block;height:100%}
.stack .supported{background:var(--ok)}.stack .contradicted{background:var(--bad)}
.stack .not_verifiable{background:repeating-linear-gradient(135deg,var(--bar-mid) 0 4px,var(--bar-other) 4px 6px)}
.counts{display:flex;gap:1.1rem;flex-wrap:wrap;list-style:none;padding:0;margin:.5rem 0 0;font-size:.9rem}
.counts li.supported{color:var(--ok)}.counts li.contradicted{color:var(--bad)}.counts li.not_verifiable{color:var(--na)}
.cards{list-style:none;padding:0;margin:.4rem 0 0;display:flex;flex-direction:column;gap:8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.5rem .7rem;display:grid;gap:.2rem}
.card .v{font-size:.8rem;font-weight:700;padding:1px 7px;border-radius:4px;width:fit-content}
.card .t{font-size:.94rem;color:var(--ink)}.card .where{font-weight:700;margin-right:.25rem}.card .r{font-size:.88rem;color:var(--ink2)}
.empty-state{background:var(--card);border:1px dashed var(--line);border-radius:8px;padding:.8rem;color:var(--ink2)}
.wrow{display:flex;gap:8px;flex-wrap:wrap}
.wchip{display:flex;flex-direction:column;padding:.35rem .7rem;border-radius:8px;font-weight:600;font-size:.95rem;min-width:84px}
.wchip small{font-weight:400;font-size:.72rem;color:inherit;opacity:.9}
.wchip.yes{background:var(--ok-bg);color:var(--ok)}
.wchip.no{background:repeating-linear-gradient(135deg,var(--card) 0 5px,var(--stripe) 5px 7px);color:var(--na);border:1px solid var(--line)}
.tweets{list-style:none;margin:.5rem 0 0;padding:0;display:grid;gap:12px}
.tweets .tweet{display:flex;gap:12px;align-items:flex-start;background-color:var(--card);border:1px solid var(--line);border-radius:var(--r-lg,16px);padding:14px 16px;margin:0!important}
.tweets .tw-av{flex:none;width:40px;height:40px;border-radius:50%;background-color:var(--accent);color:var(--on-accent);display:flex;align-items:center;justify-content:center}
.tweets .tw-body{min-width:0;flex:1}
.tweets .tw-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 8px;font-size:.88rem;line-height:1.3}
.tweets .tw-head b{color:var(--ink);font-family:'Inter',system-ui,sans-serif;font-weight:700}
.tweets .tw-h,.tweets .tw-d{color:var(--muted)}
.tweets .tx{margin:.3rem 0 0;font-family:'Inter',system-ui,sans-serif;font-size:1rem;line-height:1.5;color:var(--ink);overflow-wrap:anywhere;white-space:pre-line}
.tweets .tg{color:var(--accent)}
.tweets .tw-c{display:inline-block;margin-top:.55rem;padding:2px 10px;border-radius:999px;border:1px solid var(--line);background-color:var(--card-solid);color:var(--ink2);font-size:.78rem}
.ex-h,.ex-title{margin:.9rem 0 .2rem!important;font-size:.92rem!important}
.tw{display:grid;grid-template-columns:1.2fr 1fr;gap:1.6rem;color:var(--ink)}.tw h3{margin:.1rem 0 .4rem;font-size:.98rem;color:var(--ink)}
.daily{display:flex;align-items:flex-end;gap:2px;height:130px;border-bottom:1px solid var(--line)}
.daily .col{flex:1;height:100%;display:flex;align-items:flex-end}
.daily .tot{width:100%;background:var(--bar-other);display:flex;align-items:flex-end;border-radius:2px 2px 0 0}
.daily .flag{width:100%;background:var(--accent)}
.axis{display:flex;justify-content:space-between;font-size:.78rem;color:var(--muted);margin-top:.2rem}
.flagk{background:var(--accent)}.otherk{background:var(--bar-other)}
.crow{display:grid;grid-template-columns:minmax(120px,1.3fr) 2fr 3rem;gap:.5rem;align-items:center;font-size:.85rem;margin:.22rem 0;color:var(--ink2)}
.crow .cb{height:8px;background:var(--track);border-radius:2px;overflow:hidden}.crow .cb i{display:block;height:100%;background:var(--bar-mid)}
.crow .cb i.act{background:var(--accent)}.crow .cv{text-align:right;color:var(--muted)}
.fv-foot{font-size:.84rem;color:var(--muted);margin:1rem 0 .5rem;max-width:80ch}
button.primary{background:var(--accent)!important;border-color:var(--accent)!important;color:var(--on-accent)!important}
:focus-visible{outline:3px solid var(--focus)!important;outline-offset:2px}
#fv-mode label.selected{background:var(--accent)!important}
.gradio-container #fv-mode label.selected span{color:var(--on-accent)!important;background:transparent!important}
#fv-mode label span{background:transparent!important}
.textfield .token.highlighted{color:var(--ink)!important;border-left:4px solid var(--na)}
.textfield .token.highlighted .text{color:var(--ink)!important}
.textfield .token.highlighted .label{color:var(--on-solid)!important;font-weight:700;opacity:1!important}
.textfield .token.highlighted[style*="205, 235, 221"]{background:var(--ok-bg)!important;border-left-color:var(--ok)!important}
.textfield .token.highlighted[style*="247, 207, 196"]{background:var(--bad-bg)!important;border-left-color:var(--bad)!important}
.textfield .token.highlighted[style*="221, 228, 231"]{background:var(--na-bg)!important;border-left-color:var(--na)!important}
.textfield .label[style*="205, 235, 221"]{background:var(--ok)!important}
.textfield .label[style*="247, 207, 196"]{background:var(--bad)!important}
.textfield .label[style*="221, 228, 231"]{background:var(--na)!important}
.meta{display:flex;align-items:center;gap:.5rem;font-size:.8rem;color:var(--ink2);margin-top:.15rem}
.meta b{color:var(--ink)}
.conf{display:inline-block;width:46px;height:6px;border-radius:3px;background:var(--track);overflow:hidden;flex:none}
.conf i{display:block;height:100%;background:var(--accent)}
.warn ul{margin:.3rem 0 0;padding-left:1.1rem}.warn li{margin:.2rem 0}.warn code{color:inherit;background:transparent;padding:0;font-weight:600}
.simlist{list-style:none;padding:0;margin:.3rem 0 0;display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.card.sim{display:flex;gap:.7rem;align-items:flex-start}
.sim .rank{flex:none;width:28px;height:28px;border-radius:50%;background:var(--accent);color:var(--on-accent);font-weight:700;
display:flex;align-items:center;justify-content:center;font-size:.85rem}
.sim .body{min-width:0;display:grid;gap:.3rem;flex:1}.sim .bar{margin:.1rem 0}
.tag{font-size:.72rem;font-weight:600;color:var(--ink2);border:1px solid var(--line);border-radius:999px;padding:1px 8px;margin-left:.3rem}
html,body,gradio-app{background-color:var(--paper)!important;background-image:var(--blobs)!important;background-repeat:no-repeat!important;background-attachment:fixed!important}
h1,h2,h3,h4,.serif{font-family:'Fraunces',Georgia,serif!important}
[role="tab"],.label-wrap,.label-wrap span,button.primary,button.secondary,button.lg,button.sm,
.gradio-container label>span,.gradio-container .block-title,.gradio-container span[data-testid="block-info"],.card b,.sim b{font-family:'Fraunces',Georgia,serif!important}
.gradio-container [role="tab"]{color:var(--ink2)!important;font-size:1rem}
.gradio-container [role="tab"][aria-selected="true"]{color:var(--accent)!important;border-color:var(--accent)!important}
.cell,.card,.sitrep,.empty-state,.warn{border-radius:12px}
.panel-title{font-family:'Fraunces',Georgia,serif;font-size:1.3rem;font-weight:700;margin:.2rem 0 .6rem;color:var(--ink);border-left:4px solid var(--accent);padding-left:.65rem}
#fv-bar{justify-content:flex-end!important;margin:0 0 14px!important;padding:0 10px 0 0!important;position:relative;z-index:6;background:transparent!important;border:0!important}
#fv-bar>*{pointer-events:auto;flex:none!important;min-width:0!important}
#fv-mode{position:relative!important;width:max-content!important;max-width:none!important;overflow:visible!important;min-width:0!important;flex:none!important;background:var(--card)!important;border:1px solid var(--line)!important;border-radius:999px!important;padding:2px!important;box-shadow:none!important}
#fv-mode .wrap,#fv-mode fieldset{overflow:visible!important;width:max-content!important}
#fv-mode .wrap{gap:2px!important;flex-wrap:nowrap!important;flex-direction:row!important}
#fv-mode label{border-radius:999px!important;padding:3px 10px!important;border:0!important;background:transparent!important;cursor:pointer;white-space:nowrap}
#fv-mode label span{font-family:'Fraunces',Georgia,serif!important;font-weight:600;font-size:.82rem!important}
#fv-mode input[type=radio]{position:absolute!important;opacity:0!important;width:1px!important;height:1px!important;margin:0!important;appearance:none;-webkit-appearance:none;pointer-events:none}
#fv-mode label:focus-within{outline:3px solid var(--focus)}
#hero{position:relative;overflow:hidden;display:flex;justify-content:space-between;align-items:center;gap:1.5rem;background:linear-gradient(115deg,#12304A 0%,#0A1824 78%);border:1px solid #243B52;border-radius:16px;padding:30px 34px 34px 48px;margin:0 0 10px}
#hero::before{content:"";position:absolute;left:0;top:0;bottom:0;width:14px;background:linear-gradient(180deg,#7CD0F5 0 55%,#0B6FB8 55% 100%)}
#hero .hero-text{position:relative;z-index:1;max-width:64ch}
#hero .eyebrow{font-size:.92rem;font-weight:600;color:#7CD0F5;margin-bottom:6px}
#hero h1{display:flex;align-items:center;gap:.6rem;margin:.1em 0 .25em;font-size:3rem;font-weight:700;line-height:1.05;color:#F2F7FB!important;letter-spacing:-.01em}
#hero p{margin:0;font-family:'Fraunces',Georgia,serif;font-style:italic;font-size:1.2rem;line-height:1.45;color:#D3E0EC;max-width:none}
.pill{display:inline-block;margin:16px 8px 0 0;padding:4px 12px;border-radius:999px;font-size:12.5px;font-weight:600;color:#8FD0F7;background:rgba(90,180,240,.12);border:1px solid rgba(90,180,240,.45)}
#hero .hero-art{position:relative;z-index:1;flex:none;width:190px;height:190px;margin-top:40px}
#hero .hero-wave{position:absolute;left:0;right:0;bottom:0;width:100%;height:64px;pointer-events:none}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;list-style:none;padding:0;margin:.4rem 0 .8rem}
.steps li{display:flex;gap:.75rem;align-items:flex-start;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;color:var(--ink2);font-size:.9rem;line-height:1.4}
.steps .n{flex:none;width:30px;height:30px;border-radius:50%;background:var(--accent);color:var(--on-accent);font-family:'Fraunces',Georgia,serif;font-weight:700;display:flex;align-items:center;justify-content:center}
.steps b{display:block;color:var(--ink);font-family:'Fraunces',Georgia,serif;font-size:1.02rem;margin-bottom:.1rem}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:.2rem 0 .8rem}
.kpis6{grid-template-columns:repeat(3,1fr)}
.kpi{--kc:#5AB4F0;--kn:#8FD0F7;background:#0F2438;border:1px solid #243B52;border-bottom:4px solid var(--kc);border-radius:12px;padding:14px 18px;color:#B7C6D4}
.kpi b{display:block;font-family:'Fraunces',Georgia,serif;font-size:34px;line-height:1.1;color:var(--kn)}
.kpi span{font-size:12.5px;color:#B7C6D4!important;font-family:'Inter',system-ui,sans-serif!important}
.kpi.ok{--kc:#5CC79B;--kn:#5CC79B}.kpi.bad{--kc:#FF8F70;--kn:#FF8F70}.kpi.na{--kc:#A9B7C1;--kn:#C3CED6}
.foot{margin:40px 0 4px;padding:18px 16px 0;border-top:1px solid var(--line);text-align:center;font-size:.85rem;line-height:1.5;color:var(--muted)}
.foot p{margin:.15rem auto;max-width:72ch}
.foot .mem{white-space:nowrap}
.foot a{display:inline!important;padding:0!important;margin:0!important;color:inherit!important;text-decoration:underline;text-underline-offset:2px}
.foot .team{margin-bottom:.6rem}
.foot b{color:var(--ink);font-weight:600}
footer{display:none!important}
.gradio-container label.selected span{color:var(--on-accent)!important}
.gradio-container button.sm,.gradio-container button.secondary{border:1px solid var(--line)!important;border-radius:999px!important;font-weight:600}
.chips-row{display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center;justify-content:center;margin:.5rem 0 .2rem}
.chips-row .lead{font-family:'Fraunces',Georgia,serif;font-weight:600;color:var(--ink2);margin:0 .2rem 0 .6rem;font-size:.9rem}
.lchip{display:inline-block;padding:3px 11px;border-radius:8px;font-size:.82rem;font-weight:600;color:#0F2438;border:1px solid rgba(15,36,56,.25)}
.lchip.l1{background:#7AB8F0}.lchip.l2{background:#FFD166}.lchip.l3{background:#F28C28}.lchip.l4{background:#B5400E;color:#fff}
.caption{font-family:'Fraunces',Georgia,serif;font-style:italic;font-size:.92rem;color:var(--muted);text-align:center;margin:.3rem 0 .6rem}
#fv-controls{align-items:flex-start}
#fv-controls label>span,#fv-controls [data-testid="block-info"]{font-family:'Inter',system-ui,sans-serif!important}
#fv-controls label>span:first-child,#fv-controls .block-title{font-size:1rem!important;font-weight:700!important;color:var(--ink)!important}
#fv-controls [data-testid="block-info"]{font-size:.88rem!important;font-weight:400!important;color:var(--ink2)!important;margin-bottom:.4rem!important}
#fv-tile .wrap,#fv-tile .wrap-inner{min-height:48px;border-radius:12px!important}
#fv-tile .wrap{border:2px solid var(--accent)!important;background:var(--card)!important}
#fv-tile input{font-family:'Inter',system-ui,sans-serif!important;font-size:1.05rem!important;font-weight:600!important;color:var(--ink)!important}
#fv-tile svg{color:var(--accent)!important}
#fv-tile ul.options,.gradio-container ul.options{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:12px!important;box-shadow:0 12px 32px rgba(0,0,0,.28)!important;padding:6px!important;max-height:340px!important}
.gradio-container ul.options li{font-family:'Inter',system-ui,sans-serif!important;font-size:1rem!important;color:var(--ink)!important;padding:10px 12px!important;border-radius:8px!important}
.gradio-container ul.options li.active,.gradio-container ul.options li:hover{background:var(--track)!important}
.gradio-container ul.options li.selected{background:var(--accent)!important;color:var(--on-accent)!important;font-weight:700}
.gradio-container ul.options li .inner-item,.gradio-container ul.options li span{color:inherit!important}
.gradio-container ul.options li .inner-item{display:none!important}
#fv-step{gap:8px!important;margin-top:.5rem}
#fv-step button{flex:1;min-height:40px;font-size:.92rem!important}
#fv-keep .wrap{display:grid!important;grid-template-columns:1fr 1fr;gap:8px!important}
#fv-keep label{border:1.5px solid var(--line)!important;border-radius:12px!important;padding:9px 12px!important;background:var(--card)!important;min-height:44px;cursor:pointer}
#fv-keep label span{font-family:'Inter',system-ui,sans-serif!important;font-size:.98rem!important;font-weight:600!important;color:var(--ink)!important}
#fv-keep label::before{content:"";flex:none;width:14px;height:14px;border-radius:4px;margin-right:2px;background:var(--sw);border:1px solid rgba(0,0,0,.35)}
#fv-keep label:nth-child(1){--sw:#7AB8F0}#fv-keep label:nth-child(2){--sw:#FFD166}#fv-keep label:nth-child(3){--sw:#F28C28}#fv-keep label:nth-child(4){--sw:#B5400E}
#fv-keep label.selected{background:var(--accent)!important;border-color:var(--accent)!important}
#fv-keep label.selected span{color:var(--on-accent)!important}
#fv-keep label.selected::before{content:"✓";display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;color:#0A1824;border-color:var(--on-accent)}
#fv-keep input[type=checkbox]{position:absolute;opacity:0;width:0;height:0;margin:0}
#fv-keep label{position:relative}
#fv-keep label:focus-within{outline:3px solid var(--accent);outline-offset:2px}
#fv-step button,#fv-step button span{font-family:'Inter',system-ui,sans-serif!important;font-weight:600}
.chip.nomention{background:var(--card);color:var(--ink2);border:1px solid var(--line)}
.why{margin:.6rem auto 0;max-width:900px;border:1.5px dashed var(--line);border-radius:12px;padding:.8rem 1rem;background:var(--card)}
.why b:first-child{font-family:'Inter',system-ui,sans-serif;font-size:1rem;color:var(--ink)}
.why p{margin:.3rem 0 .4rem;color:var(--ink2);font-size:.95rem}
.why ul{margin:0;padding-left:1.1rem;color:var(--ink)}.why li{margin:.15rem 0;font-size:.95rem}
.chk{list-style:none;margin:.4rem 0 0;padding:0;display:flex;flex-direction:column;gap:.5rem}
.chk li{display:flex;gap:.7rem;align-items:flex-start}
.chk .mk{flex:none;width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.9rem}
.chk li.yes .mk{background:var(--ok-bg);color:var(--ok);border:1.5px solid var(--ok)}
.chk li.no .mk{background:var(--bad-bg);color:var(--bad);border:1.5px solid var(--bad)}
.chk b{display:block;color:var(--ink);font-size:.98rem}.chk small{display:block;color:var(--ink2);font-size:.92rem;line-height:1.4}
.sitrep{position:relative;background:var(--card);border:2px solid var(--accent);border-radius:16px;padding:1.5rem 1.7rem 1.4rem;margin:1.2rem 0 .4rem;box-shadow:0 14px 36px rgba(11,111,184,.18);overflow:hidden;color:var(--ink)}
.sitrep::before{content:"";position:absolute;left:0;right:0;top:0;height:7px;background:linear-gradient(90deg,var(--accent),#1F7A5A)}
.sr-head h2{margin:.2rem 0 .15rem;font-size:2rem;line-height:1.15;color:var(--ink)}
.sitrep .sub{margin:0 0 1rem;color:var(--ink2);font-size:.95rem}
.sr-flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 1.1rem;padding:.7rem .8rem;background:var(--paper);border-radius:12px}
.sr-flow .st{font-size:.95rem;padding:.35rem .75rem;border-radius:999px;border:1.5px solid var(--line);color:var(--ink);background:var(--card)}
.sr-flow .st b{font-size:1.15rem;margin-right:.15rem}.sr-flow .ar{color:var(--muted);font-size:1.2rem}
.sr-flow .st.yes{border-color:var(--ok);background:var(--ok-bg);color:var(--ok)}
.sr-flow .st.no{border-color:var(--bad);background:var(--bad-bg);color:var(--bad)}
.sr-flow .st.na{border-style:dashed;background:var(--na-bg);color:var(--na)}
.sr-cols{display:grid;grid-template-columns:1.35fr 1fr;gap:1.6rem}
.sitrep h3{margin:.2rem 0 .5rem;font-size:1.15rem;color:var(--ink);font-family:'Fraunces',Georgia,serif}
.sr-list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:.6rem}
.sr-list li{display:flex;gap:.7rem;align-items:flex-start;color:var(--ink);font-size:1.02rem;line-height:1.4}
.sr-list small{display:block;color:var(--ink2);font-size:.88rem;margin-top:.1rem}
.sr-list.small li{font-size:.93rem}
.sr-list .mk{flex:none;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.95rem;border:1.5px solid}
.sr-list li.yes .mk{background:var(--ok-bg);color:var(--ok);border-color:var(--ok)}
.sr-list li.no .mk{background:var(--bad-bg);color:var(--bad);border-color:var(--bad)}
.sr-list li.na .mk{background:var(--na-bg);color:var(--na);border-color:var(--na);border-style:dashed}
.sr-list.small .mk{width:24px;height:24px;font-size:.85rem}
.sr-empty{color:var(--ink2);font-size:.98rem;margin:.2rem 0}
.advbox{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px 16px;height:auto}
.advbox h3{margin:0 0 .5rem;font-size:.95rem;color:var(--muted)}
.advtext{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;font-size:.98rem;color:var(--ink);max-height:none;overflow:visible}
.advisory{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;height:auto}
.advisory.advbox{padding:0}
.advisory .adv-bar{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.2rem 1rem;padding:.7rem 1rem;background:var(--accent);color:var(--on-accent)}
.advisory .adv-bar h3{margin:0;font-family:'Fraunces',Georgia,serif;font-size:1.15rem;font-weight:700;color:inherit}
.advisory .adv-bar span{font-size:.85rem;color:inherit}
.advisory .adv-basis{padding:.65rem 1rem;background:var(--paper);border-bottom:1px solid var(--line);font-size:.9rem;color:var(--ink2)}
.advisory .adv-basis ul{list-style:none;margin:.25rem 0 0;padding:0;display:flex;flex-direction:column;gap:.1rem}
.advisory .adv-basis b{color:var(--ink)}
.advisory .adv-list{list-style:none;margin:0;padding:.35rem 1rem .5rem;display:flex;flex-direction:column}
.advisory .adv-list li{display:flex;justify-content:space-between;align-items:flex-start;gap:.9rem;padding:.75rem 0;border-bottom:1px solid var(--line)}
.advisory .adv-list li:last-child{border-bottom:0}
.advisory .adv-list li>div{border-inline-start:3px solid var(--accent);padding-inline-start:.8rem;min-width:0}
.advisory .adv-head{margin:0;font-family:'Fraunces',Georgia,serif;font-size:1.08rem;font-weight:600;line-height:1.3;color:var(--ink)}
.advisory .adv-rest{margin:.3rem 0 0;font-size:.95rem;line-height:1.55;color:var(--ink2)}
.advisory .adv-src{flex:none;margin-top:.15rem;padding:.05rem .55rem;border:1px solid var(--line);border-radius:999px;font-size:.78rem;color:var(--muted);white-space:nowrap;direction:ltr;unicode-bidi:isolate}
.advisory[dir=rtl] .adv-head,.advisory[dir=rtl] .adv-bar h3{font-family:'Inter',system-ui,sans-serif;font-weight:650}
@media (max-width:520px){.advisory .adv-list li{flex-direction:column;gap:.4rem}}
#fv-adv-en,#fv-adv-ar{height:auto!important;max-height:none!important;overflow:visible!important}
.sr-sum{margin:.9rem 0 .2rem;font-size:1.02rem;color:var(--ink);line-height:1.5}
.topics{list-style:none;margin:.3rem 0 0;padding:0;display:grid;gap:.25rem}
.topics li{display:flex;justify-content:space-between;gap:1rem;font-size:.92rem;color:var(--ink);border-bottom:1px dashed var(--line);padding:.15rem 0}
.topics b{font-variant-numeric:tabular-nums}
.sr-more{margin-top:1.1rem;padding-top:.9rem;border-top:1px solid var(--line);display:grid;grid-template-columns:1fr 1fr;gap:1.6rem}
.sr-more p{margin:.1rem 0;font-size:.95rem;color:var(--ink)}
.sitrep .adv{white-space:pre-wrap;font-family:inherit;font-size:.92rem;background:var(--paper);color:var(--ink);border-radius:8px;padding:.6rem .8rem}
@media (max-width:760px){.sr-cols,.sr-more{grid-template-columns:1fr}.sr-head h2{font-size:1.6rem}}
/* reduced motion: nothing moves for people who ask for that */
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
/* map toolbar and slider reset: bigger touch targets */
.gradio-container .icon-button-wrapper button,.gradio-container .icon-button-wrapper a.download-link{min-width:32px!important;min-height:32px!important;display:inline-flex;align-items:center;justify-content:center}
.gradio-container .icon-button-wrapper svg{width:20px!important;height:20px!important}
.gradio-container .reset-button{min-width:32px!important;min-height:32px!important}
/* map titles wrap instead of being cut off */
.gradio-container [data-testid="block-label"]{white-space:normal!important;max-width:calc(100% - 90px);height:auto!important;line-height:1.25}
/* caption list labels: words, never broken mid-word */
.gradio-container .textfield .token .label{text-transform:none!important;letter-spacing:0!important;white-space:nowrap!important;overflow-wrap:normal!important;word-break:normal!important;font-size:.8rem!important;font-weight:700}
/* reasons open on tap or keyboard */
details.chip{display:block;padding:0}details.chip summary{list-style:none;cursor:pointer;padding:5px 8px;min-height:24px;display:block}
details.chip summary::-webkit-details-marker{display:none}details.chip summary:focus-visible{outline:3px solid var(--focus);outline-offset:1px}
details.chip .tip{padding:2px 8px 6px;font-weight:400;font-size:.8rem;line-height:1.35;color:var(--ink)}
.daily-nums{margin-top:.5rem;color:var(--ink)}.daily-nums summary{cursor:pointer;font-size:.9rem;color:var(--accent);min-height:24px}
.daily-nums ul{margin:.3rem 0 0;padding-left:1.1rem;font-size:.88rem}
/* the three steps all look and act like links */
.steps b a{color:inherit;text-decoration:underline;text-underline-offset:3px}.steps b a:hover{color:var(--accent)}
.steps b a:focus-visible{outline:3px solid var(--focus);outline-offset:2px;border-radius:3px}
/* phones: stack the rows, keep the tab bar inside its own strip */
/* controls row: two separate cards, equal gap */
#fv-controls>.form{gap:16px!important;background:none!important;background-color:transparent!important;border:0!important;box-shadow:none!important;-webkit-backdrop-filter:none!important;backdrop-filter:none!important;overflow:visible!important}
#fv-controls>.form>.block{border:1px solid var(--glass-edge)!important;border-radius:var(--r-lg)!important;overflow:hidden}
/* floating glass tab dock */
.gradio-container{overflow:clip!important}
.gradio-container .tab-wrapper{position:sticky!important;top:max(24px,env(safe-area-inset-top));z-index:40;width:fit-content!important;max-width:calc(100% - 8px)!important;margin:6px auto 18px!important;padding:0!important;border:0!important;background:none!important;box-shadow:none!important}
.gradio-container .tab-wrapper::after,.gradio-container .tab-wrapper::before{display:none!important}
.gradio-container .tab-container{display:flex!important;gap:4px;padding:6px!important;border:1px solid var(--glass-edge)!important;border-radius:999px!important;background-color:var(--card)!important;-webkit-backdrop-filter:blur(18px) saturate(170%);backdrop-filter:blur(18px) saturate(170%);box-shadow:var(--glass-shadow),inset 0 1px 0 var(--glass-hi);overflow:visible!important;height:auto!important}
.gradio-container .tab-container::after{display:none!important}
.gradio-container .tab-container button[role=tab]{border:0!important;border-bottom:0!important;border-radius:999px!important;padding:8px 18px!important;min-height:40px;background:transparent!important;transition:background-color .15s,color .15s}
.gradio-container .tab-container button[role=tab]:hover{background-color:rgba(10,98,163,.10)!important}
.gradio-container .tab-container button[role=tab].selected,.gradio-container .tab-container button[role=tab][aria-selected="true"]{background-color:var(--accent)!important;color:var(--on-accent)!important;box-shadow:0 2px 8px rgba(10,98,163,.28)}
.gradio-container .tab-container button[role=tab].selected::after,.gradio-container .tab-container button[role=tab]::after{display:none!important}
@media (prefers-reduced-transparency:reduce){.gradio-container .tab-container{background-color:var(--card-solid)!important;-webkit-backdrop-filter:none;backdrop-filter:none}}
.dcard{background-color:var(--card);border:1px solid var(--line);border-radius:var(--r-lg,16px);padding:16px 18px;margin:.3rem 0 .6rem;color:var(--ink)}
.dcard h3{margin:.1rem 0 .5rem;font-size:.98rem;color:var(--ink)}
.dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin:.3rem 0 .6rem}.dgrid .dcard{margin:0}
.dgrid .drow{grid-template-columns:minmax(90px,.9fr) 2fr auto}
.drow{display:grid;grid-template-columns:minmax(130px,.8fr) 3fr minmax(96px,auto);gap:14px;align-items:center;margin:.55rem 0;font-size:.9rem;color:var(--ink)}
.dn-name{font-weight:600}.dn{color:var(--muted);text-align:right;font-variant-numeric:tabular-nums;font-size:.85rem}
.sbar{display:flex;height:16px;border-radius:8px;overflow:hidden;background-color:var(--track)}.sbar i{display:block;height:100%;min-width:2px}
.dkey{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:.84rem;color:var(--ink2);margin:.1rem 0 .5rem}
.dkey i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.daily.mini{height:56px}.tw-row .axis{margin-top:2px}
.dscroll{overflow-x:auto;margin:.3rem 0 .4rem;width:0;min-width:100%}
.dtable{width:100%;border:0!important;border-collapse:collapse;font-size:.92rem;color:var(--ink);background:none}
.dtable thead,.dtable tbody,.dtable tr{border:0!important;background:none!important}
.dtable th,.dtable td{padding:9px 12px;text-align:right;border:0;border-bottom:1px solid var(--line);background:none!important;color:var(--ink);font-variant-numeric:tabular-nums}
.dtable th:first-child,.dtable td:first-child{text-align:left}.dtable thead th{color:var(--ink2);font-weight:600;font-size:.84rem}
.dtable tbody th{font-weight:600}.dtable tr.tot th,.dtable tr.tot td{border-top:2px solid var(--line);font-weight:700}
@media (max-width:640px){.drow,.dgrid .drow{grid-template-columns:1fr;gap:4px}.dn{text-align:left}.dgrid{grid-template-columns:1fr}}
.steps{align-items:stretch!important}.steps{grid-auto-rows:1fr}.steps li{margin:0!important;height:auto}
.dcard,.gradio-container .block:not(.hide-container),.gradio-container .form,.cell,.card,.steps li,.sitrep,.advbox,#fv-mode{-webkit-backdrop-filter:blur(14px) saturate(160%);backdrop-filter:blur(14px) saturate(160%);border-color:var(--glass-edge)!important;box-shadow:var(--glass-shadow),inset 0 1px 0 var(--glass-hi)}
.sitrep{border:1px solid var(--glass-edge)!important;box-shadow:var(--glass-shadow),0 0 0 1px rgba(10,98,163,.25),inset 0 1px 0 var(--glass-hi)}
.gradio-container .block:has(.image-container),.gradio-container button,.gradio-container input,.gradio-container .icon-button-wrapper,.gradio-container .icon-button-wrapper button{-webkit-backdrop-filter:none;backdrop-filter:none}
.gradio-container ul.options,.gradio-container .options{background:var(--card-solid)!important;-webkit-backdrop-filter:none;backdrop-filter:none}
@media (prefers-reduced-transparency:reduce){.dcard,.gradio-container .block:not(.hide-container),.gradio-container .form,.cell,.card,.steps li,.sitrep,.advbox,#fv-mode{background:var(--card-solid)!important;-webkit-backdrop-filter:none;backdrop-filter:none}}
/* A backdrop-filter on an ancestor makes the dropdown list (position:fixed) measure from the card, so it landed off-screen. Keep the event card free of it. */
#fv-tile,.gradio-container .form:has(#fv-tile){-webkit-backdrop-filter:none!important;backdrop-filter:none!important;overflow:visible!important;background-color:var(--card-solid)!important}
:root{--r-sm:8px;--r:12px;--r-lg:16px}
.gradio-container .block,.gradio-container .form,.gradio-container fieldset.block,.gradio-container .wrap.default,.gradio-container .tab-like-container{border-radius:var(--r)!important}
.gradio-container .block:has(.image-container){overflow:hidden}
.gradio-container label.float,.gradio-container .icon-button,.gradio-container .icon-button-wrapper,.gradio-container .icon-button-wrapper button,.gradio-container [data-testid="block-label"]{border-radius:var(--r-sm)!important}
.gradio-container .image-container,.gradio-container .image-frame,.gradio-container .upload-container{border-radius:var(--r)!important}
.gradio-container input[type=number],.gradio-container .reset-button{border-radius:var(--r-sm)!important}
.gradio-container #fv-keep label,.gradio-container #fv-tile .wrap{border-radius:var(--r)!important}
.gradio-container ul.options{border-radius:var(--r)!important;overflow:hidden}
.advbox,.sr-flow,.cell,.kpi,.card,details.chip .tip{border-radius:var(--r)!important}
#hero,.sitrep{border-radius:var(--r-lg)!important}
.gradio-container div.block,.gradio-container div.block.padded,.gradio-container #fv-tile{border-radius:var(--r)!important}
.gradio-container .chip,.gradio-container details.chip,.gradio-container .token{border-radius:var(--r-sm)!important}
@media (max-width:640px){
 #fv-step{width:100%!important;max-width:100%!important;box-sizing:border-box;gap:8px}
 #fv-step button{flex:1 1 0!important;min-width:0!important;padding-left:8px!important;padding-right:8px!important}

 .gradio-container{overflow-x:hidden}
 .gradio-container .row{flex-wrap:wrap!important}
 .gradio-container .row>*{min-width:0!important;flex:1 1 100%!important;max-width:100%!important}
 #fv-bar{margin:0 0 10px!important;padding:0!important}
 #fv-mode label{padding:3px 8px!important}#fv-mode label span{font-size:.8rem!important}
 #fv-bar>*{flex:none!important}
 #fv-step{flex-wrap:nowrap!important}#fv-step>*{flex:1 1 0!important}
 #fv-keep .wrap{grid-template-columns:1fr}
 .gradio-container button{min-width:0!important}
 .tab-wrapper,.tab-container{max-width:100%}
 .tab-container{flex-wrap:wrap!important;height:auto!important}
 .tab-container button[role=tab]{flex:none!important;white-space:nowrap;min-height:40px}
 .fv-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
 .sitrep{padding:1.1rem 1rem}.sr-flow{gap:6px}
 #hero{padding:20px 16px 26px 28px!important}#hero::before{width:10px}
 #hero .hero-text h1{font-size:1.9rem;flex-wrap:wrap}#hero .hero-text p{font-size:1.05rem}
}
@media (max-width:760px){#hero .hero-art{display:none}#hero h1{font-size:2.2rem}.steps{grid-template-columns:1fr}.kpis,.kpis6{grid-template-columns:repeat(2,1fr)}.simlist{grid-template-columns:1fr}.fv-steps{flex-direction:column;gap:.4rem}.tw{grid-template-columns:1fr}.cell .big{font-size:1.2rem}}
"""
CSS += XP_CSS
CSS += GL_CSS

TOP = f"""<div id="hero"><div class="hero-text"><div class="eyebrow">Flood damage reports, fact-checked</div>
<h1><svg width="44" height="44" viewBox="0 0 44 44" aria-hidden="true"><path d="M22 3C22 3 8 18 8 28a14 14 0 0 0 28 0C36 18 22 3 22 3Z" fill="#5AB4F0"/><path d="M15 27l5 5 10-11" fill="none" stroke="#0A1824" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/></svg> FloodVerify</h1>
<p>A model writes a damage report for a flooded area. FloodVerify checks every statement against building-level evidence and keeps only what the evidence supports.</p>
</div>
<svg class="hero-art" viewBox="0 0 200 200" role="img" aria-label="A place split into nine areas, each marked as matching, not matching or cannot be checked"><rect x="8" y="8" width="56" height="56" rx="8" fill="#5CC79B" fill-opacity=".16" stroke="#5CC79B" stroke-width="2"/><text x="36" y="46" text-anchor="middle" font-size="26" font-weight="700" fill="#5CC79B" font-family="Inter,sans-serif">✓</text><rect x="72" y="8" width="56" height="56" rx="8" fill="#5CC79B" fill-opacity=".16" stroke="#5CC79B" stroke-width="2"/><text x="100" y="46" text-anchor="middle" font-size="26" font-weight="700" fill="#5CC79B" font-family="Inter,sans-serif">✓</text><rect x="136" y="8" width="56" height="56" rx="8" fill="#A9B7C1" fill-opacity=".16" stroke="#A9B7C1" stroke-width="2" stroke-dasharray="5 4"/><text x="164" y="46" text-anchor="middle" font-size="26" font-weight="700" fill="#A9B7C1" font-family="Inter,sans-serif">–</text><rect x="8" y="72" width="56" height="56" rx="8" fill="#5CC79B" fill-opacity=".16" stroke="#5CC79B" stroke-width="2"/><text x="36" y="110" text-anchor="middle" font-size="26" font-weight="700" fill="#5CC79B" font-family="Inter,sans-serif">✓</text><rect x="72" y="72" width="56" height="56" rx="8" fill="#FF8F70" fill-opacity=".16" stroke="#FF8F70" stroke-width="2"/><text x="100" y="110" text-anchor="middle" font-size="26" font-weight="700" fill="#FF8F70" font-family="Inter,sans-serif">✕</text><rect x="136" y="72" width="56" height="56" rx="8" fill="#5CC79B" fill-opacity=".16" stroke="#5CC79B" stroke-width="2"/><text x="164" y="110" text-anchor="middle" font-size="26" font-weight="700" fill="#5CC79B" font-family="Inter,sans-serif">✓</text><rect x="8" y="136" width="56" height="56" rx="8" fill="#A9B7C1" fill-opacity=".16" stroke="#A9B7C1" stroke-width="2" stroke-dasharray="5 4"/><text x="36" y="174" text-anchor="middle" font-size="26" font-weight="700" fill="#A9B7C1" font-family="Inter,sans-serif">–</text><rect x="72" y="136" width="56" height="56" rx="8" fill="#5CC79B" fill-opacity=".16" stroke="#5CC79B" stroke-width="2"/><text x="100" y="174" text-anchor="middle" font-size="26" font-weight="700" fill="#5CC79B" font-family="Inter,sans-serif">✓</text><rect x="136" y="136" width="56" height="56" rx="8" fill="#5CC79B" fill-opacity=".16" stroke="#5CC79B" stroke-width="2"/><text x="164" y="174" text-anchor="middle" font-size="26" font-weight="700" fill="#5CC79B" font-family="Inter,sans-serif">✓</text></svg><svg class="hero-wave" viewBox="0 0 900 64" preserveAspectRatio="none" aria-hidden="true"><path d="M0 34 C150 6 300 62 450 34 S750 6 900 34 V64 H0Z" fill="rgba(90,180,240,.10)"/><path d="M0 46 C170 22 320 68 470 46 S760 22 900 46 V64 H0Z" fill="rgba(90,180,240,.08)"/></svg></div>"""
STEPS = """<ol class="steps"><li><span class="n">1</span><div><b><a href="#fv-choose">Pick a flood</a></b>Or start with one of the examples.</div></li>
<li><span class="n">2</span><div><b><a href="#fv-checked">Read the checked caption</a></b>Each line says whether it matches, does not match, or cannot be checked.</div></li>
<li><span class="n">3</span><div><b><a href="#fv-report">Read the verified report</a></b>It is built only from the statements the evidence supports.</div></li></ol>"""
# Light / Dark / Auto. Sets data-fv on <html> (our colours) and .dark on <body> (Gradio's own components).
SET_MODE = """(m) => { window.__fvMode = m;
  const dark = m === 'Dark' || (m === 'Auto' && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.dataset.fv = dark ? 'dark' : 'light';
  document.body.classList.toggle('dark', dark);
  try { localStorage.setItem('fv-mode', m); } catch (e) {} }"""
LOAD_MODE = """() => { let m = 'Auto'; try { m = localStorage.getItem('fv-mode') || 'Auto'; } catch (e) {}
  const apply = (v) => { window.__fvMode = v;
    const dark = v === 'Dark' || (v === 'Auto' && matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.dataset.fv = dark ? 'dark' : 'light'; document.body.classList.toggle('dark', dark); };
  apply(m);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (window.__fvMode === 'Auto') apply('Auto'); });
  return m; }"""
HELP = (
    "One line per area, like `r1c1: many of the buildings show major damage.` Areas run `r0c0` (top left) to `r2c2` "
    "(bottom right). Words: none = 0%, few = under 10%, some = 10–30%, many = 30–60%, most = 60% or more. "
    "Try changing a word and checking again."
)
# Group members: (name, GitHub id). Add the remaining members here; each shows as a link to their profile.
TEAM = [
    ("Antaun Bobby", "Antaun"),
    ("Aravind Vinod", "aaarrvind"),
    ("Aryan Bute", "AryanBute"),
    ("Nathan Mathew", "nathanmathew24"),
    ("Vinayak Parayil Nair", "v1n4y4k04-code"),
    ("Yog Golakiya", "YogGolakiya"),
]
_team = " · ".join(
    f'<span class="mem">{esc(n)} (<a href="https://github.com/{esc(g)}" target="_blank" rel="noopener">@{esc(g)}</a>)</span>'
    for n, g in TEAM
)
FOOT = (
    f"<p class='team'><b>Built by</b> {_team}</p>"
    "<p><b>Research prototype from a Master's project.</b> Not validated for emergency decisions.</p>"
    "<p>Imagery: xBD, CC BY-NC-SA 4.0. Keep this demo private while imagery is included.</p>"
)
COLOR_MAP = {"Matches": "#CDEBDD", "Does not match": "#F7CFC4", "Cannot check": "#DDE4E7"}

ABOUT = """### What FloodVerify does
A model writes a damage caption for a flooded area, one line for each part of a 3 by 3 grid. FloodVerify splits each line
into statements, such as “many of the buildings show minor damage”, and checks every statement against the building
damage labels for that part. The verified report keeps the statements the evidence supports and leaves out the rest.

### What you see on this page
- **Maps:** the area before and after the flood, with buildings coloured by damage, and where the caption is wrong.
- **Caption, checked:** each line is tagged Matches, Does not match or Cannot check, with the reason.
- **Verified situation report:** the confirmed statements, a one-line summary, advice for residents written from them,
  and what people were posting about the same event.

### How a statement is judged
The share of damaged buildings in a part is turned into a word: none is 0%, few is under 10%, some is 10–30%, many is
30–60% and most is 60% or more. If the caption uses the same word, the statement matches. If not, it does not match. A part
with less than half of its imagery usable is not judged.

### Limits
- Small parts cannot support every word. “few” needs at least 11 buildings, “some” needs 4 and “many” needs 2. With fewer,
  the statement shows Cannot check. The slider above shows this for any number of buildings.
- Tweets are matched to a place by event and time, not by place. They are not checked against the imagery.
- The evidence is either ground-truth building labels or model-predicted buildings. The report says which one it used.
"""

with gr.Blocks(title="FloodVerify") as demo:
    with gr.Row(elem_id="fv-bar"):
        mode = gr.Radio(
            [("☀  Light", "Light"), ("☾  Dark", "Dark"), ("◐  Auto", "Auto")],
            value="Auto",
            label="Theme",
            show_label=False,
            container=False,
            elem_id="fv-mode",
        )
    gr.HTML(TOP)
    gr.HTML(STEPS)
    with gr.Tabs():
        with gr.Tab("Check a report"):
            if EXAMPLES:
                gr.HTML('<h2 class="panel-title">Start with an example</h2>')
                with gr.Row():
                    ex_btns = [(gr.Button(label), t) for label, t in EXAMPLES.items()]
            note = gr.HTML()
            gr.HTML('<h2 class="panel-title" id="fv-choose">Choose a flood</h2>')
            with gr.Row(equal_height=False, elem_id="fv-controls"):
                choices = tile_choices()
                with gr.Column(scale=3, min_width=300):
                    ev_choices = event_choices()
                    first_tile = choices[0][1] if choices else None
                    first_tile = default_tile(tile_event(first_tile)) if first_tile else None
                    event_dd = gr.Dropdown(
                        ev_choices,
                        value=tile_event(first_tile) if first_tile else None,
                        label="Flood event",
                        info="Choose a flood. The demo shows one affected place from it.",
                        interactive=True,
                        elem_id="fv-tile",
                        filterable=False,
                    )
                    tile = gr.Dropdown(
                        choices, value=first_tile, visible=False, elem_id="fv-tile-id"
                    )
                    with gr.Row(elem_id="fv-step"):
                        prev_btn = gr.Button("← Previous place", size="sm")
                        next_btn = gr.Button("Next place →", size="sm")
                alpha = gr.Slider(
                    0,
                    1,
                    0.65,
                    step=0.05,
                    label="Colour strength",
                    scale=2,
                    min_width=240,
                    info="How solid the damage colours are on the photo.",
                )
                keep = gr.CheckboxGroup(
                    [n for n, _ in LAYERS.values()],
                    label="Show only these buildings",
                    scale=3,
                    min_width=300,
                    info="Tick one or more. Leave empty to show every building.",
                    elem_id="fv-keep",
                )
            place_html = gr.HTML(place_note(first_tile))
            gr.HTML('<h2 class="panel-title" style="margin-top:.8rem">Where it happened</h2>')
            globe = gr.HTML(
                globe_value(first_tile), html_template=GL_TEMPLATE, js_on_load=GL_JS_FULL
            )
            pick = gr.Textbox(
                elem_id="fv-pick",
                elem_classes="fv-offscreen",
                label="Globe selection",
                show_label=False,
                container=False,
            )
            gr.HTML('<h2 class="panel-title" style="margin-top:.8rem">Read the maps</h2>')
            with gr.Row():
                img_pre = gr.Image(
                    label="Before the flood",
                    interactive=False,
                    buttons=["fullscreen", "download"],
                    alt_text="Satellite view of the place before the flood, divided into a 3 by 3 grid.",
                )
                img_post = gr.Image(
                    label="After the flood",
                    interactive=False,
                    buttons=["fullscreen", "download"],
                    alt_text="The place after the flood, with buildings coloured by damage level.",
                )
                img_verdict = gr.Image(
                    label="Where the caption is wrong",
                    interactive=False,
                    buttons=["fullscreen", "download"],
                    alt_text="The place with each area marked as matching, not matching or cannot be checked.",
                )
            legend = gr.HTML()
            sitrep = gr.HTML()
            kpis = gr.HTML()
            gr.HTML('<h2 class="panel-title" style="margin-top:.8rem">Explore the place</h2>')
            explore = gr.HTML(js_on_load=XP_JS)
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    gr.HTML('<h2 class="panel-title">Counts per area</h2>')
                    grid = gr.HTML()
                with gr.Column(scale=6):
                    gr.HTML('<h2 class="panel-title" id="fv-checked">The caption, checked</h2>')
                    shown = gr.HighlightedText(
                        label="Each line of the caption",
                        color_map=COLOR_MAP,
                        combine_adjacent=False,
                        show_legend=False,
                        interactive=False,
                    )
                    summary = gr.HTML()
                    with gr.Accordion("Edit the caption and check again", open=False):
                        report = gr.Textbox(lines=7, label="Caption (one line per area)", info=HELP)
                        btn = gr.Button("Check statements", variant="primary")
                    with gr.Accordion("Why each statement got its result", open=True):
                        cards = gr.HTML()
            gr.HTML(
                '<h2 class="panel-title" style="margin-top:1rem">Places with a similar damage pattern</h2>'
            )
            sim = gr.HTML()
            sim_ids = gr.State([])
            with gr.Row():
                sim_btns = [gr.Button(visible=False, size="sm") for _ in range(3)]
        with gr.Tab("Public posts"):
            gr.HTML('<h2 class="panel-title">What people were saying about this event</h2>')
            tw = gr.HTML()
        with gr.Tab("Evidence and advice"):
            gr.HTML('<h2 class="panel-title">Evidence for this place</h2>')
            table = gr.Dataframe(
                headers=["Area", "Buildings", "Damaged", "Usable imagery", "Confidence"],
                interactive=False,
            )
            gr.HTML(
                '<h2 class="panel-title" style="margin-top:1rem">Advice for residents (full text)</h2>'
            )
            adv_why = gr.HTML()
            with gr.Row():
                adv_en = gr.HTML(elem_id="fv-adv-en")
                adv_ar = gr.HTML(elem_id="fv-adv-ar")
        with gr.Tab("Dashboard"):
            for _h, _body in dashboard_blocks():
                gr.HTML(f'<h2 class="panel-title">{esc(_h)}</h2>' + _body)
        with gr.Tab("How it works"):
            gr.HTML('<h2 class="panel-title">Why can\'t some statements be checked?</h2>')
            n_slider = gr.Slider(1, 40, value=3, step=1, label="Buildings in an area")
            explain = gr.HTML(explainer_html(3))
            gr.Markdown(ABOUT)
    gr.HTML(f'<div class="foot">{FOOT}</div>')

    rendered = [
        img_pre,
        img_post,
        img_verdict,
        legend,
        shown,
        grid,
        summary,
        cards,
        sitrep,
        kpis,
        explore,
    ]
    tile.change(
        on_tile,
        [tile, alpha, keep],
        [note, report, *rendered, adv_why, adv_en, adv_ar, tw, table, sim, sim_ids, *sim_btns],
    )
    btn.click(on_check, [tile, report, alpha, keep], rendered)
    alpha.release(on_check, [tile, report, alpha, keep], rendered)
    keep.change(on_check, [tile, report, alpha, keep], rendered)
    n_slider.change(explainer_html, n_slider, explain)
    event_dd.input(default_tile, event_dd, tile)
    tile.change(
        lambda t: (tile_event(t) if t else None, place_note(t), globe_value(t)),
        tile,
        [event_dd, place_html, globe],
    )
    pick.change(pick_target, pick, [tile, pick])
    prev_btn.click(lambda t: step_tile(t, -1), tile, tile)
    next_btn.click(lambda t: step_tile(t, 1), tile, tile)
    for i, b in enumerate(sim_btns):
        b.click(lambda ids, i=i: ids[i], sim_ids, tile)
    for b, t in ex_btns if EXAMPLES else []:
        b.click(lambda t=t: t, None, tile)
    demo.load(
        on_tile,
        [tile, alpha, keep],
        [note, report, *rendered, adv_why, adv_en, adv_ar, tw, table, sim, sim_ids, *sim_btns],
    )
    demo.load(
        lambda t: (tile_event(t) if t else None, place_note(t), globe_value(t)),
        tile,
        [event_dd, place_html, globe],
    )
    mode.change(None, mode, None, js=SET_MODE)
    demo.load(None, None, mode, js=LOAD_MODE)

if __name__ == "__main__":
    kw = {
        "theme": gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
        "css": CSS,
    }
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        demo.theme, demo.css = kw["theme"], kw["css"]
        demo.launch(share=True)
    else:
        demo.launch(**kw, share=True)
