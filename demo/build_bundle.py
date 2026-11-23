"""Build demo/data files from your own outputs. Owner: member 5. Run in Colab or locally.

    python demo/build_bundle.py --captions claims_verdicts.csv --evidence-dir <evidence folder> \\
        [--xbd-root <xbd folder> --manifest <xbd>/rscc_flood_manifest.csv] \\
        [--tweet-stats tweet_stats.json] [--tweet-examples tweet_examples.json] [--linked linked_tweets_demo.csv]

--captions   claims_verdicts.csv from the 50-tile run (tile_id, cell, text, ...), or a CSV with tile_id, report_text.
             Verdict columns are ignored: the checker is re-run on the evidence you give.
--xbd-root   optional. With --manifest it renders before, after and damage-overlay images for each tile into
             demo/data/images. xBD is CC BY-NC-SA 4.0 and must not be put in a public place: keep the images out of
             git (demo/data/images is in .gitignore) and use them only locally or in a private Space.
--tweet-stats  JSON from export_tweet_stats.py (counts only, no text).
--tweet-examples  JSON from export_tweet_examples (tweet text, links and @names removed). Private Space only.
--guidance-passages  optional. passages.csv from notebook 03. With it, advice is built for every tile with advise() from
             floodsitrep.advisory (retrieval over BAAI/bge-m3). Needs sentence-transformers and the model download
             (about 2 GB). Vectors are cached in --guidance-vectors (default: passages_bge_m3.npy next to the csv).
             The advice is the retrieved guidance text itself: keep the Space private until the licence questions
             in the retrieval handoff are answered. Without this option the demo says that advice is not switched on.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

ADVISE = None  # set to True after calling set_backends(...) in this process

DAMAGE_FILL = {"no-damage": None, "minor-damage": (243, 207, 168), "major-damage": (217, 128, 60),
               "destroyed": (138, 53, 17)}
MASK_CODE = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}  # 0 = not a building
OUTLINE = {"no-damage": (255, 255, 255), "minor-damage": (243, 207, 168), "major-damage": (240, 150, 70),
           "destroyed": (200, 70, 25)}


def render_images(xbd_root, manifest, tile_ids, out_dir):
    """Write {tile}_pre.jpg, _post.jpg, _damage.jpg (post image with building outlines by damage) and _mask.png
    (one value per pixel: 0 not a building, 1 undamaged, 2 minor, 3 major, 4 destroyed; the demo colours it live)."""
    from PIL import Image, ImageDraw

    from floodsitrep.data.labels_to_evidence import parse_wkt
    from floodsitrep.data.tiling import crop_box, read_manifest, source_paths, tile_name

    rows = {tile_name(r): r for r in read_manifest(manifest)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for tid in tile_ids:
        row = rows.get(tid)
        if row is None:
            print("not in manifest:", tid)
            continue
        paths = source_paths(xbd_root, row)
        if not (paths["pre"].exists() and paths["post"].exists()):
            print("image missing:", tid)
            continue
        box = crop_box(row)
        pre = Image.open(paths["pre"]).convert("RGB").crop(box)
        post = Image.open(paths["post"]).convert("RGB").crop(box)
        overlay = Image.new("RGBA", post.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        mask = Image.new("L", post.size, 0)
        mdraw = ImageDraw.Draw(mask)
        polys = []
        label = json.loads(paths["label"].read_text()) if paths["label"].exists() else {"features": {"xy": []}}
        for feat in label["features"]["xy"]:
            sub = feat["properties"].get("subtype")
            if sub not in OUTLINE:
                continue
            pts = [(x - box[0], y - box[1]) for x, y in parse_wkt(feat["wkt"])]
            if len(pts) < 3:
                continue
            fill = DAMAGE_FILL[sub]
            draw.polygon(pts, fill=(*fill, 110) if fill else None, outline=(*OUTLINE[sub], 255))
            polys.append((MASK_CODE[sub], pts))
        for code, pts in sorted(polys, key=lambda t: t[0]):  # worst damage drawn last, so it wins where outlines overlap
            mdraw.polygon(pts, fill=code)
        mask.save(out_dir / f"{tid}_mask.png")
        damage = Image.alpha_composite(post.convert("RGBA"), overlay).convert("RGB")
        for name, img in (("pre", pre), ("post", post), ("damage", damage)):
            img.save(out_dir / f"{tid}_{name}.jpg", quality=88)
        done += 1
    print(f"{done} tiles rendered to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--evidence-dir", default="evidence/samples",
                    help="folder with <tile_id>.json, e.g. xbd/processed/evidence_labels or the predicted evidence")
    ap.add_argument("--xbd-root")
    ap.add_argument("--manifest")
    ap.add_argument("--tweet-stats")
    ap.add_argument("--tweet-examples")
    ap.add_argument("--linked")
    ap.add_argument("--guidance-passages", help="passages.csv from notebook 03; switches advice on")
    ap.add_argument("--guidance-vectors", help="cached vectors (.npy); default passages_bge_m3.npy next to the csv")
    ap.add_argument("--min-score", type=float, help="retrieval cutoff (default: the module's MIN_SCORE)")
    ap.add_argument("--device", help="cpu, cuda or mps for the embedding model (default: automatic)")
    ap.add_argument("--out", default="demo/data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    caps, lines = {}, {}
    with open(a.captions, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "report_text" in r:
                caps[r["tile_id"]] = r["report_text"]
            else:
                lines.setdefault(r["tile_id"], []).append(f"{r['cell']}: {r['text']}")
    caps.update({t: "\n".join(sorted(v)) for t, v in lines.items()})
    ev_out = out / "evidence"
    ev_out.mkdir(exist_ok=True)
    missing = [t for t in caps if not (Path(a.evidence_dir) / f"{t}.json").exists()]
    caps = {t: v for t, v in caps.items() if t not in missing}
    for t in caps:
        shutil.copy(Path(a.evidence_dir) / f"{t}.json", ev_out / f"{t}.json")
    if missing:
        print(f"{len(missing)} tiles skipped, no evidence file in {a.evidence_dir}: {missing[:3]} ...")
    (out / "captions.json").write_text(json.dumps(caps, ensure_ascii=False, indent=1))
    if a.xbd_root and a.manifest:
        render_images(a.xbd_root, a.manifest, list(caps), out / "images")
    if a.tweet_stats:
        shutil.copy(a.tweet_stats, out / "tweet_stats.json")
    if a.tweet_examples:
        shutil.copy(a.tweet_examples, out / "tweet_examples.json")
        print("tweet_examples.json contains tweet text: keep the Space private.")
    advise_on = ADVISE
    if a.guidance_passages:
        from floodsitrep.advisory import retrieve as R
        from floodsitrep.advisory.generate import make_retriever, set_backends

        passages = R.load_passages(a.guidance_passages)
        vec_path = a.guidance_vectors or str(Path(a.guidance_passages).with_name("passages_bge_m3.npy"))
        print(f"{len(passages)} guidance passages; loading {R.MODEL_NAME} (first run downloads about 2 GB)")
        model = R.load_model(device=a.device)
        vectors = R.load_or_embed(model, passages, vec_path)
        set_backends(retrieve_fn=make_retriever(model, vectors, passages, min_score=a.min_score))
        advise_on = True
    if advise_on:
        from floodsitrep.advisory.generate import advise
        from floodsitrep.common.interfaces import load_evidence
        from floodsitrep.nlp.checker import verify

        adv = {}
        for tid, text in caps.items():
            claims = verify(text, load_evidence(Path(a.evidence_dir) / f"{tid}.json"))
            adv[tid] = {lang: advise(claims, lang) for lang in ("en", "ar")}
        (out / "advisories.json").write_text(json.dumps(adv, ensure_ascii=False, indent=1))
        with_guidance = sum("No guidance found" not in v["en"] for v in adv.values())
        print(f"advice written for {len(adv)} tiles; {with_guidance} have guidance, "
              f"{len(adv) - with_guidance} have none (no confirmed statement, or no passage close enough)")
        print("The advice contains the guidance passages themselves: keep the Space private (licences).")
    if a.linked:
        shutil.copy(a.linked, out / "linked_tweets.csv")
    print(f"{len(caps)} tiles with captions written to {out}")


if __name__ == "__main__":
    main()
