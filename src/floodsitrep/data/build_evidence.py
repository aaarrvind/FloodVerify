"""Build label-derived evidence and tiles.csv for every manifest row. Owner: member 1.

Run on a computer that has the xBD folders (no GPU needed):

    python -m floodsitrep.data.build_evidence --xbd-root /path/to/xbd \
        --manifest /path/to/xbd/rscc_flood_manifest.csv \
        --out-evidence /path/to/xbd/processed/evidence_labels \
        --out-tiles /path/to/xbd/processed/tiles.csv \
        --schema evidence/schema/evidence.schema.json

Safe to re-run: existing evidence files are skipped unless --force is given.
Needs numpy and pillow (valid_fraction) and jsonschema (optional, for --schema).
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from floodsitrep.data.labels_to_evidence import evidence_from_label, valid_fractions, write_evidence
from floodsitrep.data.splits import role_of
from floodsitrep.data.tiling import read_manifest, source_paths, tile_name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--xbd-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-evidence", required=True)
    ap.add_argument("--out-tiles", required=True)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N rows")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    rows = read_manifest(a.manifest)
    if a.limit:
        rows = rows[:a.limit]
    validator = None
    if a.schema:
        import jsonschema
        validator = jsonschema.Draft202012Validator(json.loads(Path(a.schema).read_text()))

    out_dir = Path(a.out_evidence)
    label_cache = {}
    done, missing, bad = [], [], []
    stats = Counter()
    for i, r in enumerate(rows, 1):
        name = tile_name(r)
        p = source_paths(a.xbd_root, r)
        target = out_dir / f"{name}.json"
        if not all(p[k].exists() for k in ("pre", "post", "label")):
            missing.append(name)
            continue
        if target.exists() and not a.force:
            done.append(r)
            continue
        key = (r["split"], r["event"], r["tile_id"])
        if key not in label_cache:
            label_cache[key] = json.loads(p["label"].read_text(encoding="utf-8"))
        vf = valid_fractions(p["pre"], p["post"], r["x0"], r["y0"])
        ev = evidence_from_label(label_cache[key], name, r["x0"], r["y0"], vf)
        if validator is not None:
            errs = [e.message for e in validator.iter_errors(ev)]
            if errs:
                bad.append((name, errs[0]))
                continue
        write_evidence(ev, out_dir)
        done.append(r)
        stats["built"] += 1
        if sum(c["buildings"] for c in ev["cells"]) == 0:
            stats["no_buildings"] += 1
        if any(c["valid_fraction"] < 0.5 for c in ev["cells"]):
            stats["has_unusable_cell"] += 1
        if len(label_cache) > 200:
            label_cache.clear()
        if i % 1000 == 0:
            print(f"{i}/{len(rows)} rows", flush=True)

    Path(a.out_tiles).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_tiles, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tile_id", "event", "xbd_split", "part", "role"])
        for r in done:
            w.writerow([tile_name(r), r["event"], r["split"], r["part"], role_of(r["event"])])

    print(f"rows in manifest: {len(rows)}")
    print(f"evidence present: {len(done)}  newly built: {stats['built']}")
    print(f"missing source files: {len(missing)}  schema errors: {len(bad)}")
    print(f"tiles with no buildings: {stats['no_buildings']}  "
          f"tiles with a cell below 0.5 valid: {stats['has_unusable_cell']}")
    roles = Counter((role_of(r["event"]), r["event"]) for r in done)
    for (role, ev), n in sorted(roles.items(), key=str):
        print(f"  {role:5s} {ev:20s} {n}")
    for name, msg in bad[:5]:
        print("SCHEMA ERROR", name, msg)
    for name in missing[:5]:
        print("MISSING", name)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
