# The shared contract

## Evidence file (CV -> NLP)
One JSON file per tile in `evidence/`. Schema: `evidence/schema/evidence.schema.json`.

- `source` is `labels` (built from xBD ground truth, available now) or `predicted` (from the CV model, Gate 2).
- Grid is 3x3; each cell has `buildings`, `class_fraction` (none / minor / major / destroyed, summing to 1) and `confidence` (temperature-scaled).
- Quantifier bins used by the checker: few < 10%, some 10-30%, many 30-60%, most > 60%.

## Claims file (NLP verification -> advisory and demo)
List of atomic claims with a verdict: `supported`, `contradicted` or `not_verifiable`. Types in `src/floodsitrep/nlp/schema.py`.

## Changing the contract
Open a PR touching `common/` or the schema, tag all five members, merge only with member 1's approval.

## Evidence rules (added 14 Nov)
- Cells with no buildings have all-zero class fractions.
- `valid_fraction` is the share of the cell with real imagery. Below 0.5 the cell has no usable imagery, and the checker must return `not_verifiable` for any claim about it.
- Label evidence counts buildings from the xBD label polygons (exact, each building once, by centroid). Predicted evidence counts buildings from the model's mask, so its counts are approximate and can merge touching buildings.

## Checker rules (added 19 Nov, member 1)
Checker v2. Shares of whole buildings are k/n, so with n buildings some quantity words cannot occur: `few` needs at least 11 buildings, `some` at least 4, `many` at least 2. `none` and `most` are always possible. A claim whose quantity word cannot occur at the cell's building count (usable cells only) is `not_verifiable`, never `contradicted` (`reachable` in `src/floodsitrep/nlp/checker.py`).

- Supported verdicts are the same in v1 and v2. Only some contradicted verdicts become not verifiable.
- v2 was added after Gate 2 and after the single-annotator human-check dry run. Results should state which version scored them, and report v1 as well where claims were scored before 19 Nov.
- Effect on the 50-tile Qwen3-VL-4B run: 112 claims, v1 4 supported / 92 contradicted / 16 not verifiable, v2 4 / 22 / 86.

## Event split (added 15 Nov, member 1)
Tiles are split by whole event, set in `src/floodsitrep/data/splits.py` and written to
`tiles.csv` (column `role`): train = hurricane-harvey, hurricane-matthew, midwest-flooding;
val = hurricane-florence; test = nepal-flooding. Midwest is not used for validation because only
about 2% of its buildings are major or destroyed.
