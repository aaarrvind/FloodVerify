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
