# FloodVerify

Verified flash-flood situation reports: before/after satellite images -> damage report ->
every sentence checked against image-derived evidence -> priority list and EN/AR advisory.
Humans make the final decision.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
make test
```

## Who owns what
See [docs/OWNERSHIP.md](docs/OWNERSHIP.md). Each member owns one folder tree, so pull requests rarely collide.

## The one rule
The only thing the CV and NLP sides share is the evidence file: [docs/CONTRACT.md](docs/CONTRACT.md).
Do not change `evidence/schema/evidence.schema.json` or `src/floodsitrep/common/` without a PR approved by member 1.

## Layout
```
src/floodsitrep/
  common/      shared contract: evidence schema, types, config (member 1)
  data/        xBD tiling, labels -> evidence (member 1)
  cv/          models M0-M2, training, evidence export (members 2, 3)
  nlp/         captioning, claim extraction, checker (member 4)
  advisory/    HumAID classifier, retrieval, EN/AR advisory (member 5)
  pipeline/    end-to-end glue (member 1)
demo/          Gradio app for the Hugging Face Space (member 5)
evidence/      JSON schema and sample evidence files (member 1)
reports/       cv/, nlp/, paper/ (see ownership)
```
