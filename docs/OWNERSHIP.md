# Ownership (5 members)

Replace `@memberN` with GitHub handles in `.github/CODEOWNERS` and here.

| # | Role | Owns (folders) | Deliverables | Report / paper duty |
|---|------|----------------|--------------|---------------------|
| 1 | Data, contract, integration | `src/floodsitrep/common/`, `data/`, `evidence/`, `pipeline/`, `.github/`, `docs/`, `reports/paper/` | xBD tiles + 3x3 grid labels, evidence schema frozen Fri 13 Nov, label-to-evidence generator, CI, end-to-end pipeline | Assembles the single combined paper; CV report data section |
| 2 | CV models | `src/floodsitrep/cv/models/`, `cv/train.py`, `configs/cv/` | M0 early-fusion U-Net, M1 siamese U-Net (ResNet-34), M2 siamese SegFormer, checkpoints | CV report: methods, models, training |
| 3 | CV evaluation | `src/floodsitrep/cv/eval/`, `cv/export_evidence.py`, `reports/cv/` | Metrics (IoU, F1, xView2 score), temperature scaling and ECE, 1 m / 3 m ladder, BRIGHT / Dubai / WorldFloods eval, predicted evidence export | CV report: experiments, results, limitations |
| 4 | NLP verification | `src/floodsitrep/nlp/`, `configs/nlp/` | Qwen2.5-VL captions, claim extraction, quantifier bins, checker verdicts, unsupported-claim rate, human check | NLP report: methods and verification results |
| 5 | Advisory and demo | `src/floodsitrep/advisory/`, `demo/`, `reports/nlp/` | HumAID classifier, retrieval, EN/AR advisory, priority list, Gradio demo on a Hugging Face Space | NLP report: advisory, Arabic, demo, ethics |

## Who depends on whom
- 2 -> 3: checkpoints (`checkpoints/` on shared drive, never in git).
- 3 -> 4: predicted evidence JSON (Gate 2, Wed 18 Nov). Until then 4 uses label-derived evidence from member 1.
- 4 -> 5: verified claims JSON (`nlp/schema.py`). 5 reads it for advisory and demo.
- 1 wires all of it in `pipeline/`.

## Parallel working rules
1. Work only inside your folders. Touching another member's folder needs a PR with them as reviewer.
2. Branches: `m<N>/<short-name>`, e.g. `m2/siamese-unet`. Never push to `main`.
3. Small PRs, merged within a day. One approval from the CODEOWNER.
4. Code against `common/interfaces.py` stubs, not against each other's code.
5. Notebooks go in `notebooks/m<N>/`. Move working code into `src/` before opening a PR.
6. Large files (data, checkpoints, outputs) never go in git; see `data/README.md`.
