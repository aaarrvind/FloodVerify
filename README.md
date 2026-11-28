# FloodVerify

**An evidence-grounded NLP pipeline for verified flood damage reports and safety advice from satellite images.**

Before and after satellite images go in. A damage report comes out, with every sentence checked against image-derived evidence, plus a priority list and an English and Arabic safety advisory. Humans make the final decision.

FloodVerify is the group project for MAAI901 (Master of Applied AI, University of Wollongong in Dubai). Vision-language models and large language models can describe post-disaster imagery in minutes, but they can also report damage that is not there or suggest unsafe measures. FloodVerify limits that risk in three ways:

1. **Constrained captioning.** A vision-language model describes each cell of a cropped tile using a restricted prompt, so its output can be parsed into atomic damage claims.
2. **Deterministic verification.** A checker with no learned parameters compares each claim with structured evidence and returns `supported`, `contradicted` or `not_verifiable`.
3. **Retrieval, not generation, for guidance.** Advice comes from NCEMA, ready.gov and FEMA passages retrieved with a multilingual embedding model. The language model only phrases the cited bullets.

## Pipeline

| Component | Model / method | Trained? |
|---|---|---|
| Evidence model | Siamese segmentation (U-Net / SegFormer) | Yes |
| Captioner | Qwen3-VL-4B-Instruct | No (zero-shot) |
| Claim parser | Regular expressions (quantity word + damage class) | No |
| Checker | Rule-based quantity bins (v2) | No |
| Tweet classifier | `cardiffnlp/twitter-roberta-base`, 11 HumAID classes | Yes |
| Usefulness filter | P(actionable) ≥ 0.75 | No (threshold) |
| Retrieval | `BAAI/bge-m3`, top-3 | No |
| Advisory generation | Qwen3-8B, cited EN/AR bullets | No |
| Tweet–tile linking | Event + time rule | No |

The computer-vision and NLP sides share only one thing: the evidence file described in [docs/CONTRACT.md](docs/CONTRACT.md) and validated by `evidence/schema/evidence.schema.json`. The CV models write one evidence JSON per tile, and the NLP components read those files, so the two parts can be developed independently. Sample evidence files are in `evidence/samples/`.

## Quick start

```bash
git clone https://github.com/aaarrvind/FloodVerify.git
cd FloodVerify
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
make test
```

Other commands: `make lint` runs ruff on `src` and `tests`, and `make demo` starts the Gradio app in `demo/`.

The experiments were run in Google Colab. The notebooks are listed below, and the training notebook installs the package with `pip install -q -e . --no-deps jsonschema`.

## Repository layout

```
src/floodsitrep/
  common/      shared contract: evidence schema, types, config
  data/        xBD tiling, splits, labels -> evidence
  cv/          siamese U-Net and SegFormer models, training, evaluation, evidence export
  nlp/         captioning, claim extraction, deterministic checker
  advisory/    HumAID classifier, bilingual retrieval, EN/AR advisory
  pipeline/    end-to-end glue
configs/       cv/ and nlp/ YAML settings
notebooks/     m1 tiles and evidence, m2 training, m3 evaluation and export,
               m4 captions and claim checking, m5 HumAID classifier
evidence/      JSON schema and sample evidence files
demo/          Gradio app for the Hugging Face Space
docs/          CONTRACT.md, OWNERSHIP.md, TIMELINE.md
data/          data notes only; no data is stored in git
reports/       cv/, nlp/ and paper/
tests/         unit tests for the contract, data, CV, NLP and pipeline code
```

## Data

Raw data, tiles, evidence folders and model checkpoints are **not** stored in this repository. See [data/README.md](data/README.md) for where they live, how the xBD splits are built and how to rebuild the evidence files. Download the datasets from the sources below and follow their terms.

| Dataset | Source | Licence / terms | Link |
|---|---|---|---|
| xBD (flood subset) | Carnegie Mellon University SEI and Defense Innovation Unit | CC BY-NC-SA 4.0 (non-commercial) | https://xview2.org |
| HumAID | Qatar Computing Research Institute (QCRI), HBKU | Research use only | https://crisisnlp.qcri.org/humaid_dataset.html |
| RSCC | Zhejiang University (Chen et al.) | Follow the xBD licence; research use only (see the RSCC repository) | https://github.com/Bili-Sakura/RSCC |
| NCEMA guidance | National Emergency Crisis and Disaster Management Authority (UAE) | Open Data Reuse Terms | https://www.ncema.gov.ae |
| ready.gov / FEMA guidance | Federal Emergency Management Agency (US) | Public information; credit FEMA | https://www.ready.gov/floods |

Models are downloaded from Hugging Face on first use. Qwen3-VL-4B-Instruct is Apache 2.0. Qwen2.5-VL-3B-Instruct, which we used only as a comparison captioner, is under the Qwen Research Licence (research and non-commercial use).

Events are split whole, never by tile: Hurricanes Harvey and Matthew and the Midwest floods for training, Florence for validation, and the 2015 Nepal floods as an unseen test set.

## Headline results

All numbers are from the project report.

- **Claim verification.** On 112 claims from a 50-tile run, checker v1 judged 86% of claims and v2 judged 23%. The stricter v2 marks a claim `not_verifiable` when a cell has too few buildings for the claimed quantity word to occur.
- **Human agreement.** On 49 claims with a clear human label, the checker agreed with the annotators 71.4% of the time (95% Wilson interval 0.58 to 0.82), against 61.2% to 65.3% for three open-weight LLM judges.
- **Tweet classification.** The fine-tuned twitter-roberta-base model reached a test macro-F1 of 0.750, against 0.714 for a TF-IDF baseline.
- **Actionable-tweet flagging.** A threshold of 0.75 gave precision 0.848, recall 0.873 and F1 0.860 on the test split (15,160 tweets).
- **Predicted evidence.** The segmentation models find where buildings are (agreement 0.95) but count only about 45% of labelled buildings, mostly on the unseen Nepal event, so about half of label-supported claims survive on predicted evidence.

## Limitations

- The captioner was tested on only 50 tiles, and the smaller Qwen2.5-VL-3B collapsed on 20 tiles.
- Human verification used two annotators on 81 claims, and many disagreements were about regions that were not visible.
- The transformer's event-holdout test covers only two flood events, so it says little about generalisation across hazard types.
- The usefulness score is a proxy built from HumAID classes, not human usefulness labels.
- xBD labels building damage after floods and storms, not flood extent. RSCC captions are machine-written and noisy, so they are never treated as ground truth.
- The advisory is only as complete as the guidance passages it retrieves from (154 evaluated, 210 in the demo).

## AI use disclosure

We used AI tools (Claude) during this project to help scaffold and debug the code, and to improve the written content of the report, making it grammatically correct, clear and well formatted. We ran all experiments ourselves in Google Colab and checked every number in this report against our own outputs.

## Contributing and ownership

Each member owns one folder tree, listed in [docs/OWNERSHIP.md](docs/OWNERSHIP.md), so pull requests rarely collide. Work on a branch named `m<N>/<short-name>`, run `make lint test` before pushing, and open a pull request. Do not change `evidence/schema/evidence.schema.json` or `src/floodsitrep/common/` without an approved review. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Team

MAAI901 group project, University of Wollongong in Dubai: Antaun Bobby, Aravind, Aryan, Nathan, Vinayak Parayil Nair and Yog.

## Licence

The code in this repository is released under the [MIT Licence](LICENSE). The datasets and models it uses keep their own licences, listed above.
