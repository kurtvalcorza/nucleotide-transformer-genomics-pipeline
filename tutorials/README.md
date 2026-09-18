# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/tutorials/nucleotide_transformer_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-InstaDeepAI%2Fnucleotide--transformer--v2--50m--multi--species-ffcc4d?style=flat)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> [!IMPORTANT]
> The model weights are **CC BY-NC-SA 4.0 (non-commercial)**, and the default path **executes the checkpoint's own Python** (`trust_remote_code=True`) after digest-verifying it. Both facts are stated in the notebook before the relevant cell runs. Read them before you run anything.

Notebook specification: **DIMER Notebook Specification 2.0** — the notebook is **standalone** (§4) and declares its
profile and pedagogical mode (§3.4). Because this row is on licence HOLD it ships no DIMER pipeline package, so the
notebook carries its own helper code in its cells rather than embedding a repository module. It is emitted by
`tools/make_notebook.py` — a plain emitter for this repository, **not** the fleet's `build_notebook.py` generator —
so that the inline manifest and the dependency pins cannot drift. Do not edit the notebook by hand; edit
`tools/make_notebook.py` and regenerate (`--check` is enforced by CI).

| Notebook | Profile | Mode | Carrier | Capability | Default runtime | Sample | BYOD | Run-all | Release status |
|---|---|---|---|---|---|---|---|---|---|
| `nucleotide_transformer_colab.ipynb` | `TASK-INFERENCE` | `GUIDED` | standalone (emitted; no repository package) | 6-mer tokenization of DNA and its single-base fallback; 512-dimensional mean-pooled sequence embeddings; a nearest-centroid probe on a composition-matched synthetic task against majority-class and GC-content baselines; one masked-token prediction; exported embeddings, probe report and provenance | CPU float32 (CUDA used automatically when available) | automatic (deterministic 48-sequence generator, no download) | FASTA or `id,sequence,label` CSV via the `USE_BYOD` form gate | Kaggle Tesla T4 clean-runtime PASS on `f9dc86f` / blob `9f67399f7bcf`: 9/9 cells, 218.6 s | **Hold** — technical run passed; licence and remote-code gates in `../MODEL_CARD.md` remain unresolved |

## Conformance notes

- **Standalone (ST1–ST8):** the default path performs no clone, no repository install and no repository import; the helper functions (`stage_missing_files`, `verify_snapshot`, `validate_sequences`, `generate_sample`, `embed`) are defined in the notebook's own cells, which is what ST5 permits when there is no package to carry. The inline manifest and the `requirements.txt` pins are emitted from the repository, and `metadata.dimer.generated_from` records the emitter and the pinned model identity.
- **Remote code (MOD6), stated rather than hidden:** the notebook declares `remote_code_executed: true` in its metadata, warns in its header, and devotes Section 4 to why `trust_remote_code=True` is required here (bias-free SwiGLU feed-forward; no `model_type`), showing the native loader's refusal before performing the remote-code load. The two Python files are manifest entries verified in Section 3 **before** they are imported, and `tools/check_assets.py` fails if that ordering breaks.
- **Model acquisition (MOD1–MOD8):** `stage_missing_files` fetches only absent manifest entries from the Hub at the immutable revision `81b29e5786726d891dbf929404ef20adca5b36f1` (never `main`); `verify_snapshot` re-hashes all 8 files and raises on the first mismatch; there is no fallback to a different download.
- **Licence (§10 DAT6 and beyond):** the non-commercial condition is stated in the header, given its own section before the weights are downloaded, restated in the closing section, and recorded in the exported provenance JSON. The notebook does not attempt to enforce it and says so.
- **Sample data (DAT1–DAT9):** generated in code with a fixed seed, no download, no upload. The two classes share an identical base multiset by construction and the cell **asserts** that before any conclusion is drawn from the comparison; the sample is labelled as synthetic tutorial evidence, not a genomics benchmark.
- **Evaluation (EVAL2/EVAL6/EVAL9/EVAL10):** an embedding has no intrinsic metric, so the notebook attaches the smallest honest downstream task — a nearest-centroid probe that trains nothing — on a seeded stratified split, reported against a majority-class baseline and a GC-content baseline fitted on the training half only. The cosine margin is reported next to the accuracy because the margin is small; when the user's data carries no labels the probe reports `not-measurable` and names what would be needed.
- **Score semantics (UNC1/UNC2):** masked-token outputs are raw logits, labelled as such and never presented as probabilities; embeddings are labelled as representations, not predictions.
- **Ceilings (VAL6):** `MAX_BASES = 6000` (1,000 tokens — the length the upstream card says the model was trained at), `MAX_SEQUENCES = 64`, the `ACGTN` alphabet and the 24-base minimum are printed with the input manifest before the model runs, and three deliberate rejection probes (lowercase, too short, gap character) are executed so the failure messages are visible.
- **BYOD (DAT10–DAT19):** `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog; when enabled, FASTA or CSV input flows through the same validation, embedding, probe and export cells. Labels are optional for embedding and required for the probe, which the notebook states before the upload.
- `tools/check_assets.py` performs source checks only. It does not satisfy the clean-runtime execution requirement, and in this repository even a clean hosted run would not clear the licence and remote-code gates — see `../docs/release-verification.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
