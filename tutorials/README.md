# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/tutorials/nucleotide_transformer_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-InstaDeepAI%2Fnucleotide--transformer--v2--50m--multi--species-ffcc4d?style=flat)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> [!IMPORTANT]
> The model weights are **CC BY-NC-SA 4.0 (non-commercial)** — and so is every embedding, probe and adapter the notebook produces — and the default path **executes the checkpoint's own Python** (`trust_remote_code=True`) after digest-verifying it. Both facts are stated in the notebook before the relevant cell runs. Read them before you run anything.

Notebook specification: **DIMER Notebook Specification 2.0** — the notebook is **standalone** (§4) and declares its
profile and pedagogical mode (§3.4). It carries the repository package (`src/nucleotide_transformer_genomics_pipeline/`:
`pipeline.py`, `metrics.py`, `samples.py`) verbatim in tagged cells, the inline snapshot manifest and the exact runtime
pins, so it keeps working after export even if the repository changes or disappears. It is emitted by the fleet
generator `tools/build_notebook.py` from `tools/notebook_template.py`; `tests/test_notebook_parity.py` and
`tools/validate_release_assets.py` fail whenever the notebook, the package, the manifest or the pins diverge. Do not
edit the notebook by hand; edit the template or the package and regenerate (`--check` is enforced by CI).

| Notebook | Profile | Mode | Carrier | Capability | Default runtime | Sample | BYOD | Run-all | Release status |
|---|---|---|---|---|---|---|---|---|---|
| `nucleotide_transformer_colab.ipynb` | `E2E` | `GUIDED` | standalone (3 package modules carried verbatim) | 512-dimensional mean-pooled sequence embeddings and one masked-token prediction through the inference contract; majority and GC-threshold baselines and a logistic probe on the frozen embeddings; bounded fine-tuning of a mean-pooled head plus the last two encoder blocks for promoter classification with validation-based epoch selection; held-out accuracy and MCC; a safetensors adapter exported with a manifest and reloaded with verified parity | CPU (float32); CUDA used automatically when present — build record 34 s fine-tuning on an RTX 5070 Ti, 213 s on the workstation CPU | 2,200 human non-TATA promoter windows (251 bases; Genomic Benchmarks interval lists at a pinned Apache-2.0 commit, GRCh38 bases) carried inline with their digest; 1,600 / 200 / 400 | one CSV (`id`, `sequence`, `label`) through the same cells; off by default | yes — no clone, no credential, no upload, no configuration edit | **Candidate** — see `../docs/release-verification.md` |

## Conformance notes

- **Standalone (ST1–ST8):** the default path performs no clone, no repository install and no repository import; the three package modules are carried verbatim in dependency order with their SHA-256 tags, the identity constants live in the carried module only, and the inline manifest and pins are asserted against the repository by the parity tests.
- **Remote code (MOD6), stated rather than hidden:** the model cell states that this checkpoint ships its own model code and that `verify_snapshot` re-hashes the two Python files before `from_pretrained` imports them; the flag appears exactly once, inside the carried `from_pretrained`, after the verification, and the release validator fails the build otherwise. The tokenizer loads natively.
- **Model acquisition (MOD1–MOD8):** `stage_missing_files` fetches only absent manifest entries from the Hub at the immutable revision `81b29e5786726d891dbf929404ef20adca5b36f1` (never `main`); `verify_snapshot` re-hashes all 8 files and raises on the first mismatch; there is no fallback to a different download and no manifest-less load path.
- **Licence (§10 DAT6 and beyond):** the non-commercial condition is stated in the header and the prerequisites, restated in the closing section, recorded in the exported provenance JSON, and written into the adapter manifest (`license: cc-by-nc-sa-4.0`), which `load_artifact` requires. The notebook does not attempt to enforce it and says so.
- **Sample data (DAT1–DAT9):** carried inline with its digest and refused if edited; provenance (origin commit, four interval-list digests, reference genome, Ensembl) is printed and exported; the draw keeps the origin's train/test separation and `check_split_disjoint` asserts no sequence is shared; four refusal probes are executed so the failure messages are visible.
- **Evaluation (EVAL2/EVAL6/EVAL9/EVAL10/EVAL21):** the headline metric is MCC, read beside accuracy and per-class precision/recall/F1; the majority and GC-threshold baselines and the frozen probe are fitted on the training split only and scored on the same 400 held-out windows as the adapted model; `evaluation_report` writes the structured verdict; the cells assert the ordering probe > GC > majority and adapted > probe.
- **Score semantics (UNC1/UNC2):** embeddings are labelled as representations, masked-token outputs as the pre-training objective, and adapter probabilities as uncalibrated; the one seeded draw is labelled as having no dispersion estimate.
- **Ceilings (VAL6):** `MIN_BASES` 12, `MAX_BASES` 6,000 (1,000 tokens), `MAX_SEQUENCES` 20,000, `MIN_RECORDS` 8, `MAX_RECORDS` 20,000 and the `ACGTN` alphabet are printed with the input manifest before the model runs, and a uracil probe is rejected on record.
- **Artifact (OUT8, VER2, VER4):** `save_artifact` writes safetensors plus a manifest (format, base identity and digest, remote-code files, licence, tensor set, file digest, training history); `from_artifact` verifies before deserialising and reloads into a fresh pipeline; the cell asserts identical labels on 64 held-out windows.
- **BYOD (DAT10–DAT19):** `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog; when enabled, one CSV flows through the same validation, sequence-disjoint split, baselines, probe, fine-tuning, evaluation and export cells.
- `tools/validate_release_assets.py` performs source checks only. It does not satisfy the clean-runtime execution requirement — see `../docs/release-verification.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
