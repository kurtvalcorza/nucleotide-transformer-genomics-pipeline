# Nucleotide Transformer v2 50M genomics pipeline

DIMER pipeline for **`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`**, pinned to an immutable Hugging Face revision: DNA sequence representations (one mean-pooled 512-d vector per sequence), masked-token prediction, a logistic probe on the frozen embeddings, and bounded supervised fine-tuning of a promoter classifier — a mean-pooled head plus the last two encoder blocks — exported as a safetensors adapter that reloads against the pinned base.

> [!IMPORTANT]
> **Non-commercial model licence.** The weights are distributed under **CC BY-NC-SA 4.0** — attribution, NonCommercial, ShareAlike. Copying, redistribution and adaptation are permitted for non-commercial purposes; **commercial use is not permitted under this licence**. The test is whether your use is primarily intended for commercial advantage or monetary compensation — a question about your use, not about your organisation's type. Every embedding, probe and adapter this pipeline produces is derived from the weights and carries the same conditions (the adapter manifest records them). This repository grants no rights beyond those the upstream licensor grants.

> [!WARNING]
> **This checkpoint executes its own model code.** It cannot be loaded by the native Transformers ESM classes (its `config.json` carries an `auto_map` and its feed-forward block is a bias-free SwiGLU the native implementation lacks), so `from_pretrained` imports the checkpoint's `modeling_esm.py` and `esm_config.py` with `trust_remote_code=True`. Both files are manifest entries and are digest-verified **before** they are imported; there is no manifest-less load path; the tokenizer loads natively. Digest verification proves the executed code is the pinned upstream code byte for byte — it is not a safety claim about that code. See `docs/WEIGHTS.md`.

## Upstream alignment

- Model: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
- Revision: `81b29e5786726d891dbf929404ef20adca5b36f1`
- Upstream weight licence: **CC BY-NC-SA 4.0** (non-commercial)
- Developed by: InstaDeep, NVIDIA and TUM (Dalla-Torre et al., Nature Methods 2024)
- Architecture: 12-block transformer encoder, hidden size 512, 16 heads, rotary positions, bias-free SwiGLU feed-forward; 53,534,401 encoder parameters, 55,904,972 with the masked-LM head as loaded
- Tokenization: 6-mers over a 4,107-token vocabulary, falling back to single bases around `N` and at a trailing remainder; a 251-base window is 47 tokens
- Trained sequence length: 1,000 tokens ≈ 6,000 bases (the pipeline's ceiling)

## Quick start

```python
from nucleotide_transformer_genomics_pipeline import NucleotideTransformerPipeline

pipe = NucleotideTransformerPipeline.from_pretrained(allow_download=True)   # stage -> verify every file (incl. the model code) -> load
vectors = pipe.embed(["ATTCCGATTCCGATTCCGACGTACGTACGT"])                    # one 512-d mean-pooled vector per sequence
masked = pipe.predict_masked("ATTCCG" * 3 + "<mask>" + "ATTCCG" * 3)        # the pre-training objective at one masked position
```

`pip install -e .` inside the pinned runtime (`pyproject.toml`: `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `huggingface-hub==0.36.2`, `numpy==2.5.3`; Python 3.12). `predict` is refused until an adapter exists — the checkpoint is a representation model, not a classifier.

## Weights layout

`weights/nt-v2-50m-multi-species/dimer-base-manifest.json` pins all 8 snapshot files (paths, byte sizes, SHA-256; 223,752,120 bytes): `model.safetensors`, `config.json`, the tokenizer files, the upstream `README.md`, and the two Python files the loader executes. The small text files are committed; `weights/**/*.safetensors` and `weights/**/*.py` are git-ignored and staged from the pinned revision by `stage_missing_files(..., allow_download=True)`, then re-hashed by `verify_snapshot`. The upstream `pytorch_model.bin` and JAX checkpoint are not in the manifest and are never staged. `.gitattributes` carries `weights/** -text` so a Windows checkout cannot break a digest.

## Adaptation contract

```python
from nucleotide_transformer_genomics_pipeline import (
    NucleotideTransformerPipeline, gc_threshold_baseline, majority_baseline, sample_dataset,
    load_byod_dataset, split_dataset,
)

splits = sample_dataset()                                    # pinned inline draw: train 1,600 / validation 200 / test 400
# or: splits = split_dataset(load_byod_dataset("my_sequences.csv"), seed=42)   # columns id,sequence,label
pipe = NucleotideTransformerPipeline.from_pretrained(allow_download=True)

majority = majority_baseline(splits["train"], splits["test"])            # 0.5 / MCC 0 on the balanced split
gc = gc_threshold_baseline(splits["train"], splits["test"])              # composition-only floor, fitted on train
frozen = pipe.linear_probe(splits["train"], splits["test"])              # logistic probe on the frozen embeddings
result = pipe.adapt(splits["train"], splits["validation"])               # head + last 2 blocks; 6 epochs, lr 3e-5, batch 16
adapted = pipe.evaluate(splits["test"])                                  # accuracy, MCC, per-class P/R/F1, confusion

pipe.save_artifact("outputs/adapter")                                    # adapter.safetensors + manifest.json (licence inherited)
reloaded = NucleotideTransformerPipeline.from_artifact("outputs/adapter")  # fresh verified base + verified adapter
```

Records are `{id, sequence, label}` with `label` in `{0, 1}`; 8..20,000 records, unique ids, both labels in a training split; `split_dataset` de-duplicates by exact sequence and `check_split_disjoint` asserts no leakage. `adapt` trains a seeded mean-pooled head (263,682 parameters) plus the last `layers` encoder blocks (2 by default; 8,660,482 trainable in all) with cross-entropy, AdamW without weight decay, gradient clipping at 1.0 and epoch selection on validation MCC; on any exception the frozen weights are restored. `save_artifact` writes the trained tensors as safetensors (34,645,456 bytes for the default recipe) with a manifest naming the base identity and weight digest, the remote-code files, the inherited licence, the tensor set, the digest and the training history; `load_artifact` refuses anything that disagrees before deserialising.

**Build record (2026-09-20, RTX 5070 Ti, seed 0, 400 held-out windows):** majority 0.5 / MCC 0.0; GC ≥ 0.560 → 0.715 / 0.431; frozen probe 0.815 / 0.632; **adapted 0.8425 / 0.693** (epoch 2 of 6 kept; 33.9 s; 558 MiB peak); reload parity exact. The same run on CPU reproduced every metric to four decimals (adapt 212.9 s). Sample-sanity evidence on one seeded draw with no dispersion estimate — not a benchmark.

### The sample and its provenance

The default data is the human non-TATA promoter benchmark of Genomic Benchmarks (Grešová et al. 2023): 251-base windows of the GRCh38 reference that either centre on a promoter from the Eukaryotic Promoter Database or contain none. Interval lists are read from the Apache-2.0 repository `ML-Bioinfo-CEITEC/genomic_benchmarks` at commit `605d8539830e16c85abe7826990958303ffc5e1c` (four gzipped CSVs pinned by size and SHA-256); bases come from the public-domain reference through Ensembl REST. The seeded draw is carried **inline** in `samples.py` with its digest, so the notebook downloads no data; `tools/pin_sample.py` reproduces it from the origin and asserts the digest (every window was also cross-checked against the benchmark authors' Hub re-upload when the pin was made).

## Input ceilings

`MIN_BASES` 12, `MAX_BASES` 6,000 (= `MAX_TOKENS` 1,000 six-mers), `MAX_SEQUENCES` 20,000 per call, alphabet `A/C/G/T/N` (upper-cased on entry), `MIN_RECORDS` 8, `MAX_RECORDS` 20,000, ids `[A-Za-z0-9_.:-]{1,64}`. `validate_inputs` returns an input manifest before any model runs.

## Tutorials

`tutorials/nucleotide_transformer_colab.ipynb` — the standalone `E2E` notebook (DIMER Notebook Specification 2.0 §4): it carries the package verbatim, the inline manifest and the exact pins, stages and verifies the snapshot (including the model code) before loading, reads the inline sample, exercises the representation contract, scores the baselines and the frozen probe, fine-tunes, evaluates on the held-out windows, and exports and reloads the adapter. Generated by `tools/build_notebook.py` from `tools/notebook_template.py`; do not edit cells by hand. See `tutorials/README.md`.

## Release status

**Candidate** — the package, tests, generator parity and release-asset validation are in place and the default path has a package-API build record on GPU and CPU (`docs/release-verification.md`); promotion to Release-grade follows the exact committed notebook blob executing top-to-bottom in a clean supported runtime with no repository checkout. Static and unit checks are necessary but are never that evidence.

## Checks

```
pip install -e .
ruff check src tests tools
pytest
python tools/validate_release_assets.py
python tools/build_notebook.py --check
```

`pytest` runs 51 tests: the offline contract tests always, the model-backed tests only where the snapshot is staged under `weights/nt-v2-50m-multi-species/` (and the CUDA test only where a device is visible). `python tools/pin_sample.py` (network) re-derives the inline sample from the origin and asserts its digest.

## Layout

```
src/nucleotide_transformer_genomics_pipeline/
  pipeline.py                     identity constants, snapshot verification and staging, the remote-code perimeter, the pipeline class
  metrics.py                      accuracy / MCC / per-class measures, majority and GC-threshold baselines
  samples.py                      the pinned inline promoter sample, dataset validation, digests, split and BYOD readers
tests/                            offline contract tests, model-backed tests, notebook parity tests
tools/build_notebook.py           fleet notebook generator (v2; this copy adds the optional `model_cell_note` key)
tools/notebook_template.py        the E2E notebook's template
tools/validate_release_assets.py  static release-asset validation incl. the remote-code perimeter rule
tools/pin_sample.py               reproduces the inline sample from the origin (network) and asserts its digest
tutorials/                        the standalone notebook and its registry
weights/nt-v2-50m-multi-species/  the pinned snapshot manifest and the small text files
docs/WEIGHTS.md                   provenance, the remote-code trust boundary, the sample's provenance, hosting notes
docs/release-verification.md      what has and has not been executed, and the promotion procedure
MODEL_CARD.md                     MODEL_CARD_SPEC 1.1 card
```

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, adaptation contract and build record, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, the remote-code trust boundary, the sample's provenance and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code, notebook, tooling and interval-derived sample are Apache-2.0 (see `LICENSE`). The upstream weights — and everything derived from them, including adapters — are **CC BY-NC-SA 4.0** (non-commercial); the checkpoint's `modeling_esm.py` / `esm_config.py` carry Meta / Hugging Face copyright headers and derive from the Apache-2.0 Transformers ESM implementation. A permissive licence on the code does not remove the NonCommercial condition from the weights; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
