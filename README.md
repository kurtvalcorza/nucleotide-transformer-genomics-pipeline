# Nucleotide Transformer v2 50M Multi-Species — model card and tutorial

Documentation and a standalone tutorial for **`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`**, pinned to an immutable Hugging Face revision. This repository is **not** a DIMER pipeline package: it ships no `src/` module, no DIMER adapter and no release-asset validator, because the row is blocked on two open decisions (see below). What it does ship is an immutable pin with per-file digests, a MODEL_CARD_SPEC 1.1 model card, and a notebook that demonstrates the model honestly.

> [!IMPORTANT]
> **Non-commercial model licence.** The weights are distributed under **CC BY-NC-SA 4.0** — attribution, NonCommercial, ShareAlike. Copying, redistribution and adaptation are permitted for non-commercial purposes; **commercial use is not permitted under this licence**. The test is whether your use is primarily intended for commercial advantage or monetary compensation — a question about your use, not about your organisation's type. This repository grants no rights beyond those the upstream licensor gives you.

## Two open gates

1. **Licence.** CC BY-NC-SA 4.0 is a non-commercial licence. Whether DIMER may host the weights, and how attribution and ShareAlike are discharged for anything derived from them, has not been decided. The MODEL_MATRIX row stays **HOLD**.
2. **Remote code.** The checkpoint cannot be loaded by the native Transformers ESM classes: its `config.json` carries an `auto_map` instead of a `model_type`, and its feed-forward block is a **bias-free SwiGLU** that the native implementation does not have, so the native class would not reproduce this model. `trust_remote_code=True` is an architectural requirement here, not a workaround for a loading error — and a model requiring unresolved remote code is not ordinarily qualified for DIMER.

The repository narrows the second gate as far as it can: `modeling_esm.py` and `esm_config.py` are manifest entries, so the Python that gets executed is pinned and digest-verified **before** it is imported, and `tools/check_assets.py` fails the build if that ordering is ever broken. Verification is not a safety claim about third-party code.

## Upstream alignment

- Model: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
- Revision: `81b29e5786726d891dbf929404ef20adca5b36f1`
- Upstream weight licence: **CC BY-NC-SA 4.0** (non-commercial)
- Developed by: InstaDeep, NVIDIA and TUM
- Architecture: 12-layer transformer encoder, hidden size 512, 16 heads, rotary positions, bias-free SwiGLU feed-forward; 55,904,972 parameters
- Tokenization: 6-mers over a 4,107-token vocabulary, falling back to single bases around `N` and at a trailing remainder
- Trained sequence length: 1,000 tokens ≈ 6,000 bases

## What the tutorial does

`tutorials/nucleotide_transformer_colab.ipynb` is a standalone `TASK-INFERENCE` notebook (DIMER Notebook Specification 2.0 §4). Its default path states the licence before it downloads anything, stages and digest-verifies all 8 pinned files, shows the native loader refusing the checkpoint and then loads it with remote code, walks through 6-mer tokenization and its single-base fallback, generates a deterministic 48-sequence sample, extracts 512-dimensional mean-pooled embeddings, evaluates them with a nearest-centroid probe against majority-class and GC-content baselines on a held-out split, reads one masked-token prediction, and exports embeddings, metrics and provenance. BYOD (FASTA or CSV) is optional and gated off by default.

The sample is built so that the evaluation means something: the two classes contain **exactly the same multiset of bases** — identical GC content, identical length — and differ only in whether a fixed 6-mer motif is present or its letters are permuted. A composition baseline therefore cannot separate them, which is what makes the embedding probe's result informative.

Recorded CPU run (2026-09-18, weights already staged): 8 files verified in 0.1 s, model load 0.4 s, 48 sequences embedded in 0.7 s, probe accuracy **1.0** on 24 held-out sequences with a mean cosine margin of ±0.0036, against a majority baseline of 0.5 and a GC baseline of **0.4167**. Tutorial sample-sanity evidence on synthetic sequence — not a genomics benchmark.

## Layout

```
MODEL_CARD.md                     MODEL_CARD_SPEC 1.1 card, including the licence and DIMER gates
tutorials/                        the standalone notebook and its registry
tools/make_notebook.py            emits the notebook (not the fleet generator; this row has no package)
tools/check_assets.py             source hygiene: notebook parses/compiles, identity consistency, card sections
weights/nt-v2-50m-multi-species/  the pinned snapshot manifest and the small text files
docs/WEIGHTS.md                   provenance, the remote-code trust boundary, upload notes
docs/release-verification.md      what has and has not been executed
```

Nothing executable from upstream is vendored here: `weights/**/*.safetensors` and `weights/**/*.py` are git-ignored, so the weights and the model code are staged from the pinned revision and verified against the manifest.

## Checks

```
pip install -r requirements.txt
python tools/make_notebook.py --check
python tools/check_assets.py
```

These are source checks. They are not clean-runtime execution evidence; see `docs/release-verification.md`.

## Release status

**Hold.** The notebook has one recorded local CPU execution of its committed blob, and the static checks pass, but the row is blocked on the licence and remote-code decisions above rather than on engineering. No DIMER profile is planned until both are resolved.

## Licensing

- **Model weights:** CC BY-NC-SA 4.0 (non-commercial), redistributed by nobody here — staged from the pinned upstream revision.
- **The checkpoint's supporting Python:** Meta/Hugging Face copyright, derived from the Apache-2.0 Transformers ESM implementation. A permissive code licence does **not** remove the NonCommercial condition from the weights.
- **This repository's content:** Apache-2.0 (`LICENSE`).

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
