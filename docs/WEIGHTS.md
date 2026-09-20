# Weight provenance, the remote-code boundary, and DIMER notes

This repository pins **one** snapshot with its own `dimer-base-manifest.json` and redistributes none of it.

## Pinned snapshot

- Upstream: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
- Immutable revision: `81b29e5786726d891dbf929404ef20adca5b36f1`
- Weight format: SafeTensors (`model.safetensors`, 223,642,688 bytes)
- Upstream weight licence: **CC BY-NC-SA 4.0** — attribution, NonCommercial, ShareAlike. **Commercial use is not permitted under this licence.** Adapters written by `save_artifact` are derivatives and inherit it; the artifact manifest records `license: cc-by-nc-sa-4.0` and `load_artifact` refuses a manifest that says otherwise.
- Local layout: `weights/nt-v2-50m-multi-species/` holds the 8 manifest entries (`config.json`, `model.safetensors`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`, upstream `README.md`, **`modeling_esm.py`** and **`esm_config.py`**; 223,752,120 bytes total) with byte size and SHA-256 for each.
- Cross-check: the manifest digest for `model.safetensors` equals the `oid sha256` of the Hub LFS pointer at the pinned revision. The upstream repository also hosts a `pytorch_model.bin` (223,682,957 bytes) and a `jax_model/` directory; neither is in the manifest, and `stage_missing_files` fetches only manifest entries, so no pickle is ever staged or opened.
- **Nothing executable is vendored.** `weights/**/*.safetensors` and `weights/**/*.py` are git-ignored: a clone carries the manifest and the small text files, and `stage_missing_files(..., allow_download=True)` stages the rest from the pinned revision; `verify_snapshot` re-hashes every file before use. This also keeps the upstream licence question attached to the upstream files rather than to a copy in this repository.

## Remote-code trust boundary

This checkpoint requires `trust_remote_code=True`, and the reason is architectural rather than incidental:

- its `config.json` carries an `auto_map` (`esm_config.EsmConfig`, `modeling_esm.EsmForMaskedLM`, `...ForTokenClassification`, `...ForSequenceClassification`) and **no `model_type`**, so `AutoConfig` cannot resolve a native class; and
- more importantly, `modeling_esm.py` implements a **bias-free SwiGLU** feed-forward block — it projects to twice the intermediate width, splits the result and gates one half with SiLU, with `bias=config.add_bias_fnn` (false in this config) — whereas the native Transformers ESM implementation uses a plain GELU feed-forward **with** bias. Loading these weights into the native class would not reproduce this model.

Verified, not assumed: `AutoModel.from_pretrained(..., trust_remote_code=False)` refuses the checkpoint with a `ValueError` naming the custom code. A second trap, also verified: `AutoModel.from_pretrained(..., trust_remote_code=True)` **silently falls through to the native `EsmModel`** because the `auto_map` has no `AutoModel` entry, and then fails on the SwiGLU shape mismatch (4096 vs 2048). The pipeline loads through `AutoModelForMaskedLM`, which the `auto_map` does map, and asserts that the loaded class comes from the snapshot's `modeling_esm` module.

What this repository does about it:

1. both Python files are **manifest entries**, so their bytes are pinned to one immutable revision and their SHA-256 is verified **before** the loader imports them; an upstream change fails loudly instead of executing;
2. `verify_snapshot` and `stage_missing_files` refuse a manifest that does not list `model.safetensors`, `modeling_esm.py` and `esm_config.py`, and `from_pretrained` has **no manifest-less path** — without a digest check nothing is imported;
3. the tokenizer is the native `EsmTokenizer`, loaded with `trust_remote_code=False`;
4. `tools/validate_release_assets.py` fails the build if `trust_remote_code=True` appears anywhere in the notebook outside the carried module cells, more than once inside them, or before `verify_snapshot(root)` inside `from_pretrained`; and
5. the notebook's model cell states that remote code runs, and why, before the load rather than flipping the flag quietly.

One consequence for callers: the remote module caches its rotary-embedding `cos`/`sin` tables on first forward, so the pipeline runs every inference path under `torch.no_grad()` — a first call under `torch.inference_mode()` would make every later backward pass fail.

What it does **not** do: claim that pinning and hashing make third-party code safe. No security review of `modeling_esm.py` has been performed beyond reading what it does architecturally. Executing it is a trust decision about InstaDeep's published code; the fleet made that decision for this row on 2026-09-20 on the perimeter above, and this remains the only row of the fleet where remote code runs.

## DIMER notes

- **Status: Candidate.** The licence gate closed on 2026-09-19 (CC BY-NC-SA 4.0 accepted with its obligations) and the remote-code gate on 2026-09-20 (accepted inside the verified perimeter). Promotion to Release-grade follows the clean-runtime execution evidence in `release-verification.md`.
- The upload is `model.safetensors` plus the tokenizer files **and** the two Python files, because the checkpoint cannot be loaded without them. An upload-format review should note that two of the eight files are executable Python.
- Redistribution of the weights is permitted subject to CC BY-NC-SA 4.0's attribution, NonCommercial and ShareAlike conditions; DIMER grants no rights beyond the upstream licensor's, and hosting implies no affiliation with or endorsement by InstaDeep, NVIDIA, TUM or Hugging Face.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.

## The fine-tuning sample

The data the E2E tutorial fine-tunes on is not a weight, but it is pinned the same way and documented here so the provenance chain is in one place:

- Interval lists: Genomic Benchmarks `human_nontata_promoters` from `ML-Bioinfo-CEITEC/genomic_benchmarks` (Apache-2.0) at commit `605d8539830e16c85abe7826990958303ffc5e1c` — four gzipped CSVs (`train/positive.csv.gz` 188,240 B, `train/negative.csv.gz` 137,784 B, `test/positive.csv.gz` 63,260 B, `test/negative.csv.gz` 46,480 B) pinned by size and SHA-256 in `samples.py`; 27,097 train and 9,034 test rows, all 251-base windows over 24 chromosomes.
- Bases: the GRCh38 human reference genome (Genome Reference Consortium; public domain), read interval by interval through the Ensembl REST batch endpoint with the origin's convention (0-based half-open intervals, `-` strand reverse-complemented).
- The seeded draw (seed 42; 1,600 / 200 / 400, balanced; train and validation from the origin train lists, test from the origin test lists) is carried **inline** in `samples.py` with its digest `d913c4bd4cf74199456d24ced1f3ddc8f39e9241ab2eca132ffb9885efc62cfb`, so the notebook downloads no data. Every one of the 2,200 windows was cross-checked, sequence and label, against the benchmark authors' own Hub re-upload when the pin was made (an oracle, not a source: that re-upload declares no licence, which is why the origin repository is the pinned source).
- `tools/pin_sample.py` reproduces the draw from those sources and asserts the digest (`--write` regenerates the inline block); `sample_dataset()` refuses to return a block whose digest has drifted.
