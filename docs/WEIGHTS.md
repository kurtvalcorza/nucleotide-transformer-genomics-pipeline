# Weight provenance, the remote-code boundary, and DIMER notes

This repository pins **one** snapshot with its own `dimer-base-manifest.json` and redistributes none of it.

## Pinned snapshot

- Upstream: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
- Immutable revision: `81b29e5786726d891dbf929404ef20adca5b36f1`
- Weight format: SafeTensors (`model.safetensors`, 223,642,688 bytes)
- Upstream weight licence: **CC BY-NC-SA 4.0** — attribution, NonCommercial, ShareAlike. **Commercial use is not permitted under this licence.**
- Local layout: `weights/nt-v2-50m-multi-species/` holds the 8 manifest entries (`config.json`, `model.safetensors`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`, upstream `README.md`, **`modeling_esm.py`** and **`esm_config.py`**; 223,752,120 bytes total) with byte size and SHA-256 for each.
- Cross-check: the manifest digest for `model.safetensors` equals the `oid sha256` of the Hub LFS pointer at the pinned revision.
- **Nothing executable is vendored.** `weights/**/*.safetensors` and `weights/**/*.py` are git-ignored: a clone carries the manifest and the small text files, and the notebook stages the rest from the pinned revision and verifies every digest before use. This also keeps the upstream licence question attached to the upstream files rather than to a copy in this repository.

## Remote-code trust boundary

This checkpoint requires `trust_remote_code=True`, and the reason is architectural rather than incidental:

- its `config.json` carries an `auto_map` (`esm_config.EsmConfig`, `modeling_esm.EsmForMaskedLM`) and **no `model_type`**, so `AutoConfig` cannot resolve a native class; and
- more importantly, `modeling_esm.py` implements a **bias-free SwiGLU** feed-forward block — it projects to twice the intermediate width, splits the result and gates one half with SiLU, with `bias=config.add_bias_fnn` (false in this config) — whereas the native Transformers ESM implementation uses a plain GELU feed-forward **with** bias. Loading these weights into the native class would not reproduce this model.

Verified, not assumed: `AutoModel.from_pretrained(..., trust_remote_code=False)` refuses the checkpoint with a `ValueError` naming the custom code (recorded in `MODEL_CARD.md` → Runtime and in the tutorial's own output).

What this repository does about it:

1. both Python files are **manifest entries**, so their bytes are pinned to one immutable revision and their SHA-256 is verified **before** the loader imports them; an upstream change fails loudly instead of executing;
2. `tools/check_assets.py` fails if either file leaves the manifest, or if the notebook's `verify_snapshot(WEIGHTS_DIR)` call stops preceding its `trust_remote_code=True` load; and
3. the notebook explains the boundary in its own section before the load, rather than flipping the flag quietly.

What it does **not** do: claim that pinning and hashing make third-party code safe. No security review of `modeling_esm.py` has been performed beyond reading what it does architecturally. Executing it is a trust decision about InstaDeep's published code, and under the fleet's asset rules a model that requires unresolved remote code is not ordinarily qualified — which is why this row ships no DIMER pipeline module.

## DIMER notes

- **Status: Planned / conditional — not approved, not deployed, not live.** Two gates are open: the non-commercial licence, and the remote-code requirement above.
- If it is ever approved, the upload is `model.safetensors` plus the tokenizer files **and** the two Python files, because the checkpoint cannot be loaded without them. An upload-format review should note that two of the eight files are executable Python.
- Redistribution of the weights is permitted subject to CC BY-NC-SA 4.0's attribution, NonCommercial and ShareAlike conditions; DIMER would grant no rights beyond the upstream licensor's, and hosting would imply no affiliation with or endorsement by InstaDeep, NVIDIA, TUM or Hugging Face.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
