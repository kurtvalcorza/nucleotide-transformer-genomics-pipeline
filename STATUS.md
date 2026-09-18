# Release status

Current status: **Hold** — blocked on two decisions, not on engineering.

1. **Licence.** The upstream weights are CC BY-NC-SA 4.0 (non-commercial). Whether DIMER may host them, and how the attribution and ShareAlike conditions are discharged for anything derived from them, has not been decided. The MODEL_MATRIX row stays HOLD.
2. **Remote code.** The checkpoint cannot be loaded by the native Transformers ESM classes and requires `trust_remote_code=True` (a bias-free SwiGLU feed-forward the native implementation does not have). Under the fleet's asset rules a model requiring unresolved remote code is not ordinarily qualified.

Because both gates are open, this repository ships the model card, the standalone tutorial notebook, the pinned snapshot manifest and the supporting documentation — and deliberately **no** DIMER pipeline module, adapter or release-asset validator. The notebook has one recorded local CPU execution of its committed blob (`docs/release-verification.md`) and the static checks (`tools/make_notebook.py --check`, `tools/check_assets.py`) pass. Neither fact is a readiness claim.
