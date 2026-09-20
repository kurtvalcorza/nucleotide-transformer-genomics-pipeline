# Release status

Current status: **Candidate** — the `E2E` tutorial notebook `tutorials/nucleotide_transformer_colab.ipynb` carries the pipeline package with its adaptation contract, the digest-pinned inline promoter sample, the pinned snapshot manifest and the exact runtime pins, and its default path has a recorded package-API build record (RTX 5070 Ti and CPU, 2026-09-20, `docs/release-verification.md`). It becomes **Release-grade** when the exact committed notebook blob executes top-to-bottom in a clean supported runtime (Kaggle Tesla T4 or Colab, no repository checkout, empty Hugging Face cache) and the row is recorded in `docs/release-verification.md`.

Both gates that previously held this row are closed:

1. **Licence — accepted 2026-09-19.** The upstream weights are CC BY-NC-SA 4.0 (non-commercial, attribution, ShareAlike). The row proceeds under those obligations; every embedding, probe and adapter inherits them, and the adapter manifest records the licence.
2. **Remote code — accepted 2026-09-20.** The checkpoint requires `trust_remote_code=True` (a bias-free SwiGLU feed-forward the native ESM class does not have). Accepted inside the perimeter `docs/WEIGHTS.md` describes: the two Python files are manifest entries, digest-verified before import, with no manifest-less load path, a native tokenizer, and a release validator that fails the build if the flag leaves that perimeter.

What the repository now ships: the package `nucleotide_transformer_genomics_pipeline` (representation, frozen probe, bounded fine-tuning of a mean-pooled head plus the last two encoder blocks, safetensors adapter round trip), 51 tests, the fleet notebook generator and its template, `tools/validate_release_assets.py`, `tools/pin_sample.py`, and the documentation. The static checks (`ruff`, `pytest`, `python tools/validate_release_assets.py`, `python tools/build_notebook.py --check`) establish source conformance, not DIMER readiness.
