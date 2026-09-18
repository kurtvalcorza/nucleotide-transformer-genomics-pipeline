# Release verification

`tutorials/nucleotide_transformer_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) has one recorded
local CPU execution and no hosted clean-runtime run. More importantly, this row is **not on a release path at
all**: it is on HOLD behind two decisions — the upstream non-commercial licence and the remote-code
requirement — recorded in `MODEL_CARD.md` → *DIMER deployment notes*. Execution evidence is necessary for a
release but is nowhere near sufficient here, and this file does not pretend otherwise.

Static checks, JSON validation and code-cell compilation are **not** runtime evidence under DIMER Notebook
Specification 2.0 (REL8).

## Automatic coverage (static, every pull request)

CI runs `python tools/make_notebook.py --check` and `python tools/check_assets.py`, which check:

- the notebook is byte-identical to what `tools/make_notebook.py` emits, so the inline manifest and the
  dependency pins cannot drift from `weights/.../dimer-base-manifest.json` and `requirements.txt`;
- the notebook JSON parses, every code cell compiles as plain Python, no cell carries stored outputs or an
  execution count, and every code cell is preceded by an explanatory markdown cell;
- `metadata.dimer` declares the `TASK-INFERENCE` profile, spec `2.0`, `standalone: true`, the upstream
  licence, and `remote_code_executed: true` — the last of which is a fact about this notebook that a reader
  must not have to discover by running it;
- the snapshot manifest names the pinned identity, lists `model.safetensors`, and **lists the two Python
  files the loader executes**, so their digests are inside the verified perimeter;
- the notebook pins `MODEL_REVISION`, loads the tokenizer with `trust_remote_code=False`, keeps `USE_BYOD`
  defaulted to `False`, and calls `verify_snapshot(WEIGHTS_DIR)` **before** its `trust_remote_code=True`
  load — the ordering check is the one that matters most in this repository;
- no `git clone`, mutable `revision='main'`, `pickle.load`, unguarded `torch.load`, `extractall`, notebook
  magic or credential-bearing URL appears in any cell;
- the model card has its MODEL_CARD_SPEC 1.1 front matter, exactly one H1, all 19 required section headings
  in order, an immutable-provenance section, and the licence statements the other documents promise are
  there (`CC BY-NC-SA 4.0`, "Commercial use is not permitted under this licence", `ShareAlike`, the DIMER
  status, and the `trust_remote_code` disclosure);
- `README.md`, `MODEL_CARD.md`, `docs/WEIGHTS.md` and `tutorials/README.md` all name the same model id and
  the same 40-hex revision with no stray revisions, carry no placeholder text, and make no unsupported
  release, licence or "live in DIMER" claim; and
- `STATUS.md` declares `Hold` and the other documents agree.

CI also runs `ruff check tools`. These are source and provenance checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here would be promotion evidence — if the row were on a release path |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU container, Python 3.12 image; the committed notebook executed verbatim with a `google.colab` shim and **no repository checkout** | Reproducible clean-room executor of the same class |
| Local harness (pre-flight only) | Workstation, sequential cell executor, pins pre-installed | Builder pre-flight to catch defects; **not** a supported runtime and **not** promotion evidence |

## What a hosted run would have to show

Kept here so that whoever resolves the gates does not have to reconstruct it:

1. the exact commit under review, with static CI green;
2. a fresh runtime with **no repository checkout**, an empty Hugging Face cache and no pre-staged files under
   `weights/nt-v2-50m-multi-species/`;
3. `Run all` at the defaults (`USE_BYOD = False`, `SEED = 42`, `BATCH_SIZE = 8`) with no edits;
4. Section 1 reporting the installed versions equal to the inline `PINS` (`torch==2.14.0`,
   `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`);
5. Section 3 fetching the 8 entries at the immutable revision and verifying every digest, naming the two
   `.py` files it verified;
6. Section 4 printing the native loader's refusal, then loading tokenizer (no remote code) and model (remote
   code) and reporting 55,904,972 parameters;
7. Section 5 showing the 6-mer tokens and the single-base fallback after `N`;
8. Section 6 asserting the two sample classes share their base multiset;
9. Sections 7–9 producing 512-dimensional embeddings, a probe report with both baselines, and a masked-token
   top-5; and
10. Section 10 writing `outputs/nucleotide_transformer_{embeddings.csv,probe_report.json,result.json}`.

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `nucleotide_transformer_colab.ipynb` | `__LOCAL_ROW__` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12, CPU, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/nucleotide_transformer_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/nucleotide_transformer_colab.ipynb`).

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | `__LOCAL_ROW__` | Local pre-flight harness (Windows, CPython 3.12, CPU float32) | Default path (verify → load → tokenize → embed → probe → masked-LM → export) | __LOCAL_WALL__ | **PASSED** — pre-flight; see the gates above before reading this as readiness |

## Current status

The notebook source is complete, the static checks pass, and one local pre-flight execution of the committed
blob completed the whole default path. The repository stays at **Hold**: the blocking questions are the
CC BY-NC-SA 4.0 licence and the remote-code requirement, not engineering, and neither is resolved by running
the notebook again.
