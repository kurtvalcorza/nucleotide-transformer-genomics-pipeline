# Release verification

`tutorials/nucleotide_transformer_colab.ipynb` (`E2E`, **standalone** carrier) is promoted from `Candidate` to
`Release-grade` only by the exact committed notebook blob executing top-to-bottom in a clean supported runtime with no
repository checkout, recorded in the tables below. Static checks, JSON validation, unit tests and code-cell
compilation are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8), and the package-API build
records are a builder's pre-flight, not promotion evidence.

The two decisions that previously held this row are closed and recorded in `MODEL_CARD.md` → *DIMER deployment
notes*: the CC BY-NC-SA 4.0 licence was accepted with its obligations on 2026-09-19 and the remote-code requirement
was accepted inside the digest-verified perimeter on 2026-09-20. This file therefore gates on execution evidence only.

## Automatic coverage (static, every pull request)

CI installs the pinned runtime, runs `ruff check src tests tools`, `pytest`, `python tools/validate_release_assets.py`
and `python tools/build_notebook.py --check`, which together check:

- the notebook is byte-identical to what `tools/build_notebook.py` renders from `tools/notebook_template.py` and the
  package at HEAD (PAR3), so the carried modules, the inline manifest and the dependency pins cannot drift
  (`tests/test_notebook_parity.py` PAR1–PAR3, ST1);
- the notebook JSON parses, every code cell compiles as plain Python, no cell carries stored outputs or an execution
  count, every code cell is preceded by an explanatory markdown cell, and `metadata.dimer` declares the `E2E`
  profile, spec `2.0`, `standalone: true`, the pedagogical mode and the generating revision with the package digest;
- the snapshot manifest names the pinned identity, lists `model.safetensors`, and **lists the two Python files the
  loader executes**, so their digests are inside the verified perimeter (`tests/test_pipeline.py`);
- **the remote-code perimeter:** `trust_remote_code=True` appears nowhere in the notebook outside the carried module
  cells, exactly once inside them, inside `from_pretrained` and after `verify_snapshot(root)`; the tokenizer load is
  native; the carried module names the remote-code files;
- `USE_BYOD` is a Colab form parameter assigned exactly once to `False`; `google.colab` is imported only inside the
  BYOD gate; the default path performs no clone, no repository install, no repository import, no mutable
  `revision='main'`, no `pickle.load`, no unguarded `torch.load`, no `extractall`, no notebook magic and no
  credential-bearing URL;
- the notebook calls the carried package's public API for every stage (data, inference contract, baselines, probe,
  adaptation, evaluation, artifact export and reload) and does not reimplement it with direct library calls; the
  required outputs are written; the learner-facing statements the template promises are present;
- the model card has its MODEL_CARD_SPEC 1.1 front matter, exactly one H1, all 19 required section headings in order,
  an immutable-provenance section and no unsupported release or benchmark claim; `README.md`, `MODEL_CARD.md` and
  `docs/WEIGHTS.md` name the same model id and the same 40-hex revision (plus the pinned `genomic_benchmarks` commit)
  with no stray revisions and no placeholder text; `STATUS.md`, `README.md` and `tutorials/README.md` agree on one
  status token;
- the package contract offline: snapshot verification refuses missing, resized, altered or unlisted files and a
  manifest without the remote code; sequence and dataset validation refuse each malformed input before any model
  import; the inline sample hashes to `SAMPLE_DIGEST` and its splits are balanced, disjoint and of the pinned sizes;
  the metrics and baselines behave; the artifact manifest checker refuses each deviation (format, base, digest, size,
  licence, tensor set); the trainable-tensor selection picks the last blocks only.

Model-backed tests (`tests/test_model_backed.py`) run where the snapshot is staged: the loaded class comes from the
snapshot's `modeling_esm` module with 55,904,972 parameters; embeddings, token counts and masked prediction;
the frozen probe leaves the pipeline's state untouched; a one-epoch adaptation of the head plus one block on a slice
of the sample, the artifact round trip with identical predictions from a fresh pipeline, refusal of a tampered tensor
set, the transactional restore after a failure inside training, the argument guards, and — where CUDA is visible —
the same path on the accelerator. They are pre-flight, not clean-runtime evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness / package-API build record (pre-flight only) | Workstation, pins pre-installed, snapshot pre-staged | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/nt-v2-50m-multi-species/` (the standalone path writes the manifest itself and stages the
   missing files — including the two Python files — from the Hub; the directory may not be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 6`, `LEARNING_RATE = 3e-5`, `TRAINED_LAYERS = 2`, `SEED = 0`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `huggingface-hub==0.36.2`,
   `numpy==2.5.3` (an interpreter restart after the install is expected where the runtime's preinstalled torch or
   numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `NucleotideTransformerPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_sequences`, `validate_inputs`, `evaluation_report`, `classification_metrics`,
     `gc_content`, `majority_baseline`, `gc_threshold_baseline`, `sample_dataset`, `validate_dataset`,
     `dataset_digest`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv` and the
     ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from
     `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` at the immutable revision on a clean runtime,
     `verify_snapshot` returning its dict (8 files, `remote_code_files` naming `modeling_esm.py` and
     `esm_config.py`, the 224 MB `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)`
     loading from the verified directory with `remote_code_executed: True` (the upstream `pytorch_model.bin` and JAX
     checkpoint are not in the manifest and must not be staged);
   - Section 4: `sample_dataset` returning the pinned draw (digest `d913c4bd…`; 1,600 / 200 / 400, each half
     promoter) with `check_split_disjoint` reporting no shared sequence and the three dataset digests printed; mean
     GC about 0.62 for promoters and 0.48 for negatives; `outputs/…_train.csv` written; the four dataset refusal
     probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_BASES` 12, `MAX_BASES` 6000, `MAX_TOKENS` 1000, `MAX_SEQUENCES` 20000,
     `MIN_RECORDS` 8, `MAX_RECORDS` 20000) and `remote_code_executed` surfaced; `validate_inputs` writing
     `outputs/…_input_manifest.json` with one recorded rejection finding from the uracil probe; `embed` returning
     six 512-d vectors with every sanity check `True`, 47 tokens per 251-base window and 51 per 300-base synthetic
     sequence, the motif-versus-permuted cosine printed as an observation, `outputs/…_embeddings.csv` written;
     `predict_masked` returning a top-5 distribution at the masked position; `predict` refused before adaptation;
   - Section 6: the majority baseline (0.5 / MCC 0), the GC-threshold baseline (≈ 0.715 / 0.431 with the rule
     `GC >= 0.560 -> positive` in the build record) and the frozen probe (≈ 0.815 / 0.632) with per-class rows, and
     the cell's assertion that probe MCC > GC MCC > majority MCC;
   - Section 7: `pipe.adapt` printing epoch 0 as the untrained head on the frozen encoder, 8,660,482 trainable
     (263,682 in the head) of 53,798,083 parameters, and a six-epoch history with validation accuracy and MCC (build
     record: 0.505 / 0.071 → peak 0.805 / 0.621 at epoch 2, kept; training loss 0.51 → 0.14);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on both measures, the
     per-class rows, the `evaluation_report` verdict (`adapted_beats_frozen` and `adapted_beats_baselines` both
     true) and `outputs/…_evaluation_report.json` written (the cell asserts adapted MCC > frozen-probe MCC — 0.693
     versus 0.632 in the build record — and the baselines verdict);
   - Section 9: eight held-out windows printed with GC, the GC rule's answer, the adapted label, the promoter
     probability and the truth; the synthetic pair classified; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors,manifest.json}` (32 tensors, 34,645,456 bytes, `license`
     `cc-by-nc-sa-4.0`, `base.remote_code_files` naming both Python files) and
     `NucleotideTransformerPipeline.from_artifact` reloading it with 64/64 identical labels and a maximum probability
     difference below 1e-4 (the cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model
     identity and licence, the `remote_code` block, the snapshot block (`weight_file`, `weight_format`,
     `weight_sha256`), the `corpus` block with the interval-list pins and the sample digests, the inference-contract
     block, the comparison, the verdict, the artifact digest and licence, the reload parity, the runtime versions and
     device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache and the weights directory were clean, outcome, produced
   outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD` deviation
   in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `nucleotide_transformer_colab.ipynb` (`E2E`) | `3e37b6c` / `c1aa7361` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-nucleotide-transformer` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` / `numpy 2.0.2` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` / `numpy 2.5.3` after, Python 3.12.13, `cuda:0`, driver 580.159.04) | **PASSED** — 11/11 code cells ok (1 restart after the install cell, as the stale-import guard is designed to force); 18 files / 224 MB staged from the Hub into a clean cache, all 8 manifest entries fetched and digest-verified, `remote_code.verified_before_import` true; comparison accuracy {majority 0.5, gc_threshold 0.715, frozen_probe 0.815, adapted **0.8425**} / MCC {0.0, 0.4308, 0.6315, **0.6934**}; validation curve 0.505/0.071 → 0.795/0.609, 0.805/0.621 (epoch 2 kept), 0.795/0.606, 0.78/0.577, 0.765/0.533, 0.80/0.612; adapt 32.1 s; confusion tp 153 / tn 184 / fp 16 / fn 47; adapter 34,645,456 B / 32 tensors, licence `cc-by-nc-sa-4.0`; reload parity 64/64 labels, max probability difference 0.0 — every figure identical to the RTX 5070 Ti and CPU build records to four decimals |
| `nucleotide_transformer_colab.ipynb` (`TASK-INFERENCE`, superseded) | `f9dc86f` / `9f67399f7bcf` | 2026-09-18 | Kaggle Tesla T4 (clean runtime, empty Hugging Face cache, no repository checkout) | PASS — 9/9 code cells, 218.6 s; all 8 files downloaded and digest-verified including the two Python files before the remote-code import; nearest-centroid probe 0.9583 (n = 24) against majority 0.5 and GC 0.375. Evidence for the earlier representation-only notebook, not for the `E2E` blob. |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/nucleotide_transformer_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/nucleotide_transformer_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | `3e37b6c` / `c1aa7361` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-nucleotide-transformer` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` / `numpy 2.0.2` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` / `numpy 2.5.3` after, Python 3.12.13, `cuda:0`, driver 580.159.04) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution; executor `.agent/scripts/kaggle-serial-test-suite.py`, fresh-subprocess `nbclient`) | 248.3 s (163.2 s install + restart, 85.0 s for the whole default path) | **PASSED** — 11/11 code cells ok (1 restart after the install cell, as the stale-import guard is designed to force); 18 files / 224 MB staged from the Hub into a clean cache, all 8 manifest entries fetched and digest-verified, `remote_code.verified_before_import` true; comparison accuracy {majority 0.5, gc_threshold 0.715, frozen_probe 0.815, adapted **0.8425**} / MCC {0.0, 0.4308, 0.6315, **0.6934**}; validation curve 0.505/0.071 → 0.795/0.609, 0.805/0.621 (epoch 2 kept), 0.795/0.606, 0.78/0.577, 0.765/0.533, 0.80/0.612; adapt 32.1 s; confusion tp 153 / tn 184 / fp 16 / fn 47; adapter 34,645,456 B / 32 tensors, licence `cc-by-nc-sa-4.0`; reload parity 64/64 labels, max probability difference 0.0 — every figure identical to the RTX 5070 Ti and CPU build records to four decimals |
| 2026-09-20 | package API at the candidate revision (not the notebook) | Windows fleet venv, Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, RTX 5070 Ti (`cuda:0`), snapshot pre-staged | Package-API build record on the pinned sample: `sample_dataset` → baselines → `linear_probe` → `adapt` (defaults) → `evaluate` → `save_artifact` → `from_artifact` parity | load 7.9 s, probe 3.1 s, adapt 33.9 s | PASS — majority 0.5 / 0.0; GC 0.715 / 0.4308 (threshold 0.5598); frozen probe 0.815 / 0.6315; adapted **0.8425 / 0.6934** (epoch 2 kept; tp 153, tn 184, fp 16, fn 47); artifact 34,645,456 B / 32 tensors; parity 64/64 labels, max probability difference 0.0; 558 MiB peak. Pre-flight, not promotion evidence. |
| 2026-09-20 | package API at the candidate revision (not the notebook) | same venv, CPU (`cpu`) | Same path | load 4.5 s, probe 19.4 s, adapt 212.9 s | PASS — every metric identical to the GPU row to four decimals; artifact 34,645,456 B; parity exact. Pre-flight, not promotion evidence. |
| 2026-09-20 | `tests/` at the candidate revision | same venv (CPU and `cuda:0`) | `pytest`: 51 offline, model-backed and parity tests | — | 51 passed; `ruff` clean; `validate_release_assets.py` PASS; `build_notebook.py --check` clean. Static, not runtime evidence. |
| 2026-09-18 | `f9dc86f` / `9f67399f7bcf` (`TASK-INFERENCE`, superseded) | Kaggle Tesla T4, clean runtime | Default path of the earlier representation-only notebook | 218.6 s | PASS — 9/9 code cells; not evidence for the `E2E` blob |

## Current status

**Release-grade.** The `E2E` notebook blob `c1aa7361` (committed at `3e37b6c`) executed top-to-bottom in a clean Kaggle Tesla T4
runtime on 2026-09-20 (11/11 ok with 1 restart after the install cell, 248.3 s, 18 files / 224 MB fetched from the Hub and
digest-verified inside the notebook — the two model-code files before they were imported) with no repository checkout —
the REL1/REL10 supported-runtime evidence this file gates on. The package-API build records above are what preceded it
and remain history. Any later change to the carried modules or to the notebook template changes the blob and returns
the row to `Candidate` until a new clean run is recorded.

Facts a reviewer should weigh: the frozen probe is a strong reference (0.815 / 0.632) because promoter windows are
GC-rich and the representation already carries composition and more, so the fine-tuning's gain is 0.06 MCC — a few
times the run-to-run spread observed for the recipe, not a large margin; the 200-window validation split is noisy to
about ±0.03 MCC and the kept epoch can differ between runs (the build record kept epoch 2 of 6 while later epochs
overfit), so a Kaggle number a few hundredths off the build record would have been expected, not a finding — and in the event the
T4 run reproduced the RTX 5070 Ti and CPU build records to four decimals (accuracy 0.8425, MCC 0.6934, the same kept
epoch and the same confusion counts), although GPU kernel non-determinism can still move a probability near the
boundary on another device; the remote code runs in the executor's process, verified but not reviewed; and nothing here is a benchmark
reproduction — it is one seeded draw of one Genomic Benchmarks task under one window convention.
