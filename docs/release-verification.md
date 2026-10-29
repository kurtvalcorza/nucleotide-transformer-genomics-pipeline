# Release verification

`tutorials/nucleotide_transformer_colab.ipynb` (`E2E`, **standalone** carrier) is promoted from `Candidate` to
`Release-grade` only by the exact committed notebook blob executing top-to-bottom in a clean supported runtime with no
repository checkout, recorded in the tables below. Static checks, JSON validation, unit tests and code-cell
compilation are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8), and the package-API build
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
  profile, spec `2.2`, `standalone: true`, the pedagogical mode and the generating revision with the package digest;
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
   `numpy==2.5.3` (they are installed into the isolated environment Section 1 builds, so no interpreter restart is
   expected);
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
| `nucleotide_transformer_colab.ipynb` (`E2E`, 2026-10-17 requote) | `e2b0af9` / `06c6e6d8004f` | 2026-10-18 | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 | **PASSED** — one pass, no restart, 0 errors, 13/13 code cells, 246.6 s; REL13: **reproduction of the pre-registered evaluation** (hosted Colab T4, blob `06c6e6d8`); fresh-test MCC adapted 0.664, frozen probe 0.636, GC rule 0.36, adapted minus probe +0.0281 with 95% interval −0.0283 to +0.0864, all equal to the 2026-10-17 evaluation of record; see *Recorded executions* |
| `nucleotide_transformer_colab.ipynb` (`E2E`, 2026-10-17 tutorial-review fix; an earlier PR #8 head) | `d606232` / `b28c43a70791` | 2026-10-17 | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 | **PASSED** — one pass, no restart, 0 errors, 13/13 code cells, 252.1 s; REL13: pre-registered evaluation of record (first scoring of the fresh test); see *Recorded executions* |
| `nucleotide_transformer_colab.ipynb` (`E2E`, review fixes) | `7b53e06` / `fc2fd3663dfa` | 2026-10-08 | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 | REL13: earlier-blob history (pre-fix test split, no fresh test). **PASSED** — one pass, no restart, 0 errors, 12/12 code cells, 126.2 s; see *Recorded executions* |
| `nucleotide_transformer_colab.ipynb` (`E2E`) | `3e37b6c` / `c1aa7361` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-nucleotide-transformer` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` / `numpy 2.0.2` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` / `numpy 2.5.3` after, Python 3.12.13, `cuda:0`, driver 580.159.04) | **PASSED with a manual restart** (not a one-pass Run all: RUN1/RUN10, corrected 2026-10-08 after review NTP-M1) — 11/11 code cells ok (1 restart after the install cell, as the stale-import guard was designed to force); 18 files / 224 MB staged from the Hub into a clean cache, all 8 manifest entries fetched and digest-verified, `remote_code.verified_before_import` true; comparison accuracy {majority 0.5, gc_threshold 0.715, frozen_probe 0.815, adapted **0.8425**} / MCC {0.0, 0.4308, 0.6315, **0.6934**}; validation curve 0.505/0.071 → 0.795/0.609, 0.805/0.621 (epoch 2 kept), 0.795/0.606, 0.78/0.577, 0.765/0.533, 0.80/0.612; adapt 32.1 s; confusion tp 153 / tn 184 / fp 16 / fn 47; adapter 34,645,456 B / 32 tensors, licence `cc-by-nc-sa-4.0`; reload parity 64/64 labels, max probability difference 0.0 — every figure identical to the RTX 5070 Ti and CPU build records to four decimals |
| `nucleotide_transformer_colab.ipynb` (`TASK-INFERENCE`, superseded) | `f9dc86f` / `9f67399f7bcf` | 2026-09-18 | Kaggle Tesla T4 (clean runtime, empty Hugging Face cache, no repository checkout) | PASS — 9/9 code cells, 218.6 s; all 8 files downloaded and digest-verified including the two Python files before the remote-code import; nearest-centroid probe 0.9583 (n = 24) against majority 0.5 and GC 0.375. Evidence for the earlier representation-only notebook, not for the `E2E` blob. |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/nucleotide_transformer_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/nucleotide_transformer_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-18 (00:02:17 UTC start) | commit `e2b0af9` (`e2b0af916b620d6725fc2f709fa7c382681c6ccc`, PR #8 head) / blob `06c6e6d8004f63f1546c0c1a4d58ba5717f7c488` (`NOTEBOOK_SOURCE.repository_revision` `4fd5b20`, the requote commit; embedded `module_sha256` `6f0f49d6dd9d…`, generator `build_notebook.py/2.3`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 VM (session `suite-nucleotide-e2b0af9-1c69`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; isolated `uv` environment Python 3.12.12, 47 locked packages, built in 57 s; device `cuda:0` | default settings only (the pre-registered settings); forms, uploads, BYOD and the optional experiments not exercised | 246.6 s | **PASSED — one pass, no restart, 0 errors**, 13/13 code cells in order. REL13: **reproduction of the pre-registered evaluation** (hosted Colab T4, blob `06c6e6d8`): `preregistered` `True`, `settings_differing` `[]`, `test_look` `reproduction of the pre-registered evaluation`. Every printed metric line equals the `d606232` record: grid 0.5417 / 0.6679 / 0.6621, 2 blocks kept; fresh test accuracy {0.5, 0.68, 0.8175, **0.83**} / MCC {0.0, 0.36, 0.636, **0.664**}, difference +0.0281, CI [−0.0283, 0.0864], confusion 155 / 177 / 23 / 45; exposed development test 0.8425 / 0.6934; reload parity 64/64. Evidence in `docs/execution-evidence/2026-10-18-e2b0af9/` |
| 2026-10-17 (17:05:57 UTC start) | commit `d606232` (`d6062323786e5a5618772bc3000deb86c7f38349`, PR #8 head) / blob `b28c43a70791f14e78a656d887319eadc764ec24` (`NOTEBOOK_SOURCE.repository_revision` `4ddce2c`, the source revision the notebook was generated from; `d606232` is the regeneration commit; embedded `module_sha256` `f802c1f3d9e7…`, generator `build_notebook.py/2.3`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-nucleotide-d606232-1c69`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout, empty Hugging Face cache; kernel Python 3.13.15; isolated `uv` environment Python 3.12.12, 47 locked packages, built in 55 s; device `cuda:0` | default settings only (the pre-registered settings; form fields at their defaults); forms, uploads, BYOD and the optional experiments not exercised | 252.1 s | **PASSED — one pass, no restart, 0 errors**, 13/13 code cells in order (the four carried module cells print nothing). **REL13: pre-registered evaluation of record** (`preregistered` `True`, `settings_differing` `[]`, `test_look` `pre-registered evaluation (this runtime)`; first hosted run with the pre-registered settings). Grid (validation MCC, best epoch): 0 blocks 0.5417 (6), 2 blocks 0.6679 (2), 12 blocks 0.6621 (1); 2 blocks kept; grid 138.6 s. fresh test (400 windows, 200 / 200) accuracy {majority 0.5, GC rule 0.68, frozen probe 0.8175, adapted **0.83**} / MCC {0.0, 0.36, 0.636, **0.664**}; adapted minus probe MCC +0.0281, paired bootstrap 95% interval [-0.0283, +0.0864] (2,000 resamples, seed 0), which includes 0; exposed development test adapted 0.8425 / 0.6934 against probe 0.815 / 0.6315 and GC rule 0.715 / 0.4308; adapter 34,645,456 B / 32 tensors; reload parity 64/64, max probability difference 0.0. Fresh-test limitation (Amendment 3) applies; see the record section below. Evidence in `docs/execution-evidence/2026-10-17-d606232/`: `nucleotide_transformer_colab_d606232_colab-cli-t4_output.ipynb` sha256 `14e8fc69f6a7d5a70a23bb260a85a4eb0329706ea1caeaa2d2d556ca457250a0`, `run_summary.json` sha256 `4990bcf3390a50423249057d105d854059300b1ed797cd244c80a1deb820aa3e`, `exec.log` sha256 `0a9b1852484fce6288eb0f4691c46c5b5e1162b4259f9ceb9d8d04fe99201f29` |
| 2026-10-08 (10:54:48 UTC start) | commit `7b53e06` / blob `fc2fd3663dfaf8ddf6222011880115a036610374` (`NOTEBOOK_SOURCE.repository_revision` `b7866c7`, the source revision the notebook was generated from; `b7866c7..7b53e06` is the review-fix commit itself; embedded `module_sha256` `1b69d70c3d78…`, generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-nucleotide-7b53e06-1c69`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout, empty Hugging Face cache; kernel Python 3.13.15; isolated `uv` environment Python 3.12.12, 47 locked packages, built in 51 s; torch 2.14.0+cu130, transformers 4.57.6; device `cuda:0` | default settings only (`USE_BYOD = False`, `EPOCHS = 6`, `TRAINED_LAYERS = 2`, `SEED = 0`); optional journeys (BYOD, experiments) not exercised | 126.2 s | REL13: earlier-blob history (pre-fix validation and test split, no fresh test). **PASSED — one pass, no restart, 0 errors**, 12/12 code cells in order (the three carried module cells print nothing); 8 manifest files fetched and digest-verified, the model code before import, `encoder_is_base` true after the load; split 1,600 / 200 / 400; accuracy {majority 0.5, gc_threshold 0.715, frozen_probe 0.815, adapted **0.8425**} / MCC {0.0, 0.4308, 0.6315, **0.6934**}, verdict `probe > GC rule > majority`; validation curve 0.505/0.0709 → 0.795/0.6093, 0.805/0.6213 (epoch 2 kept), 0.795/0.6063, 0.78/0.5769, 0.765/0.5332, 0.80/0.6124; adapt 31.8 s; confusion tp 153 / tn 184 / fp 16 / fn 47; masked `ATTCCG` p = 0.6405; adapter 34,645,456 B / 32 tensors, licence `cc-by-nc-sa-4.0`; reload parity 64/64 labels, max probability difference 0.0 — the same figures as the 2026-09-20 Kaggle T4 run and the build records. Evidence in `docs/execution-evidence/2026-10-08-7b53e06/`: `nucleotide_transformer_colab_7b53e06_colab-cli-t4_output.ipynb` sha256 `f5951051bdcbd1698f554f98bd176f7534365d8cededbe17d1d2a8aa1bd23d69`, `run_summary.json` sha256 `d31195c328cae7a5ac3b964df27fe287ea20f3fe83909371094982f7b926293a`, `exec.log` sha256 `11f7fd567d4e19870e4e274d6117aefaf01b29bac61915fc19782a0bbcdd4a24` |
| 2026-09-20 | `3e37b6c` / `c1aa7361` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-nucleotide-transformer` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` / `numpy 2.0.2` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` / `numpy 2.5.3` after, Python 3.12.13, `cuda:0`, driver 580.159.04) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution; executor `.agent/scripts/kaggle-serial-test-suite.py`, fresh-subprocess `nbclient`) | 248.3 s (163.2 s install + restart, 85.0 s for the whole default path) | **PASSED with a manual restart** (not a one-pass Run all: RUN1/RUN10, corrected 2026-10-08 after review NTP-M1) — 11/11 code cells ok (1 restart after the install cell, as the stale-import guard was designed to force); 18 files / 224 MB staged from the Hub into a clean cache, all 8 manifest entries fetched and digest-verified, `remote_code.verified_before_import` true; comparison accuracy {majority 0.5, gc_threshold 0.715, frozen_probe 0.815, adapted **0.8425**} / MCC {0.0, 0.4308, 0.6315, **0.6934**}; validation curve 0.505/0.071 → 0.795/0.609, 0.805/0.621 (epoch 2 kept), 0.795/0.606, 0.78/0.577, 0.765/0.533, 0.80/0.612; adapt 32.1 s; confusion tp 153 / tn 184 / fp 16 / fn 47; adapter 34,645,456 B / 32 tensors, licence `cc-by-nc-sa-4.0`; reload parity 64/64 labels, max probability difference 0.0 — every figure identical to the RTX 5070 Ti and CPU build records to four decimals |
| 2026-09-20 | package API at the candidate revision (not the notebook) | Windows fleet venv, Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, RTX 5070 Ti (`cuda:0`), snapshot pre-staged | Package-API build record on the pinned sample: `sample_dataset` → baselines → `linear_probe` → `adapt` (defaults) → `evaluate` → `save_artifact` → `from_artifact` parity | load 7.9 s, probe 3.1 s, adapt 33.9 s | PASS — majority 0.5 / 0.0; GC 0.715 / 0.4308 (threshold 0.5598); frozen probe 0.815 / 0.6315; adapted **0.8425 / 0.6934** (epoch 2 kept; tp 153, tn 184, fp 16, fn 47); artifact 34,645,456 B / 32 tensors; parity 64/64 labels, max probability difference 0.0; 558 MiB peak. Pre-flight, not promotion evidence. |
| 2026-09-20 | package API at the candidate revision (not the notebook) | same venv, CPU (`cpu`) | Same path | load 4.5 s, probe 19.4 s, adapt 212.9 s | PASS — every metric identical to the GPU row to four decimals; artifact 34,645,456 B; parity exact. Pre-flight, not promotion evidence. |
| 2026-09-20 | `tests/` at the candidate revision | same venv (CPU and `cuda:0`) | `pytest`: 51 offline, model-backed and parity tests | — | 51 passed; `ruff` clean; `validate_release_assets.py` PASS; `build_notebook.py --check` clean. Static, not runtime evidence. |
| 2026-09-18 | `f9dc86f` / `9f67399f7bcf` (`TASK-INFERENCE`, superseded) | Kaggle Tesla T4, clean runtime | Default path of the earlier representation-only notebook | 218.6 s | PASS — 9/9 code cells; not evidence for the `E2E` blob |

### 2026-10-18 Colab CLI T4 run of `e2b0af9` (blob `06c6e6d8`): reproduction of the pre-registered evaluation

- **REL13 label:** reproduction of the pre-registered evaluation. The settings are identical to the `d606232` evaluation
  of record. `d606232..e2b0af9` changes quotes, labels and the `PREREGISTERED_EVALUATION_RECORD` pointer
  (`protocol.py`, `tools/notebook_template.py`, tests and docs); no setting or computation the scores depend on. The
  notebook printed `preregistered` `True`, `settings_differing` `[]` and
  `test_look` `reproduction of the pre-registered evaluation`.
- **Run:**
  - Colab CLI 0.7.4, fresh Colab Tesla T4 (Colab account 2), session `suite-nucleotide-e2b0af9-1c69`, VM
    `gpu-t4-s-kkb-usw4a2-151j0jhajfedd`.
  - Started 2026-10-18 00:02:17 UTC; CLI wall 246.6 s; exit 0; 13/13 code cells in order; 0 error outputs; VM released.
  - Source identity (`inspect_run.py`): 36/36 cells, same ids and order as blob `06c6e6d8`, sources identical apart
    from form-parameter lines.
  - Boundary: CLI sequential execution, not a browser Run all; forms, uploads, BYOD and the optional experiments were
    not exercised.
- **Match with the record:** a line-by-line comparison of the 86 printed output lines of the two `exec.log` files
  differs only in:
  - the embedded revision and `module_sha256` (`4fd5b20` / `6f0f49d6…` vs `4ddce2c` / `f802c1f3…`);
  - the `test_look` label;
  - timings: environment 57 vs 55 s; grid arms 25.1 / 35.2 / 67.0 vs 26.2 / 37.8 / 74.6 s; embed 0.3 vs 0.292 s;
    probe 4.3 vs 4.4 s; data load 0.012 vs 0.011 s; CLI wall 246.6 vs 252.1 s;
  - the worker pid.
- **Equal to the record:**
  - grid 0.5417 (epoch 6) / 0.6679 (epoch 2) / 0.6621 (epoch 1), 2 blocks kept;
  - fresh test accuracy 0.5 / 0.68 / 0.8175 / 0.83 and MCC 0.0 / 0.36 / 0.636 / 0.664;
  - adapted minus probe +0.0281, paired bootstrap 95% interval −0.0283 to +0.0864 (2,000 resamples, seed 0);
  - adapted confusion tp 155 / tn 177 / fp 23 / fn 45, and the per-class rows;
  - the Amendment 3 limitation line;
  - exposed development test 0.8425 / 0.6934 (153 / 184 / 16 / 47);
  - reload parity 64/64, maximum probability difference 0.0.
- **Evidence:** `docs/execution-evidence/2026-10-18-e2b0af9/`, byte-for-byte copies of the runner's files (`-text`
  applies); SHA-256 checked in the run VM directory, the copy and the commit.
  - `nucleotide_transformer_colab_e2b0af9_colab-cli-t4_output.ipynb`: `4ef1584b50a93e193c940b12426581b02d9ec3b1f3597ec4cd32e75eb238288e`
  - `run_summary.json`: `6b0ae0df72ad7085df6c891d730cafa779eebad9edc5881d662e2048eae70aa0`
  - `exec.log`: `c75d4256e5a3f9f7b68a71ce01c5d10551b5e71d6e89e3a8b8a0615d574f35b1`
- **Status:** stays **Candidate** (REL12 hosted BYOD still open). The `d606232` run remains the evaluation of record.

### 2026-10-17 Colab CLI T4 run of `d606232` (blob `b28c43a7`): pre-registered evaluation of record

- **Run:** Colab CLI 0.7.4, fresh Colab Tesla T4 (Colab account 2), session `suite-nucleotide-d606232-1c69`, started 2026-10-17 17:05:57 UTC, CLI wall 252.1 s, exit 0, 13/13 code cells in order, 0 error outputs, VM released after the run. Executor boundary: CLI sequential execution (`colab exec -f`), not a browser Run all; forms, uploads, BYOD and the optional experiments were not exercised. Source identity: the executed notebook has the same 36 cells with the same ids and order as blob `b28c43a7`, sources identical apart from form-parameter lines (`inspect_run.py`).
- **Environment and model:** isolated `uv` environment (Python 3.12.12, 47 locked packages) built in 55 s, kernel Python 3.13.15, device `cuda:0`; all 8 manifest files of `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` at `81b29e57…` fetched and digest-verified before the model code was imported; `remote_code_executed` `True`, `encoder_is_base` `True`.
- **Data checks:** 0 validation windows overlap training; 0 fresh-test windows within 251 bases of another window; 121 of 400 exposed-test windows overlap training or validation (119 of them negatives); fresh test 400 windows (200 / 200), 24 chromosomes, digest `338824e2…`.
- **Evaluation status (REL13):** the notebook printed `preregistered` `True`, `settings_differing` `[]` and `test_look` `pre-registered evaluation (this runtime)`. This is the first hosted run of the pre-registered settings and the first time any model scored the fresh test, so it is **the pre-registered evaluation of record**. Later hosted runs with identical settings are reproductions.
- **Grid (validation, 200 re-drawn windows; lr 3e-5, 6 epochs each):** 0 blocks MCC 0.5417 / accuracy 0.77 (epoch 6, 26.2 s); 2 blocks 0.6679 / 0.825 (epoch 2, 37.8 s); 12 blocks 0.6621 / 0.825 (epoch 1, 74.6 s); rule kept **2 blocks** (no larger configuration beats 2 blocks by at least 0.03 validation MCC). Grid 138.6 s. Kept run's validation curve 0.505 / 0.0709 → 0.805 / 0.6335 → 0.825 / 0.6679 → 0.815 / 0.6543 → 0.80 / 0.6250 → 0.78 / 0.5618 → 0.82 / 0.6593. Frozen logistic probe on validation (reference, selects nothing): 0.815 / 0.6326.
- **Fresh test (the held-out result):** accuracy majority 0.5, GC rule 0.68, frozen probe 0.8175, adapted **0.83**; MCC 0.0, 0.36, 0.636, **0.664**. Adapted minus frozen-probe MCC +0.0281, paired percentile bootstrap 95% interval −0.0283 to +0.0864 (2,000 resamples, seed 0): the interval includes 0, so on the fresh test the adapted model's gain over the frozen probe is **not resolved**; both beat the GC rule and the majority baseline. Adapted confusion tp 155 / tn 177 / fp 23 / fn 45 (promoter precision 0.8708, recall 0.775; negative precision 0.7973, recall 0.885); frozen probe promoter recall 0.79, negative recall 0.845. The printed verdict (`adapted_beats_frozen` `True`, MCC gain 0.028) is a point comparison. Re-derived from the printed counts: adapted accuracy 0.83 and MCC 0.6640; the probe's counts from its per-class recall (tp 158 / tn 169 / fp 31 / fn 42) give 0.8175 / 0.6360. The GC rule's 0.68 / 0.36 equals the disclosed rule-only development look below (the rule is fitted on training alone, so it is deterministic). The bootstrap interval is as printed; the per-record predictions stayed on the released VM.
- **Fresh-test limitation (Amendment 3, recorded before this score and printed by the run):** fresh negatives cluster near training negatives (nearest training window median 612 bases; 134 of 200 within 1 kb), positives do not (median 527,822 bases). Regional clustering by class may inflate the fresh-test score, especially on the negative class.
- **Exposed development test (not independent; in view during the build):** majority 0.5 / 0.0; GC rule 0.715 / 0.4308 (threshold 0.5598); frozen probe 0.815 / 0.6315; adapted 0.8425 / 0.6934 (tp 153 / tn 184 / fp 16 / fn 47).
- **Against the round-3 CPU build check (fresh test masked):** the 0- and 2-block arms, the kept run's validation curve, the probe's validation reference and every exposed-development-test figure are equal to four decimals. The 12-block arm differs: 0.6621 / 0.825 at epoch 1 on the T4 against 0.6628 / 0.82 at epoch 4 on CPU (GPU against CPU numerics on the full-encoder arm); the selection is unchanged (12 blocks would have needed 0.6979). Against the 2026-10-08 T4 run of the pre-fix blob: exposed-test figures equal; validation differs by design (re-drawn windows).
- **Artifact and reuse:** adapter 32 tensors, 34,645,456 B, licence `cc-by-nc-sa-4.0`, sha256 `fe9305b0b53942b8…`; reload parity 64/64 identical labels, maximum probability difference 0.0, reloaded best epoch 2, 2 layers. Masked prediction `ATTCCG` p = 0.64054.
- **Evidence:** `docs/execution-evidence/2026-10-17-d606232/`: `nucleotide_transformer_colab_d606232_colab-cli-t4_output.ipynb` sha256 `14e8fc69f6a7d5a70a23bb260a85a4eb0329706ea1caeaa2d2d556ca457250a0`, `run_summary.json` sha256 `4990bcf3390a50423249057d105d854059300b1ed797cd244c80a1deb820aa3e`, `exec.log` sha256 `0a9b1852484fce6288eb0f4691c46c5b5e1162b4259f9ceb9d8d04fe99201f29` (byte-for-byte copies of the runner's files; `-text` applies). Status stays **Candidate** (REL12 hosted BYOD still open).

## Current status

**Candidate.** On 2026-10-17 the tutorial-review-fix blob `b28c43a7` (commit `d606232`) passed in one pass with no restart and 0 errors on a fresh Colab Tesla T4 under the Colab CLI (13/13 code cells, 252.1 s). That run is the pre-registered evaluation of record (REL13): the grid kept 2 blocks (validation MCC 0.5417 / 0.6679 / 0.6621); fresh test MCC adapted 0.664 against frozen probe 0.636, GC rule 0.36 and majority 0.0 (adapted minus probe +0.028, 95% interval −0.028 to +0.086, so the gain over the probe is not resolved); exposed development test adapted 0.8425 / 0.6934. The fresh-test limitation of Amendment 3 applies (negatives cluster near training negatives). On 2026-10-18 the requote blob `06c6e6d8` (commit `e2b0af9`, the PR #8 head) ran on a fresh Colab Tesla T4 (Colab CLI, 13/13 code cells, 0 errors, 246.6 s). That run is a **reproduction of the pre-registered evaluation** (REL13): every printed metric equals the record.

The current blob `854abd87` (code commit `aa738ca`) shows a refused BYOD input as exactly one `Refused: <file>: <reason>` line (previously a differently prefixed line), with no traceback, and stops Run all at that cell; genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20). Its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD=True;BYOD_PATH='/etc/hostname'`) are pending, and the hosted runs of earlier blobs, including `06c6e6d8`, are history for it (REL14).

REL13: earlier-blob history. The notebook was regenerated for the 2026-10-02 Notebook Review Framework v1 review (`docs/reviews/2026-10-02-notebook-review/`): Section 1 now builds an isolated, hash-locked environment (no install into the kernel, no restart), `adapt` always starts from the pinned base and `reset_to_base` undoes an adaptation, the quality comparisons are recorded verdicts instead of asserts, the BYOD minimum is stated as Section 4 enforces it, and the spec 2.2 guided layer was added. The review-fix blob `fc2fd3663dfa` (commit `7b53e06`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-08 (Colab CLI sequential execution, 12/12 code cells, 126.2 s; test MCC adapted 0.6934 against frozen probe 0.6315, GC rule 0.4308 and majority 0.0, epoch 2 kept, reload parity 64/64) — record above. The 2026-09-20 Kaggle T4 run of the earlier blob `c1aa7361` needed a manual restart after the install cell, so it never met RUN1/RUN10 and is kept as history only.

Remaining gates (the status stays **Candidate**, REL14): the REL12 BYOD run (a representative CSV, an incompatible one, through export and reload); a clean hosted CPU run for the stated default runtime (UX12), or keep naming the T4 GPU as the verified runtime.

Facts a reviewer should weigh: the frozen probe is a strong reference (0.815 / 0.632) because promoter windows are GC-rich and the representation already carries composition and more, so the fine-tuning's gain on the fresh test is +0.028 MCC from one seed, with a 95% interval of −0.028 to +0.086 that includes 0 (REL13 evaluation of record, 2026-10-17); the 0.06 MCC is the exposed development test. No multi-seed spread is recorded, so whether that gain exceeds seed-to-seed variation is untested, and on 400 windows the sampling error of MCC is of a similar size. The records that exist (RTX 5070 Ti, workstation CPU, Kaggle T4) agree to four decimals under `SEED = 0`, although GPU kernel non-determinism can still move a probability near the boundary on another device; the remote code runs in the executor's process, verified but not reviewed; and nothing here is a benchmark reproduction — it is one seeded draw of one Genomic Benchmarks task under one window convention.

## 2026-10-17 tutorial-review fix (row 58): CPU build check (development look)

REL13: CPU development look (fresh test masked; disclosed, score-once rule item 8). The 2026-10-17 Colab T4 run recorded above is the evaluation of record.

CPU build check of the 2026-10-17 review fix (workstation CPU, torch 2.14.0+cpu, seed 0; the notebook's own cells run in order with the fix-cycle venv standing in for Section 1; **not clean-runtime evidence**; the fresh test was **masked** per SCORE_ONCE_RULE item 8). Grid: 0 blocks: validation MCC 0.5417 / accuracy 0.77, epoch 6, 187.7 s; 2 blocks: validation MCC 0.6679 / accuracy 0.825, epoch 2, 246.3 s; 12 blocks: validation MCC 0.6628 / accuracy 0.82, epoch 4, 501.1 s; rule kept **2 blocks** (no larger configuration beats 2 blocks by at least 0.03 validation MCC). Kept run: epoch 2, validation 0.825 / MCC 0.6679 (re-drawn validation). Exposed development test (old 400 test windows; pre-fix → now): majority 0.5 / 0.0 → 0.5 / 0.0; GC rule 0.715 / 0.4308 → 0.715 / 0.4308; frozen probe 0.815 / 0.6315 → 0.815 / 0.6315; adapted 0.8425 / 0.6934 → 0.8425 / 0.6934. Reload parity: 64 of 64 identical labels and a maximum probability difference of 0. Fresh test: not scored on CPU; it is first scored on the hosted T4 run of this revision.

Development looks disclosed (SCORE_ONCE_RULE item 8): none on the fresh test with any model. One "development look" on the fresh test, without a model: a stand-in unit test (uncommitted working tree, 2026-10-17) printed the training-fitted GC-threshold rule on the fresh test once, 0.68 accuracy / 0.36 MCC, before the tests were changed to stub fresh-test baselines; nothing was changed in response.

Evaluation of record: the first hosted T4 run of this revision with the pre-registered settings — the 2026-10-17 Colab CLI T4 run of `d606232` (blob `b28c43a7`) recorded above. Later identical hosted runs are reproductions.

**Fresh-test limitation (Amendment 3, 2026-10-17).** The fresh test keeps its 251-base margin: at a 1,000-base margin the never-scored pool yields only 49 negatives (126 meet the margin to the sample, 49 of them to each other), because the source tiles its negative windows. Fresh-test negatives therefore sit close to training negatives (distance to the nearest training window: min 255, 10th percentile 290, median 612 bases; 134 of 200 within 1 kb, 174 within 5 kb), while positives do not (median 527,822 bases; 1 of 200 within 1 kb). Regional clustering by class may inflate the fresh-test score, especially on the negative class. The depth grid compares 0, 2 and 12 trained blocks at one budget (lr 3e-5, 6 epochs); the head-only arm peaked at its last epoch, so it is an under-trained comparator. Recorded before any fresh-test score; see `docs/nucleotide-transformer-preregistration.md` Amendments 2 and 3.

Round-3 CPU build check (2026-10-17, same harness, fresh test masked): the grid reproduced exactly (validation MCC 0 → 0.5417 at epoch 6, 2 → 0.6679 at epoch 2, 12 → 0.6628 at epoch 4; 2 kept); the frozen logistic probe's validation MCC, printed as a non-selecting reference, is 0.815 / 0.6326; the model-state check re-scored validation at the in-loop batch and matched; exposed development test unchanged (adapted 0.8425 / 0.6934); the pre-registered branch, bootstrap interval included, ran with the exposed test standing in for the fresh test.
