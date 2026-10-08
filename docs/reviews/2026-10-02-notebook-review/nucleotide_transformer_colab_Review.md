# nucleotide_transformer_colab.ipynb — Notebook Review (Framework v1)

**Readiness: Needs revision.** The default path runs. The current notebook blob is the one recorded as passing on a
clean Kaggle T4 run, and its promoter fine-tuning lesson is well built and honestly bounded. Four problems hold it
back. Run all needs a manual restart after the install cell (NTP-M1). Re-running Sections 6–7 for the documented
experiments or for BYOD silently reuses an encoder that was already fine-tuned in place, so the "frozen" probe and
the "head-only" experiment measure the wrong thing (NTP-M2). Hard `assert`s stop the notebook whenever an honest
result does not rank the models in the expected order (NTP-M3). BYOD rejects any dataset smaller than 50 distinct
sequences, although the notebook says the minimum is 8 (NTP-m1).

Findings: 0 Blocker, 3 Major, 6 Minor, 3 Suggestion. Prefix `NTP`.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Repository | `kurtvalcorza/nucleotide-transformer-genomics-pipeline` |
| Notebook | `tutorials/nucleotide_transformer_colab.ipynb` (25 cells: 11 code, 14 markdown) |
| Reviewed revision | `origin/main` = `8264ecacda9a7780e4e32cc60bd74ebc4e057464` (GitHub API `commits/main`, 2026-10-04) |
| Notebook blob | `c1aa73616c40519cdb4d91f29925caad2caca1a8`, identical to the blob at `3e37b6c` that the Kaggle run recorded |
| Spec baseline | NOTEBOOK_SPEC **2.2** (ml-worker `origin/main` `b1cfe13`). The notebook declares spec `2.0` in `metadata.dimer` |
| Profile / mode | `E2E` / `GUIDED`, standalone carrier (3 package modules carried verbatim; generator `tools/build_notebook.py` + `tools/notebook_template.py`) |
| Audience / prerequisites | Basic Python; DNA, GC content and promoters; accuracy and MCC; probe versus fine-tuning (cell 1) |
| Supported runtime | Colab or Jupyter, Python 3.12; CPU (float32) by default, CUDA used when present |
| Promised outcomes | Pinned install; digest-verified snapshot including remote code; inline 2,200-window promoter sample split 1,600/200/400; embeddings plus masked prediction; majority, GC and frozen-probe baselines; bounded fine-tuning (head + last 2 blocks) with validation selection; held-out accuracy and MCC; safetensors adapter export and reload parity; optional BYOD CSV and optional experiments |

### Evidence actually obtained

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Clear contract, licence and remote-code perimeter. The guided scaffolding is partial, and 700 KB of carried cells are not marked as infrastructure (NTP-m4) |
| Clean default | Documented execution evidence | Kaggle Tesla T4, 2026-09-20, the same blob `c1aa7361`: 11/11 cells ok **with 1 restart after the install cell**, MCC 0.6934 adapted against 0.6315 frozen. Never run on Colab. The stated default runtime (CPU) has no clean-runtime record (NTP-m5). Not re-executed in this review |
| Active learning | Direct execution, **reduced scale, CPU** (local conda env, torch 2.13.0+cpu, transformers 4.57.6; 48 train / 32 test windows, 1 epoch); source inspection | After `adapt`, embeddings change by up to 0.096 and the block-11 tensor stays 9.0e-5 away from base after a `TRAINED_LAYERS=0` rerun (NTP-M2). Full-scale experiments **not verified** |
| Reuse and recovery | Direct execution, reduced scale, CPU (no model needed for P1); documented evidence for reload | BYOD with 8–49 distinct sequences fails in cell 13. A GC-defined stand-in BYOD set fails the Section 6 assert (GC MCC 1.0 against probe MCC 0.667). The upload widget itself was not exercised. Artifact reload parity 64/64 is documented on Kaggle |

Limitations: no GPU, no Colab and no full-scale run were used. The local torch version (2.13.0) differs from the pin
(2.14.0). The model snapshot came from the local clone's pre-staged, digest-verified `weights/` directory. Probe P0
confirmed that the carried cells equal the package modules apart from the documented standalone rewrites.

## 2. Separate judgments

- **Technical correctness:** the default path is sound. Verification happens before the remote-code import, the
  splits are disjoint, and the artifact round trip is correct. In-place mutation of the shared `pipe` breaks every
  documented rerun (NTP-M2). The stale-import guard turns a required restart into a hard stop (NTP-M1).
- **Promise fulfilment:** the default promises are met on the recorded T4 run. Three promises fail: BYOD's
  "at least eight sequences" (NTP-m1), "BYOD flows through … the same cells" for any data that does not reproduce
  the expected ranking (NTP-M3), and "Run all … no configuration edit" without a restart (NTP-M1).
- **Learner experience:** the scientific framing is unusually careful (baselines first, MCC as the headline metric,
  limits stated). The optional experiments are one sentence with no rerun instructions (S2), and their results
  would be contaminated (NTP-M2).
- **Spec conformance:** unresolved applicable MUSTs are RUN1, RUN10 and ENV6 (restart); DAT12, DAT14, DAT19 and REL12
  (BYOD limit, BYOD downstream path, BYOD verification); and SRC2 (hidden state dependency on rerun). Spec 2.2's
  GDL layer is partial (SHOULD).

## 3. Findings

### NTP-M1 — Major: Run all needs a manual restart after the in-kernel install

- **Cell/section:** cell 3 (Section 1), generated by `tools/build_notebook.py:61–69`.
- **Observed issue:** `pip install` of `torch==2.14.0`, `numpy==2.5.3` and the other pins runs into the live kernel.
  When a loaded distribution changes, the guard raises `RuntimeError(... Restart the runtime, then rerun from the top.)`.
  Hosted images ship a different torch/numpy, so the guard always fires on first run.
- **Consequence:** a learner who chooses **Run all** stops at cell 3 and has to restart and run again. The notebook
  still records the run as "Run-all: yes" and Release-grade.
- **Evidence:** documented execution evidence. `docs/release-verification.md` records the Kaggle T4 run as "11/11
  code cells ok (1 restart after the install cell, as the stale-import guard is designed to force)", and procedure
  step 4 says "an interpreter restart after the install is expected". `tutorials/README.md` lists "Run-all: yes".
- **Recommended correction:** adopt the fleet's uv isolated-environment pattern. A carrier cell bootstraps uv, runs
  `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked requirements file with
  `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that env, so the kernel's preloaded
  torch and numpy are never replaced. Reference:
  `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on origin/main.
  Change it in `tools/build_notebook.py` (install cell) and regenerate.
- **Acceptance check:** a fresh Colab (or Kaggle) runtime completes Run all with no error output in any cell and no
  manual restart. The release record states "no restart".
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.

### NTP-M2 — Major: documented reruns reuse an encoder already fine-tuned in place

- **Cell/section:** cells 17, 19 and 13 on rerun. The cause is `NucleotideTransformerPipeline.adapt` in
  `src/nucleotide_transformer_genomics_pipeline/pipeline.py`.
- **Observed issue:** `adapt` trains the last `layers` blocks of the shared `self._model` and on success loads the
  best epoch's tensors **into that model** (`pipeline.py:565`). The base weights (`frozen_state`) are restored only on
  exception (`:574`). `embed` and `linear_probe` read the same model. After the default run (which keeps epoch 2),
  every documented rerun therefore starts from promoter-tuned blocks 10–11:
  - **Optional experiments** (closing section): "set `TRAINED_LAYERS = 0` and read how much of the gain the head alone
    recovers" trains a head on top of the already fine-tuned blocks. `TRAINED_LAYERS = 12` and a higher `EPOCHS`
    continue from them too.
  - **BYOD** ("re-run from that cell", Section 4): the Section 6 "frozen probe" embeds through the promoter-tuned
    encoder. The Section 7 adaptation starts from it, and the exported adapter carries promoter-trained blocks
    labelled as an adaptation of the pinned base on the learner's data.
- **Consequence:** the comparisons the notebook asks learners to make (head only against two blocks; frozen against
  adapted on their own data) are invalid and look normal. Nothing tells the learner that state carried over.
- **Evidence:** source inspection (above). Direct execution, reduced scale, CPU (probe P3): after
  `adapt(layers=2, epochs=1)` the mean-pooled embeddings of 4 held-out windows changed by up to **0.0964**. After a
  following `adapt(layers=0)` the block-11 query weight still differed from base by **9.0e-5**, while the result
  reported `layers: 0` and 263,682 trainable parameters.
- **Recommended correction:** make `adapt` start from the verified base every time. Either restore `frozen_state` for
  all encoder blocks before training, or keep a pristine base copy and expose `pipe.reset_to_base()`. Have
  `linear_probe` refuse, or reload, when an adapter has modified the encoder. Alternatively, have the notebook's
  BYOD and experiment instructions re-run from Section 3, or create a fresh `pipe` before Section 6. Fix it in the
  package (then regenerate) and the rerun text in `tools/notebook_template.py:52–53, 454`. Add a model-backed test:
  `adapt(layers=2)` followed by `adapt(layers=0)` leaves block 11 equal to base, and `embed` after `adapt` equals
  `embed` before it.
- **Acceptance check:** after a full default run, re-running Section 7 with `TRAINED_LAYERS = 0` (and, separately,
  re-running from Section 4 with BYOD) gives the same frozen-probe metrics and the same `adapt` result as a
  fresh-runtime run with those settings.
- **Spec:** SRC2, DAT13, DAT14, UX5, GDL10.

### NTP-M3 — Major: hard asserts make the expected model ranking a precondition

- **Cell/section:** cell 17 `assert frozen_probe['mcc'] > baseline_gc['mcc'] > baseline_majority['mcc']`; cell 21
  `assert adapted_test['mcc'] > frozen_probe['mcc']` and `assert verdict['adapted_beats_baselines']`. Source:
  `tools/notebook_template.py:270, 356–357`.
- **Observed issue:** the notebook promises that BYOD goes "through the same validation, … baselines, frozen probe,
  fine-tuning, held-out evaluation, artifact export and reload-parity cells". Its interpretation section tells
  learners that "if the probe already matches the fine-tuning, the representation was the answer". Either honest
  outcome raises `AssertionError`, and Sections 7–9 (or the export and artifact steps) never run. The same applies to
  the `TRAINED_LAYERS = 0` experiment if the head alone does not beat the probe.
- **Consequence:** BYOD only completes for data that reproduces the tutorial's ranking. A learner whose data shows
  the representation or composition is enough gets a crash instead of the conclusion the notebook teaches. This is
  framework dimension 5's "assume a predetermined winning model".
- **Evidence:** direct execution, reduced scale, CPU (probe P4). 80 synthetic 251-base windows labelled by
  GC ≥ 0.5 went through `split_dataset(seed=42)`: majority MCC 0.0, GC-threshold MCC **1.0**, frozen probe MCC
  **0.667**, so the cell 17 assert evaluates `False`. Source inspection covers cell 21.
- **Recommended correction:** keep the assertions for the pinned sample only (`if not USE_BYOD:`), or replace them
  with the `evaluation_report` verdict printed as an observation, plus a "what it means if this is False" note. Write
  the evaluation report and the artifact in every case.
- **Acceptance check:** a BYOD CSV where the GC rule beats the probe (such as the P4 stand-in with ≥ 50 rows)
  completes Sections 4–9, writes `evaluation_report.json` with `frozen_beats_baselines: false`, and exports and
  reloads the adapter. The pinned-sample default still checks its expected ordering.
- **Spec:** DAT14, REL12, RUN9, UX7.

### NTP-m1 — Minor: BYOD rejects datasets under 50 sequences, though the stated minimum is 8

- **Cell/section:** cell 0 BYOD paragraph ("at least eight sequences", `tools/notebook_template.py:53`) and cell 13.
- **Observed issue:** `split_dataset` gives 20% to test and 15% to validation. Cell 13 then calls
  `validate_dataset` on **each** split with the default `min_records=8`. Any dataset with fewer than 50 distinct
  sequences fails, with a message such as `"2 records; 8..20000 are required"` that names neither the split nor the
  real minimum.
- **Evidence:** direct execution (probe P1, no model). n = 8, 10, 20, 40 and 49 raise `ValueError`. The smallest
  accepted n is **50** (split 32/8/10).
- **Recommended correction:** state the real minimum (≥ 50 distinct sequences, both labels in train) before upload,
  or validate the val/test splits with `min_records=1` as `linear_probe`/`evaluate` already do. Name the split in the
  error.
- **Acceptance check:** the stated minimum equals the smallest n that cell 13 accepts. A too-small upload gets a
  message naming the split and the minimum.
- **Spec:** DAT12, DAT19, UX10. Two MUSTs, so release fails on conformance.

### NTP-m2 — Minor: Release-grade without REL12 BYOD verification

- **Observed issue:** `docs/release-verification.md` records only the default path. No run shows the BYOD branch
  accepting representative input, rejecting an incompatible input and reaching export and reload.
- **Consequence:** NTP-M2, NTP-M3 and NTP-m1 went undetected through promotion.
- **Evidence:** source inspection of `docs/release-verification.md` (no BYOD row).
- **Recommended correction:** after the fixes above, record one BYOD execution (for example the generated
  `outputs/nucleotide_transformer_train.csv`, a GC-defined CSV and one invalid CSV), each through cell 23.
- **Acceptance check:** a BYOD row exists in the release record with the outcome of each stage.
- **Spec:** REL12 (MUST).

### NTP-m3 — Minor: run-to-run variability claims are inconsistent and unsupported

- **Cell/section:** cell 18 ("the same recipe lands within a few hundredths of the numbers below from one GPU run
  to the next", `notebook_template.py:291`); cell 24 ("The 0.06 MCC gain … is a few times the run-to-run spread the
  build record observed", `:437`); cell 20 ("no dispersion estimate").
- **Observed issue:** the only records (RTX 5070 Ti, workstation CPU, Kaggle T4) are **identical to four decimals**
  under one seed, and no multi-seed spread is recorded anywhere. The "few times the spread" argument therefore rests
  on a number that is not shown. The notebook also says it has no dispersion estimate.
- **Consequence:** learners are told the 0.06 MCC gain is meaningful on the strength of an unpublished spread, for a
  400-window test set where the sampling error of MCC is of similar size.
- **Evidence:** source inspection of cells 18, 20 and 24 against `docs/release-verification.md`.
- **Recommended correction:** either record the seed spread (for example `SEED` 0–4) and cite it, or remove the
  claim and say the gain's significance is untested. See S1.
- **Acceptance check:** every variability statement in the notebook cites a recorded measurement or is labelled as
  untested.
- **Spec:** ENV8, ENV9.

### NTP-m4 — Minor: guided layer partial; carried infrastructure not labelled

- **Observed issue:** against spec 2.2 §3.5 there is no **How to use this notebook**, no roadmap, glossary, learner
  predictions, collapsible **Check your reasoning** answers, troubleshooting section or conclusion template. The
  three carried module cells (4,791 + 37,808 + **659,324** characters) are introduced as package code but not marked
  **Infrastructure**, and the learner scrolls past about 700 KB to reach Section 3. The metadata declares spec 2.0.
- **Evidence:** source inspection; cell sizes from probe P0.
- **Recommended correction:** add the GDL elements in `tools/notebook_template.py`. Label cells 5, 7 and 9 as
  Infrastructure (collapsed, or with a one-line "skip this" note). Bump the declared spec when migrating.
- **Acceptance check:** the GDL checklist in spec §26 passes item by item.
- **Spec:** GDL2, GDL3, GDL6, GDL7, GDL9, GDL11, GDL13, GDL14 (SHOULD).

### NTP-m5 — Minor: the stated default runtime (CPU) and Colab have no clean-runtime record

- **Observed issue:** the notebook and `tutorials/README.md` give CPU as the default runtime and Colab as the
  supported path, and tell learners to "expect a few tens of minutes on a 2-vCPU hosted runtime". The only
  clean-runtime record is a Kaggle **T4 GPU** run. The CPU figure (213 s fine-tuning) is a workstation package-API
  pre-flight, not the notebook.
- **Evidence:** documented execution evidence (release-verification tables).
- **Recommended correction:** record one clean Colab CPU run (or label the CPU timing as an untested estimate and
  name GPU as the verified runtime).
- **Acceptance check:** a release row exists for the stated default runtime, with wall time.
- **Spec:** UX12, REL1, REL10.

### NTP-m6 — Minor: Section 3 prints a hard-coded weight "source"

- **Cell/section:** cell 11, `print({'device': getattr(pipe, 'device', None), 'source': getattr(pipe, 'source', 'local-snapshot')})`.
- **Observed issue:** `NucleotideTransformerPipeline` has no `source` attribute (probe P2: `has_source_attr: false`),
  so the cell always prints the literal fallback. The markdown promises that "the effective identity, device and
  weight source are printed". The value happens to be true, but it is not observed.
- **Recommended correction:** print `snapshot['revision']`, `WEIGHTS_DIR` and `pipe.weight_sha256` (all available)
  in place of a getattr fallback.
- **Acceptance check:** every value in the Section 3 summary comes from an object the cell computed.
- **Spec:** MOD3, UX3.

### Suggestions

- **NTP-S1:** report a paired uncertainty for "adapted against frozen probe" on the 400 test windows (bootstrap CI of
  the MCC difference, or McNemar on the discordant pairs). The verdict's 0.06 gain then carries its own error bar.
- **NTP-S2:** turn the closing "Optional experiments" sentence into one **Predict → Change one thing → Run → Observe →
  Explain** activity. Name the exact cells to re-run, give the expected reading and add a reset step (GDL10). This
  depends on NTP-M2.
- **NTP-S3:** add a `BYOD_CSV_PATH` form field beside `files.upload()` so executors and learners on plain Jupyter
  can run BYOD non-interactively (EXE1, EXE2).

## 4. Positive findings and non-findings

- The digest-before-import remote-code perimeter, the explicit licence carry-through into the adapter manifest, and
  the refusal of `pytorch_model.bin` are exemplary (MOD6–MOD9, ART5).
- The baselines are well chosen and honestly framed: a GC rule fitted on train only, a frozen probe with its
  standardisation fitted on train (`pipeline.py:401–402`, SPL8), and MCC as the headline metric.
- The per-class figures in the Section 8 prose (precision 0.905, recall 0.765, negative recall 0.92) match the
  recorded confusion tp 153 / fp 16 / fn 47 / tn 184.
- Masked prediction behaves as described. At reduced scale on CPU, `ATTCCG` is top-1 at p = 0.64 inside the
  repeated context (probe P5).
- The carried cells equal the package modules apart from the 4 documented standalone rewrite lines (probe P0).

## 5. Readiness and correction order

**Needs revision.** Gates: NTP-M1 (uv isolated env), NTP-M2 (reset-to-base), NTP-M3 (conditional asserts), then
NTP-m1 and NTP-m2 (BYOD limit plus REL12 record), then a fresh clean-runtime run of the new blob. The T4 record will
not cover the regenerated notebook.

## 6. Probe inventory (`nucleotide_transformer_colab_Review_Probes.zip`)

`run_probes.py` (P0 carried-module parity and cell sizes; P1 BYOD minimum through cell 13's logic; P2 load; P3 state
carry-over after `adapt`; P4 a GC-defined BYOD stand-in through the cell 17 assert; P5 masked prediction),
`results.json` and `source_manifest.json` (sha256 of the 10 inspected files at `8264eca`). Environment: Windows,
conda `eo-notebook-test`, Python 3.12.14, torch 2.13.0+cpu, `CUDA_VISIBLE_DEVICES=-1`, 4 threads. All reduced scale.

## 7. Verified versus inferred

- **Verified (direct execution, reduced, CPU):** BYOD minimum of 50 (P1); embeddings and blocks stay mutated after
  `adapt` and a `layers=0` rerun (P3); the cell 17 assert fails on a composition-defined dataset (P4).
- **Verified (documented):** the default path passed on Kaggle T4 for this exact blob, with one manual restart.
- **Inferred:** that the contamination in NTP-M2 changes full-scale metrics materially (only an embedding shift of up
  to 0.096 was measured, at 1 epoch on 48 windows); that Colab's preinstalled torch triggers the guard (true for
  Kaggle, and expected for any image not at torch 2.14.0).
- **Most likely to be wrong:** the severity of NTP-M2. If full-scale reruns turned out to move the probe and
  head-only numbers only negligibly, it would be closer to Minor. The mechanism itself is confirmed.

*This review records findings only; no fixes are included.*
