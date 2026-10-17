# Pre-registration: trained-depth selection and the fresh test (2026-10-17)

Notebook: `tutorials/nucleotide_transformer_colab.ipynb`. Review: tutorial-triage row 58, findings T58-1..T58-8.
This file is committed on its own, before any grid run or any fresh-test score exists. Nothing below was chosen
after looking at a validation-grid result or a fresh-test number.

## Why

The default configuration (a mean-pooled head plus the last two encoder blocks) was chosen in the build
(revision `26879e0`) on comparisons whose split was not recorded. Their accuracies fit a balanced 400-window split,
which is the size of the notebook's test split, so the 400 test windows of the seed-42 draw are treated as
**exposed**: they stay in the notebook as the labelled **exposed development test**. They also overlap training
windows on the genome (113 of 400 overlap a train or validation window; 102 of them are negatives).
The trained depth is re-selected on validation only, and a new test is drawn that has never been scored.

## Data (fixed before the grid)

- **Sample:** the seed-42 draw of `tools/pin_sample.py` from the Genomic Benchmarks `human_nontata_promoters`
  interval lists at `605d8539830e16c85abe7826990958303ffc5e1c` (the four files pinned by size and SHA-256 in
  `samples.CORPUS_FILES`), with bases read from GRCh38 through Ensembl REST and checked against the authors' Hub
  re-upload.
- **Validation re-draw (T58-6, owner ruling A):** the 51 validation windows that overlap a training window on the
  genome (all 51 are negatives) are replaced by windows of the same label from the unused origin-train
  intervals. Each replacement is at least 251 bases (`LOCUS_MARGIN`) from every training window and every other
  validation window, drawn with `VALIDATION_REDRAW_SEED = 20261017` (sorted by id, shuffled, taken greedily).
  Train and the exposed test are unchanged.
  - Train digest (unchanged): `3e4fe9685bab8554df451703ee65b288365f44d7a351ee9acc0a82f220c51810`
  - Validation digest: `ccb6ac98…` → `36da9e179a257c2c8a946f98ecf9de154d015007cf493a155e58f1320ef2a6c9`
  - Exposed-test digest (unchanged): `8384c09b5cba075000fe484f268adb6568c35acd602da8d47a31de36501fc747`
  - `SAMPLE_DIGEST`: `d913c4bd…` → `96c002371dc813bc8b20edc7aa1d82103e99b69399946a6d963c2b8f4852c62b`
  - Asserted: validation–train genome overlap 0 windows.
- **Fresh test:** 400 windows, 200 promoter and 200 non-promoter, from the 8,634 origin-test intervals that are
  not in the sample. Each one is at least 251 bases from every train, validation and exposed-test window and from
  every other fresh-test window. Drawn with `FRESH_TEST_SEED = 20261018` (positives first, then negatives; each
  pool sorted by id, shuffled, taken greedily). The bases are fetched and oracle-checked like the sample.
  - `FRESH_TEST_DIGEST`: `338824e22a7d537a5a405e4f0b4757d9a5d92e2fc3f20b61fc14155c8a944c5d` (400 windows, 24 chromosomes)
  - Asserted: genome overlap with train, validation and the exposed test 0 at a 251-base margin; no fresh-test
    sequence equals, or is the reverse complement of, any sample sequence; oracle 2,600 of 2,600 matched.
- **Pre-training overlap (DAT9):** every window, including the fresh test, comes from GRCh38. The model's 850-genome
  pre-training collection (`InstaDeepAI/multi_species_genomes`, `urls.txt` entry 784) lists that assembly, so
  the test sequences may have been seen, unlabelled, during masked-LM pre-training. This cannot be avoided with
  human-reference windows and is stated, not removed.

## Fixed settings

Everything the run uses is fixed (`protocol.PREREGISTERED_SETTINGS`):

| setting | value |
|---|---|
| data source | the pinned promoter sample above (`SAMPLE_DIGEST`, split digests, 1,600 / 200 / 400), sample seed 42, `SPLIT_SEED` 42 |
| fresh test | `FRESH_TEST_DIGEST`, 400 windows |
| head | mean-pooled, 512 → 512 `tanh` → 2 (kept as is; its choice is disclosed as made on an unrecorded split) |
| epochs | 6, epoch kept by the highest validation MCC (ties: accuracy, then the earlier epoch) |
| learning rate | 3e-5, AdamW, weight decay 0, gradient clipping 1.0, no scheduler |
| batch size | 16 |
| seed | 0 |
| selection grid | on (`RUN_SELECTION_GRID = True`), `TRAINED_LAYERS = -1` (use the grid's selection) |

## The grid and the selection rule

- **Configurations:** `TRAINED_LAYERS` ∈ {0 (head only on the frozen encoder), 2 (current default), 12 (the whole
  encoder)}. Everything else is fixed as above. Configurations are trained in the order 0, 12, 2, each from the
  pinned base.
- **Score:** each configuration's best-epoch validation MCC on the 200 re-drawn validation windows.
- **Rule** (`protocol.select_trained_layers`):
  1. Keep the frozen encoder (0 blocks, head only) unless another configuration beats its validation MCC by
     **more than 0.03**.
  2. Otherwise keep 2 blocks unless the whole encoder beats it by **at least 0.03** validation MCC.
  3. When configurations are within the tolerance of one another, the smaller trainable scope wins.
- The 0.03 is a practical tolerance, roughly the noise of a 200-window validation MCC. It is not a significance
  threshold.
- **Cost:** about 2–3 min on a T4 (depth 2 measured 31.8 s on the 2026-10-08 hosted run). The grid is on by default.

## Scoring rules (SCORE_ONCE_RULE)

- The fresh test is scored **only** when every setting above holds (`protocol.preregistration_check` returns
  `preregistered: True`). That covers the majority and GC baselines, the frozen probe and the adapted model.
- Any other run is **exploratory**: changing a form field (epochs, learning rate, depth, seed, split seed, grid
  off), BYOD, the optional experiments, or the troubleshooting tips. An exploratory run is compared on validation
  and the labelled exposed development test only. It writes
  `outputs/nucleotide_transformer_evaluation_report_exploratory_<n>.json` and never the pre-registered report.
- Every report carries `test_look` ("pre-registered evaluation" | "reproduction" | "not scored (exploratory)")
  and `settings_differing`.
- The first recorded pre-registered **hosted** run is the result. A later run with identical settings is a
  reproduction of it, provided nothing was changed in response to the test number.
- A CPU build record of this revision is a build record pending that hosted run. It is reported as such.
