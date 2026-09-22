---
license: cc-by-nc-sa-4.0
model_card_spec: "1.1"
pipeline_tag: fill-mask
task: "Genomics - DNA Sequence Representation & Promoter Classification Fine-Tuning"
base_model: InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
date_published: "2023-07-27"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species); the repository hosts this checkpoint only."
---

# Nucleotide Transformer v2 50M Multi-Species — DNA Sequence Representation and Promoter Fine-Tuning Pipeline

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-InstaDeepAI%2Fnucleotide--transformer--v2--50m--multi--species-ffcc4d?style=flat)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-instadeepai%2Fnucleotide--transformer-181717?style=flat&logo=github&logoColor=white)](https://github.com/instadeepai/nucleotide-transformer)
[![Nature Methods](https://img.shields.io/badge/Nature%20Methods-10.1038%2Fs41592--024--02523--z-b31b1b.svg)](https://doi.org/10.1038/s41592-024-02523-z)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> [!IMPORTANT]
> **Non-Commercial Model License**
>
> The Nucleotide Transformer v2 50M Multi-Species model (`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`) is developed by InstaDeep, NVIDIA and TUM and distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
>
> The upstream license permits copying, redistribution and adaptation of the model for non-commercial purposes, subject to its attribution and ShareAlike requirements. **Commercial use is not permitted under this license.** Every embedding, probe and adapter this repository produces is derived from the weights and carries the same conditions; the adapter manifest records them.

> [!WARNING]
> Deployment of this model through DIMER is intended only for uses consistent with the upstream CC BY-NC-SA 4.0 license. The licence was **accepted for this profile on 2026-09-19** with its non-commercial, attribution and ShareAlike obligations, and the checkpoint's remote-code requirement was **accepted on 2026-09-20** inside a digest-verified perimeter — see *DIMER deployment notes* below. Users are responsible for ensuring that their intended use is non-commercial and otherwise complies with the license terms.
>
> If the model or model weights are modified and the resulting adaptation is distributed, the adaptation must be distributed under the same license or a compatible license as required by CC BY-NC-SA 4.0. The adapter written by `save_artifact` is such an adaptation.
>
> The model weights are redistributed unmodified from the upstream model repository — and this repository redistributes none of them at all: it records their digests and stages them from the pinned upstream revision at runtime. DIMER does not grant any additional rights to the model beyond those provided by the original licensor. Hosting the model on DIMER does not imply affiliation with or endorsement by InstaDeep, NVIDIA, TUM or Hugging Face.
>
> - Upstream model: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
> - Model license: CC BY-NC-SA 4.0
> - Original developers: InstaDeep, NVIDIA and TUM

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebook downloads third-party weights governed by their own license and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any use of this material.

---

## Interactive Colab Tutorials

This repository provides a ready-to-run interactive Google Colab notebook that exercises the pipeline end to end — stage and digest-verify the pinned snapshot including the Python it will execute, read the inline promoter sample and split it without leakage, embed sequences and predict a masked 6-mer through the inference contract, score two non-neural baselines and a logistic probe on the frozen embeddings, fine-tune a mean-pooled head plus the last two encoder blocks, evaluate on the held-out windows, and export and reload the adapter:

- **E2E promoter fine-tuning tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/tutorials/nucleotide_transformer_colab.ipynb) [`nucleotide_transformer_colab.ipynb`](https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/tutorials/nucleotide_transformer_colab.ipynb)  
  *Standalone `E2E` tutorial (DIMER Notebook Specification 2.0 §4) on pinned, digest-verified weights: the carried package, a 2,200-window human non-TATA promoter sample carried inline with its provenance and digest, 512-dimensional mean-pooled sequence embeddings and one masked-token prediction, majority and GC-threshold baselines, a logistic probe on the frozen embeddings, bounded fine-tuning with validation-based epoch selection, held-out accuracy and MCC, and a safetensors adapter that reloads with verified parity and carries the base licence. The notebook states the non-commercial licence before it downloads anything and verifies the remote code before it executes any.*

---

#### Description

`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` is the 50M-parameter multi-species member of the Nucleotide Transformer family (Dalla-Torre et al., Nature Methods 2024), pinned here to revision `81b29e5786726d891dbf929404ef20adca5b36f1`. It is a transformer encoder trained with a masked-language-model objective on DNA: the upstream model card records pretraining on a collection of 850 genomes spanning model and non-model organisms, at a sequence length of 1,000 tokens. DNA is tokenized as **6-mers** — six consecutive `A`/`C`/`G`/`T` bases become one token, and any window containing another character (such as `N`) falls back to one token per base — over a 4,107-entry vocabulary; the encoder is 12 blocks of width 512 with rotary positions and a bias-free SwiGLU feed-forward, 53,534,401 parameters, plus a masked-LM head that brings the loaded model to 55,904,972.

This repository is the DIMER pipeline for that checkpoint. The package `nucleotide_transformer_genomics_pipeline` exposes two capabilities behind one verified perimeter. **Representation** (`embed`, `token_counts`, `predict_masked`): one mean-pooled 512-d vector per sequence and the pre-training objective's distribution at a masked position. **Adaptation** (`linear_probe`, `adapt`, `predict`, `evaluate`, `save_artifact`, `from_artifact`): a logistic probe on the frozen embeddings as the reference, and a bounded fine-tuning of a fresh mean-pooled classification head together with the last two encoder blocks — 8,660,482 of 53.8 M parameters — for binary promoter classification, exported as a 35 MB safetensors adapter that reloads against the pinned base. The checkpoint cannot be loaded by the native Transformers ESM classes, so it ships its own model code; `from_pretrained` refuses to run without the snapshot manifest, stages any absent file at the pinned revision, re-hashes every file — including the two Python files the loader executes — and only then imports them. The tokenizer is the native `EsmTokenizer`, loaded without remote code.

#### Intended Use and Limitations

###### Primary Intended Uses

The tasks are DNA sequence representation and binary sequence classification by fine-tuning. Input is one or more DNA sequences as strings over `A`, `C`, `G`, `T` and `N` (case-insensitive; 12..6,000 bases); output is either a 512-dimensional mean-pooled vector per sequence, a logit distribution over the 4,107-token vocabulary at a masked position, or — after `adapt` or `load_artifact` — a label in `{0, 1}` with two class probabilities per sequence. Envisioned application domains are genomic sequence analysis where a learned representation is wanted instead of hand-built k-mer features: extracting embeddings to feed a downstream classifier, transfer learning toward a bounded labelled sequence-classification task of the kind the upstream paper evaluates (the demonstrated one is human non-TATA promoter recognition on 251-base windows), similarity search or clustering over a sequence set, and research and educational use — including non-commercial experimentation with and evaluation of genomic language models. In a larger system the base model is a feature extractor and the adapter is a task-specific classifier whose labelling convention is exactly that of its training set.

Against that, this repository provides: the representation contract, the frozen probe, the adaptation contract with a pinned real dataset and a build record, and the artifact round trip. It does not validate any downstream biological application beyond the sample-sanity evidence recorded below.

###### Primary Intended Users

Intended users are computational biologists, bioinformaticians and machine-learning engineers evaluating or adapting genomic language models, in research and educational settings — and only for uses that satisfy the upstream licence's NonCommercial condition. The pipeline assumes its users understand that a pretrained representation is not a validated predictor of any biological property; that a fine-tuned head learns its training set's window length, negative-sampling rule and label definition and nothing else; that the pinned vocabulary is a 6-mer vocabulary, so a single ambiguity code or a length that is not a multiple of six shifts every downstream token boundary; that sequences from the same gene family, locus or assembly are not independent, so a random split leaks and inflates every metric; that neither masked-token logits nor the adapter's class probabilities are calibrated; and that the licence question — is my use non-commercial? — is theirs to answer before they download the weights, not a checkbox the notebook ticks for them.

###### Out-of-scope use cases

1. **Commercial use**, unless separately authorized by the relevant rights holder(s). The upstream licence's NonCommercial condition governs; nothing in this repository or in DIMER grants rights beyond it. Uses inconsistent with CC BY-NC-SA 4.0 — including distributing an adapter under an incompatible licence, or redistributing without the required attribution — are out of scope regardless of technical merit.
2. **Capability boundary:** the base checkpoint produces representations and masked-token logits; the adapter produces a binary label under one training set's convention. Neither predicts enhancers, splice sites, chromatin state, variant effects, any multi-class or token-level task, or any other molecular phenotype; those require a task-specific head trained and validated on labelled data for that task. This repository implements binary sequence classification only — no multi-class heads, no LoRA, no fine-tuning of the embeddings or the first ten blocks, no variant scoring and no published-benchmark reproduction — and it does not package the larger Nucleotide Transformer checkpoints.
3. **Input boundary:** sequences must be over `A`/`C`/`G`/`T`/`N` (upper-cased on entry); 12..6,000 bases per sequence — 6,000 bases is 1,000 tokens, the length the upstream card says the model was trained at — and at most 20,000 sequences per call. The tokenizer's `model_max_length` is 2,048 tokens and the position embeddings allow 2,050, so longer inputs would run but leave the trained regime; the pipeline refuses them. Alignment gaps, amino-acid sequence, RNA with `U`, and non-human-readable formats are rejected rather than coerced.
4. **Decision boundary:** model output must not be treated as validated clinical or diagnostic advice, and must not drive patient management, therapeutic selection, genetic counselling, screening or any regulated decision, without task-specific validation, human expert review and whatever regulatory clearance the jurisdiction requires. Unsupported biological interpretation — reading a high embedding similarity, a confident masked-token prediction or an adapter probability as evidence of function, causality or pathogenicity — is out of scope for this model and this repository.

#### Factors

###### Groups

The model is not human-centric in the demographic sense: its inputs are DNA strings with no attached identity, and no demographic attribute is available to it or evaluated here. Two things keep the question from being empty, though. First, the upstream family's larger models were trained partly on 3,200 diverse human genomes and this multi-species checkpoint on 850 genomes across species; neither the upstream authors nor this repository publishes a breakdown by ancestry, population or clade, so representation quality across human ancestries and across the tree of life is unknown rather than balanced, and genomic reference data is historically skewed toward European-ancestry individuals and toward well-studied model organisms. The demonstrated fine-tuning task is drawn from one human reference assembly (GRCh38), which is itself a mosaic that represents no population. Second, human genomic sequence is inherently personal data: an operator who applies this model to human samples is processing information about people even though the model sees no identity — see *Ethical considerations and biases*.

###### Instrumentation

The instrument is DNA sequencing followed by assembly and annotation. The pretraining data are whole genomes as deposited in public references; the characteristics that matter are the sequencing platform and depth, the assembly quality (contig gaps appear as `N` runs), the reference build, and whether repeats were soft-masked — the pipeline upper-cases input, so soft-masking is erased rather than rejected, and the operator's masking convention no longer changes the tokens but does change what a lowercase run meant. For the demonstrated task the instrument is also the annotation: promoter windows come from the Eukaryotic Promoter Database's transcription-start-site annotations as windowed by Genomic Benchmarks (251 bases, `-` strand reverse-complemented), and negatives are windows of the same reference that contain no annotated promoter — an absence of annotation, not a verified absence of function. Assembly and basecalling errors enter as substituted or missing bases and shift 6-mer boundaries downstream of the error. The pipeline passes the caller's sequence upper-cased but otherwise verbatim: no gap removal, no reverse-complement handling, no chunking of long sequences, no quality filtering. Defects upstream of the pipeline therefore reach the model unchanged, and a BYOD CSV's labels are trusted exactly as given.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3` (`pyproject.toml`); the pipeline loads to `cuda:0` when CUDA is visible and to CPU otherwise. Only float32 is exercised — no quantisation, autocast or compiled kernels. The build record of 2026-09-20 on the pinned 1,600 / 200 / 400 sample: on an RTX 5070 Ti, model load 7.9 s, frozen probe 3.1 s, six-epoch fine-tuning with per-epoch validation 33.9 s at a 558 MiB peak; on the build workstation's CPU, load 4.5 s, probe 19.4 s, fine-tuning 212.9 s — with identical metrics to four decimals on both devices. Every sample sequence is 251 bases = 47 tokens; memory and time scale with tokens per sequence times batch size, because each batch is padded to its longest member, and the 6,000-base ceiling was not exercised. **Loading requires executing the checkpoint's own Python** (`modeling_esm.py`, `esm_config.py`) in the user's process; the pipeline digest-verifies both files against the manifest before importing them and refuses to load without a manifest. The rotary-embedding tables in that code are cached on first use, so all inference paths run under `torch.no_grad()` rather than `torch.inference_mode()`, which would poison a later backward pass.

#### Metrics

###### Performance Measures

This repository measures binary classification on a real labelled task and reports it against three references on the same 400 held-out windows. **Matthews correlation (MCC)** is the headline metric — 0 for any constant or chance predictor and symmetric in the two classes, so a balanced test split cannot be gamed by predicting one label — reported with **accuracy**, per-class precision, recall and F1, and the confusion counts. The references are a **majority baseline** (0.5 accuracy, MCC 0 on the balanced split), a **GC-threshold baseline** fitted on the training split alone (the midpoint threshold and direction that maximise training accuracy; promoters are GC-rich, so this is the composition-only floor a sequence model must beat), and the **frozen probe** — an L2-penalised logistic regression on the standardised mean-pooled embeddings, fitted on the training split with the model's weights untouched, which is what the pre-trained representation alone knows about the task. `evaluation_report` turns the four numbers into a structured verdict (adapted beats frozen; adapted beats both baselines; MCC gain; small-sample flag below 50 records). Build record (seed 0): GC ≥ 0.560 → 0.715 / 0.431; frozen probe 0.815 / 0.632; **adapted 0.8425 / 0.693** (promoter class precision 0.905, recall 0.765; negatives precision 0.797, recall 0.92). The representation contract is additionally exercised, not measured: a composition-matched synthetic pair (the same bases, one order carrying a repeated 6-mer) is embedded and its cosine distance reported as a sample-sanity observation.

###### Decision thresholds

The adapter applies one decision rule: **argmax over the two class logits**, equivalent to a 0.5 threshold on the promoter probability, and reports both probabilities so a caller can move it. Nothing in the pipeline claims these probabilities are calibrated — the per-class rows of the build record (promoter recall 0.765 against negative recall 0.92) show the default rule sits on the conservative side for the promoter class, and a deployment that needs a different operating point owns fitting it on its own labelled data. The GC-content baseline does fit a threshold — the midpoint that maximises accuracy on the training split only — and that value is reported with its training accuracy precisely so a reader can see it was fitted, not chosen to flatter the model. The frozen probe thresholds its logit at 0. Epoch selection uses one rule, stated in the adapter: the epoch with the highest validation MCC (ties: accuracy, then the earlier epoch). Thresholds deliberately not shipped: no acceptance threshold on MCC, no embedding-similarity cutoff for "same class", no logit cutoff for "confident prediction".

###### Approaches to uncertainty and variability

All numbers come from **one seeded draw** (seed 42) of the origin's interval lists — 1,600 training, 200 validation and 400 test windows — and one training seed (0). There is no cross-validation, no bootstrap and no repeated-seed averaging, and therefore **no dispersion estimate is reported**; the 0.06 MCC gain over the frozen probe should be read against the run-to-run spread the build record observed for the same recipe on one GPU (a few hundredths of MCC on the validation curve) and not as a margin with a confidence interval. Sources of variability: the draw, the head initialisation and the batch order are seeded, so the default path is reproducible on a fixed runtime — the GPU and CPU build records agree to four decimals; non-deterministic GPU kernel selection and float accumulation order are **not** seed-controlled and can move a probability in the sixth decimal and, near the decision boundary, a label. The validation split that picks the epoch is 200 windows, so its MCC is noisy to about ±0.03 and the selected epoch can differ between runs; the build record kept epoch 2 of 6 while later epochs overfit (training loss 0.51 → 0.14). A caller who needs an interval repeats the workflow across seeds, or supplies enough labelled sequence for k-fold with locus-aware folds, and owns that design.

#### Ethical considerations and biases

###### Data

Upstream discloses the pretraining data for this checkpoint as a collection of 850 genomes from a wide range of species, including model and non-model organisms, and lists `InstaDeepAI/multi_species_genome` and `InstaDeepAI/nucleotide_transformer_downstream_tasks` as associated datasets; the upstream card does not publish a per-species, per-assembly or per-population breakdown, and that is where disclosure ends. Whether human genomic sequence is included in the multi-species mix, and from which cohorts, is not established by the upstream card either way — unknown, not ruled out. The demonstrated fine-tuning data are 2,200 windows of the GRCh38 human reference genome selected by the Genomic Benchmarks `human_nontata_promoters` interval lists (Apache-2.0; commit `605d8539830e16c85abe7826990958303ffc5e1c`), read from Ensembl and cross-checked window for window against the benchmark authors' own re-upload; the reference is public domain and represents no individual. Human genomic sequence is in general personal data and in principle re-identifying: a model trained on it stores statistical regularities rather than records, but an operator who *applies* it to human samples is processing personal and potentially health data. This repository distributes documentation, code, interval-derived reference windows and digests — no weights, no individual's sequence.

###### Human Life

This model is not intended, in this repository, for decisions in health, safety, criminal justice, employment, credit or housing, and it has not been validated or certified for any of them by anyone. No clinical validation, no regulatory clearance, no external review board and no testing with any specific group has taken place here; the only evidence is the build record on one benchmark draw. Use in a sensitive domain is nevertheless foreseeable — genomic language models are applied to variant interpretation, pathogen characterisation and clinical research pipelines — and would be admissible only with a task-specific model validated on data representative of the deployment, a domain expert reviewing every consequential output, locus- and cohort-aware evaluation, and the applicable clinical-laboratory and regulatory clearance. An embedding, a masked-token logit or an adapter probability from this model is an input to such a process at most, never its output.

###### Mitigations

1. **Supply-chain integrity:** the model id and the 40-hex revision are constants in the package and in this card; the snapshot manifest records path, byte size and SHA-256 for all 8 files (223,752,120 bytes total); `stage_missing_files` fetches only absent entries from the Hub **at that revision** (never `main`) and `verify_snapshot` re-hashes every file before anything is loaded. The `model.safetensors` digest equals the `oid sha256` of the Hub LFS pointer at the pinned revision. The upstream `pytorch_model.bin` and JAX checkpoint are not in the manifest and are never staged.
2. **The remote-code files are inside the verified perimeter:** `modeling_esm.py` and `esm_config.py` are manifest entries, `verify_snapshot` refuses a manifest that does not list them, and `from_pretrained` has no manifest-less path — so the Python that `trust_remote_code=True` imports is digest-verified before it is imported and an upstream change fails loudly instead of executing. The tokenizer loads natively. `tools/validate_release_assets.py` fails the build if `trust_remote_code=True` appears anywhere in the notebook outside the carried module, more than once inside it, or before `verify_snapshot` in `from_pretrained`. This is a narrowing of the trust boundary, not a removal of it — see *DIMER deployment notes*.
3. **No vendoring:** the repository ships neither weights nor the upstream Python (`weights/**/*.safetensors` and `weights/**/*.py` are git-ignored), so nothing executable is redistributed here and the licence question stays with the upstream files.
4. **Input integrity:** `validate_sequences` / `validate_dataset` reject non-string input, sequences shorter than 12 or longer than 6,000 bases, characters outside `A`/`C`/`G`/`T`/`N` (naming them), labels outside `{0, 1}`, duplicate ids, datasets outside 8..20,000 records and training splits with one label, all before any model import; `validate_inputs` returns an input manifest with per-sequence digests.
5. **Evidence integrity:** the sample is digest-pinned and refused if edited; the draw keeps the origin's train/test separation; `check_split_disjoint` asserts no sequence is in two splits; both baselines and the probe are fitted on the training split only; epoch selection uses the validation split only; the test split is scored once per system; and `evaluation_report` states the comparison and the small-sample flag rather than a bare number.
6. **Artifact integrity:** `save_artifact` records the artifact format, the base identity and weight digest, the remote-code files, the inherited licence, the tensor names, the file digest and the training history; `load_artifact` refuses a manifest with the wrong format, base, licence, digest, size or tensor set before deserialising, and a fresh pipeline is used for the reload so parity is measured against files, not memory.
7. **Transactional adaptation:** on any exception during `adapt` the frozen encoder weights are restored and the previous adapter, if any, is kept.

Not implemented, and therefore not claimed: no security review of the upstream Python beyond reading what it does architecturally, no sandboxing of its execution, no differential-privacy or memorisation analysis, no calibration of the class probabilities, and no fairness audit of any kind.

###### Risks and harms

1. **Executing third-party code:** the load path runs upstream Python in the user's runtime. Pinning and digest verification mean the code cannot change without detection, but they do not make it safe; a reader who skips Section 3 of the notebook may not realise this happened. The operator bears the risk; DIMER accepted it for this profile on 2026-09-20 with that understanding.
2. **Licence breach through inattention:** the weights are non-commercial, and the most likely harm from this repository is somebody embedding them — or the adapter — in a commercial product because a notebook ran without complaint. The card, the README, the notebook and the adapter manifest all state the condition; nothing enforces it, and nothing can.
3. **Composition confounding:** GC content separates promoter from non-promoter windows well enough to score 0.715 alone, and repeat structure separates many other naive genomic datasets; a fine-tuned classifier can score well while learning little beyond composition. The GC baseline is in the default path so the gap is visible; a user who omits it on their own data will not see it.
4. **Convention lock-in:** the adapter learns 251-base windows centred on EPD non-TATA promoters against annotation-free windows. Applied to windows of another length, another centring rule or another negative-sampling rule, its numbers do not transfer, and nothing in the pipeline detects the shift.
5. **Locus and homology leakage:** related sequences on both sides of a split turn memorisation into apparent generalisation; the likelihood is high for any dataset assembled by a database query. The pinned sample inherits the origin's train/test separation; a BYOD CSV is de-duplicated by exact sequence only, which catches copies, not paralogues.
6. **Tokenization surprises:** one `N` or a length that is not a multiple of six changes the token sequence — a single ambiguity code costs six bases of context and shifts every boundary after it. A user who filters sequences by base count rather than token count can silently compare sequences the model saw very differently.
7. **Over-reading a representation or a probability:** cosine similarity between two embeddings is not homology, function or pathogenicity, and an adapter probability of 0.9 is not a 90 % chance of a promoter. The magnitude of the harm ranges from a wasted experiment to a mistaken claim in a manuscript.

###### Use cases

The model must not be used to infer, from a person's DNA, health status, disease risk, ancestry or any other personal characteristic used to make decisions about them — in employment, insurance, credit, education, immigration or healthcare access — nor to re-identify individuals from genomic data, nor for genetic surveillance or population profiling. It must not be used to produce clinical or diagnostic outputs presented to patients or clinicians as validated, nor to generate model-predicted annotations deposited into public sequence databases without marking them as predictions. It must not be used to design, screen or optimise sequences for toxins, pathogens or any agent intended to cause harm, nor to circumvent biosecurity screening of synthesised sequences. It must not be used to process sequence the operator has no right to process — patient material outside the scope of its consent, sequence under a data-use agreement that excludes model inference, or a third party's proprietary data — and it must not be used commercially, which is the upstream licence's condition and not this repository's to waive.

## Immutable provenance

- Model: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
- Revision: `81b29e5786726d891dbf929404ef20adca5b36f1`
- Manifest: `weights/nt-v2-50m-multi-species/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, 8 files, `totalBytes` 223752120
- `model.safetensors` (223,642,688 bytes) SHA-256: `17e75af297556ea56828716d8aa539e8f12b8b625547204b74171ab91aa33569` (equal to the `oid sha256` of the Hub LFS pointer at the pinned revision)
- `modeling_esm.py` (58,205 bytes) SHA-256: `f2b003f45d2fa4f94e92d8bbef927b719dfdd7ab7c7a95743daa2b10ab140cb3` — **executed** by the loader
- `esm_config.py` (14,876 bytes) SHA-256: `a44e859baa08465ecdcd76b0a73f6e4fe011245a212931c403de1149ae9613ec` — **executed** by the loader
- `config.json` (1,064 bytes) SHA-256: `e20f497248c7cb264c7cd4582dbcfd52dc4cbf74a97fc711559b8c8f71c635db`
- `vocab.txt` (28,718 bytes) SHA-256: `c00e0ad166d6ab3f7540ebc92270392e581bb3106763412f29034b49323e1052`
- `tokenizer_config.json` (129 bytes), `special_tokens_map.json` (101 bytes), upstream `README.md` (6,339 bytes) — digests in the manifest
- Nothing in this list is vendored in Git: the weights and the two Python files are staged from the pinned revision and verified against these digests before use.
- Fine-tuning sample: Genomic Benchmarks `human_nontata_promoters` interval lists from `ML-Bioinfo-CEITEC/genomic_benchmarks` at commit `605d8539830e16c85abe7826990958303ffc5e1c` (four gzipped CSVs pinned by size and SHA-256 in `samples.py`), bases from the GRCh38 reference via Ensembl REST; 2,200 windows carried inline, digest `d913c4bd4cf74199456d24ced1f3ddc8f39e9241ab2eca132ffb9885efc62cfb`; reproduced by `tools/pin_sample.py`.
- Upstream reference: https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species

## Model architecture

Read from the pinned `config.json` and the upstream model card, not inferred from the model's name:

| Property | Value | Source |
|---|---|---|
| Family | Nucleotide Transformer v2, multi-species | upstream model card |
| Objective | masked language modelling over DNA tokens | upstream model card and paper |
| Layers / hidden size / heads / intermediate | 12 / 512 / 16 / 2,048 | pinned `config.json` |
| Position embeddings | rotary | pinned `config.json` (`position_embedding_type`) |
| Feed-forward block | bias-free **SwiGLU** (project to 2× intermediate, split, gate with SiLU) | pinned `modeling_esm.py` |
| Vocabulary | 4,107 tokens: 6 special, 4,096 6-mers, 5 single bases (`A`,`T`,`C`,`G`,`N`) | pinned `vocab.txt`, `config.json` |
| Parameters | 55,904,972 as loaded (`EsmForMaskedLM`); encoder 53,534,401; the file stores 55,905,164 elements (+ 192 rotary buffer elements) | measured from the pinned checkpoint |
| Adapter | mean-pooled head (512→512 `tanh`, 512→2; 263,682 parameters) + encoder blocks 10 and 11 (8,396,800) = 8,660,482 trainable, 32 tensors, 34,645,456 bytes | `pipeline.py`, build record |
| Trained sequence length | 1,000 tokens (≈ 6,000 bases) | upstream model card |
| Architectural maximum | `model_max_length` 2,048 tokens; `max_position_embeddings` 2,050 | pinned tokenizer/model config |
| Pretraining data | 850 genomes, model and non-model organisms | upstream model card |

Two upstream-documentation discrepancies, recorded rather than smoothed over: the upstream card states a vocabulary size of **4,105**, while the pinned `vocab.txt` and `config.json` both give **4,107**; and it states a maximum tokenized length of **1,000**, while the pinned tokenizer config allows **2,048**. This card uses the pinned files for the vocabulary and the upstream card's trained length as the operating ceiling, which is the conservative reading of the pair.

## Inputs

- **Format:** DNA as a string over `A`, `C`, `G`, `T` and `N`; input is upper-cased, so soft-masked lowercase is accepted and its masking information discarded. Alignment gaps (`-`, `.`), RNA `U`, amino-acid letters and whitespace inside the sequence are rejected by validation, which names the offending characters; the upstream model itself has no such guard and would map unknown characters to `<unk>`.
- **Tokenization:** the native Transformers `EsmTokenizer` (loaded **without** remote code) reading the checkpoint's `vocab.txt`. Sequences are consumed left to right in windows of six: a clean window of `A`/`C`/`G`/`T` becomes one 6-mer token; a window containing anything else, and any trailing remainder shorter than six, falls back to one token per base. A `<cls>` token is prepended; a 251-base window is 47 tokens.
- **Length:** the upstream trained length is 1,000 tokens ≈ 6,000 bases, which this repository adopts as its ceiling (`MAX_BASES = 6000`); the architecture allows 2,048 tokens. Sequences are not chunked automatically — a caller with longer sequence owns the windowing and the aggregation.
- **Labelled records:** `{id, sequence, label}` with `label` in `{0, 1}`, ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; 8..20,000 records per dataset; a training split must contain both labels. BYOD is one CSV with a header naming `sequence` and `label` (optionally `id`).
- **Batching:** batches are padded to the longest member, so mixing very short and very long sequences wastes compute; the pipeline caps a call at 20,000 sequences and pools over non-padding tokens only.
- Repository-specific, not upstream: the 12-base minimum, the 6,000-base ceiling, the sequence cap, the alphabet rejection, the id rules and the dataset bounds are this repository's input contract. The upstream model imposes only the position-embedding limit.

## Outputs

- **Sequence representations** (`embed`): one 512-dimensional mean-pooled vector per sequence over non-padding tokens. These are representations; they carry no label, no class and no calibrated score, and any quality claim about them requires a downstream labelled task.
- **Masked-token prediction** (`predict_masked`): the top-k tokens and softmax probabilities at one masked position from the masked-language-model head — rankable, not calibrated, and not a downstream phenotype prediction.
- **Classification** (`predict`, after `adapt` or `load_artifact`): a label in `{0, 1}` (`no_promoter`, `promoter` for the demonstrated task) and two class probabilities per sequence; `evaluate` scores labelled records with accuracy, MCC, per-class precision/recall/F1 and the confusion counts.
- **Adapter artifact** (`save_artifact`): `adapter.safetensors` holding the head and the trained encoder tensors, plus `manifest.json` recording the format `org.valcorza.nt-v2-50m-multi-species.adapter.v1`, the inherited licence, the base identity and weight digest, the remote-code files, the tensor names, the file digest, the training configuration and the epoch history.
- **What the base model does not output:** class labels without an adapter, variant effect scores, per-base functional annotations, or any of the downstream genomics benchmark predictions reported in the upstream paper beyond the one binary task an adapter was trained for.

## Example usage

The tutorial notebook is the supported path; this is the same workflow condensed through the package API. Install with `pip install -e .` inside the pinned runtime (`pyproject.toml`).

```python
from nucleotide_transformer_genomics_pipeline import (
    NucleotideTransformerPipeline, gc_threshold_baseline, majority_baseline, sample_dataset,
)

splits = sample_dataset()                       # the pinned inline draw: train 1,600 / validation 200 / test 400
pipe = NucleotideTransformerPipeline.from_pretrained(allow_download=True)   # stage -> verify (incl. the model code) -> load

vectors = pipe.embed([r["sequence"] for r in splits["test"][:4]])           # 4 x 512, mean-pooled; a representation
masked = pipe.predict_masked("ATTCCG" * 3 + "<mask>" + "ATTCCG" * 3)       # the pre-training objective at one position

baseline = gc_threshold_baseline(splits["train"], splits["test"])           # composition-only floor, fitted on train
frozen = pipe.linear_probe(splits["train"], splits["test"])                 # what the frozen representation knows
result = pipe.adapt(splits["train"], splits["validation"])                  # head + last 2 blocks, 6 epochs, lr 3e-5
adapted = pipe.evaluate(splits["test"])                                     # accuracy / MCC on the held-out windows

pipe.save_artifact("outputs/adapter")                                       # safetensors + manifest (licence inherited)
reloaded = NucleotideTransformerPipeline.from_artifact("outputs/adapter")   # fresh base + verified adapter
print(baseline["mcc"], frozen["mcc"], adapted["mcc"], majority_baseline(splits["train"], splits["test"])["mcc"])
```

Build record (RTX 5070 Ti, seed 0): the four MCC values printed by the last line were `0.4308`, `0.6315`, `0.6934`, `0.0`; the reloaded pipeline reproduced every label of the first 64 test windows with a maximum probability difference of 0.0.

## Limitations

- A pretrained representation does not guarantee downstream predictive validity: the model was trained to predict masked DNA tokens, and nothing about that objective promises that its embeddings are informative for the phenotype you care about. The frozen probe measures exactly that for one task and is why the fine-tuning number is read against it.
- The adapter learns one labelling convention — 251-base windows centred on EPD non-TATA promoters against windows with no annotated promoter — and its numbers do not transfer to other window lengths, centring rules, negative definitions, TATA promoters or other species without new labelled data.
- Performance depends on how close your sequence is to the 850-genome pretraining mix. Species, clade, GC regime and repeat content all shift that distance, and the model reports no out-of-distribution signal.
- Genomic context and preprocessing change the input the model sees: 6-mer boundaries, `N` runs, reverse-complement orientation, and where a window was cut all alter the tokenization and therefore the representation and the prediction.
- Downstream tasks require task-specific validation with locus- or homology-aware splits. A random split of related sequences produces metrics that will not survive contact with novel sequence; BYOD de-duplicates exact sequences only.
- Pretrained model output is not causal biological evidence. Similarity in embedding space and adapter probabilities reflect sequence statistics the model learned; attributing function, regulation or pathogenicity to them requires experimental or orthogonal computational support.
- Clinical or medical use requires substantially more validation than general inference, including cohort-representative evaluation, expert review and regulatory clearance — none of which this repository provides.
- Repository-specific: the fine-tuning is bounded to a mean-pooled head plus the last two blocks; the build record is one seeded draw with no dispersion estimate; the 0.06 MCC gain over the frozen probe is a few times the observed run-to-run spread, not a large margin; the 6,000-base ceiling and datasets near the 20,000-record cap were not exercised.

## DIMER deployment notes

| Field | Status |
|---|---|
| **DIMER status** | **Release-grade** — the E2E carrier (blob `c1aa7361` at `3e37b6c`) executed 11/11 in a clean Kaggle Tesla T4 runtime on 2026-09-20 with the model code digest-verified before import; record in `docs/release-verification.md` |
| Licence status | **Accepted 2026-09-19** — CC BY-NC-SA 4.0 with its non-commercial, attribution and ShareAlike obligations |
| Commercial use | **Not permitted** under the upstream licence |
| Redistribution | Permitted subject to the attribution, NonCommercial and ShareAlike conditions; adapters inherit them |
| DIMER deployment | Non-commercial use only |
| Weights | Redistributed unmodified if hosted; this repository redistributes none and stages them from the pinned revision |
| Remote code | **Required and accepted 2026-09-20** inside the digest-verified perimeter described below |
| Upload format | `model.safetensors` (223,642,688 bytes), plus the tokenizer files and the two Python files the loader executes |

**Licence (accepted 2026-09-19).** The upstream weights are CC BY-NC-SA 4.0. Hosting them on DIMER was accepted with the understanding that every use the platform enables must satisfy the NonCommercial condition, that attribution to InstaDeep, NVIDIA and TUM travels with the weights and with every derivative, and that adapters produced by this pipeline are distributed under the same licence — which `save_artifact` records in the artifact manifest and `load_artifact` refuses to load without.

**Remote code (accepted 2026-09-20).** This checkpoint cannot be loaded by the native Transformers ESM classes: its `config.json` carries an `auto_map` instead of a `model_type`, and, materially, its feed-forward block is a bias-free SwiGLU that the native implementation does not have — so loading these weights into the native class would not reproduce this model (and `AutoModel.from_pretrained` silently falls through to that native class and fails on the SwiGLU shape; the pipeline loads through `AutoModelForMaskedLM`, which the `auto_map` maps). `trust_remote_code=True` is therefore an architectural requirement here rather than a shortcut around a loading error. This profile was accepted on the perimeter this repository draws: the two Python files are pinned, listed in the manifest, digest-verified before they are imported, and the pipeline has no manifest-less load path; the tokenizer stays native; the release validator fails the build if the remote-code flag appears anywhere in the notebook outside the carried module, more than once inside it, or before `verify_snapshot`. Digest verification proves the executed code is the pinned upstream code byte for byte. It is not a security review of that code, and this is the only DIMER pipeline where remote code runs.

## Attribution

- The model is developed by **InstaDeep, NVIDIA and TUM** (Technical University of Munich), as stated in the upstream model card ("Developed by: InstaDeep, NVIDIA and TUM").
- Original publication: Dalla-Torre, H., Gonzalez, L., Mendoza-Revilla, J., Lopez Carranza, N., Grzywaczewski, A. H., Oteri, F., Dallago, C., Trop, E., de Almeida, B. P., Sirelkhatim, H., Richard, G., Skwark, M., Beguir, K., Lopez, M., Pierrot, T. *Nucleotide Transformer: building and evaluating robust foundation models for human genomics*. Nature Methods 22, 287–297 (2024). https://doi.org/10.1038/s41592-024-02523-z (preprint: bioRxiv 2023.01.11.523679).
- Upstream Hugging Face model ID: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`.
- Upstream source repository: https://github.com/instadeepai/nucleotide-transformer (CC BY-NC-SA 4.0).
- The checkpoint's `modeling_esm.py` and `esm_config.py` carry Meta Platforms and Hugging Face copyright headers and derive from the Transformers ESM implementation; they are modified by InstaDeep for this architecture.
- Fine-tuning sample: Grešová, K., Martinek, V., Čechák, D., Šimeček, P., Alexiou, P. *Genomic benchmarks: a collection of datasets for genomic sequence classification*. BMC Genomic Data 24, 25 (2023). https://doi.org/10.1186/s12863-023-01123-8 — interval lists under Apache-2.0; promoters from the Eukaryotic Promoter Database; bases from the GRCh38 reference (Genome Reference Consortium) via Ensembl REST.

## Repository and code licensing

Three licences apply to different things, and conflating them is the mistake this section exists to prevent:

- **Model weights** (`model.safetensors`) and everything derived from them (embeddings, probes, adapters): **CC BY-NC-SA 4.0** — attribution, NonCommercial, ShareAlike. This is the licence that governs what you may do with the model.
- **The checkpoint's supporting Python** (`modeling_esm.py`, `esm_config.py`): carries Meta/Hugging Face copyright headers and derives from the Apache-2.0 Transformers ESM implementation. A permissive licence on this code **does not** remove the NonCommercial condition from the weights.
- **This repository's own content** (this card, the package, the notebook, the tooling, the interval-derived sample): Apache-2.0, per `LICENSE`. It grants you nothing with respect to the upstream weights.

The upstream licence is not an OSI-style open-source software licence: CC BY-NC-SA 4.0 carries a NonCommercial restriction and should not be described as open source.

## Runtime

- Pins (`pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`. Python 3.12.
- Build record 2026-09-20 (Windows venv, `torch 2.14.0+cu130`, RTX 5070 Ti, seed 0, pinned 1,600 / 200 / 400 sample): load 7.9 s; majority 0.5 / MCC 0.0; GC threshold 0.5598 (`GC >= threshold -> positive`) 0.715 / 0.4308; frozen probe 0.815 / 0.6315 (train accuracy 0.8975) in 3.1 s; `adapt` 33.9 s at a 558 MiB peak, validation accuracy / MCC per epoch 0.505 / 0.071 (untrained head), 0.795 / 0.609, **0.805 / 0.621 (kept)**, 0.795 / 0.606, 0.78 / 0.577, 0.765 / 0.533, 0.80 / 0.612, training loss 0.514 → 0.139; adapted test **0.8425 / 0.6934** (promoter precision 0.9053, recall 0.765, F1 0.8293; negative precision 0.7965, recall 0.92; tp 153, tn 184, fp 16, fn 47); artifact 34,645,456 bytes, 32 tensors; reload parity on 64 test windows: all labels equal, maximum probability difference 0.0. The same run on the workstation's CPU (load 4.5 s, probe 19.4 s, adapt 212.9 s) reproduced every metric to four decimals.
- Test suite: 51 tests in the standard venv (offline contract tests plus model-backed tests on CPU and CUDA), `ruff` clean, generator parity 5/5, release-asset validation PASS.
- Clean-runtime execution 2026-09-20: the committed notebook blob `c1aa7361` (at `3e37b6c`) ran 11/11 code cells on a Kaggle Tesla T4 from a fresh interpreter with an empty Hugging Face cache and no repository checkout (248.3 s including the pinned install and one restart; 18 files / 224 MB staged; the two model-code files digest-verified before import), reproducing the build record to four decimals: accuracy 0.8425 / MCC 0.6934, epoch 2 kept, tp 153 / tn 184 / fp 16 / fn 47, adapter 34,645,456 bytes, reload parity 64/64 with maximum probability difference 0.0. See `docs/release-verification.md`.
- Not executed: the 6,000-base ceiling, datasets near the record cap, any run on sequence outside the pinned sample and the unit tests' synthetic strings, any published downstream benchmark, the BYOD path in a hosted runtime, and any measurement beyond the numbers above.

## References

- Dalla-Torre, H. et al. (2024). Nucleotide Transformer: building and evaluating robust foundation models for human genomics. Nature Methods 22, 287–297. https://doi.org/10.1038/s41592-024-02523-z
- Upstream model card: https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species (pinned README, revision above)
- Upstream source repository: https://github.com/instadeepai/nucleotide-transformer
- Grešová, K. et al. (2023). Genomic benchmarks: a collection of datasets for genomic sequence classification. BMC Genomic Data 24, 25. https://doi.org/10.1186/s12863-023-01123-8 — https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks
- Eukaryotic Promoter Database: https://epd.expasy.org/ — Ensembl REST API: https://rest.ensembl.org/
- Pretraining and downstream datasets as disclosed upstream: https://huggingface.co/datasets/InstaDeepAI/multi_species_genome and https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks
- Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International — https://creativecommons.org/licenses/by-nc-sa/4.0/ (legal code: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
