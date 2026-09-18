---
license: cc-by-nc-sa-4.0
model_card_spec: "1.1"
pipeline_tag: fill-mask
task: "Genomics - DNA Sequence Representation"
base_model: InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
date_published: "2023-07-27"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species); the repository hosts this checkpoint only."
---

# Nucleotide Transformer v2 50M Multi-Species — DNA Sequence Representation Model

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-InstaDeepAI%2Fnucleotide--transformer--v2--50m--multi--species-ffcc4d?style=flat)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-instadeepai%2Fnucleotide--transformer-181717?style=flat&logo=github&logoColor=white)](https://github.com/instadeepai/nucleotide-transformer)
[![bioRxiv Paper](https://img.shields.io/badge/bioRxiv-2023.01.11.523679-b31b1b.svg)](https://doi.org/10.1101/2023.01.11.523679)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> [!IMPORTANT]
> **Non-Commercial Model License**
>
> The Nucleotide Transformer v2 50M Multi-Species model (`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`) is developed by InstaDeep, NVIDIA and TUM and distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
>
> The upstream license permits copying, redistribution and adaptation of the model for non-commercial purposes, subject to its attribution and ShareAlike requirements. **Commercial use is not permitted under this license.**

> [!WARNING]
> Deployment of this model through DIMER is intended only for uses consistent with the upstream CC BY-NC-SA 4.0 license, and **has not been approved** — see *DIMER deployment notes* below. Users are responsible for ensuring that their intended use is non-commercial and otherwise complies with the license terms.
>
> If the model or model weights are modified and the resulting adaptation is distributed, the adaptation must be distributed under the same license or a compatible license as required by CC BY-NC-SA 4.0.
>
> The model weights are redistributed unmodified from the upstream model repository — and this repository redistributes none of them at all: it records their digests and stages them from the pinned upstream revision at runtime. DIMER does not grant any additional rights to the model beyond those provided by the original licensor. Hosting the model on DIMER does not imply affiliation with or endorsement by InstaDeep, NVIDIA, TUM or Hugging Face.
>
> - Upstream model: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
> - Model license: CC BY-NC-SA 4.0
> - Original developers: InstaDeep, NVIDIA and TUM

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This repository provides a ready-to-run interactive Google Colab notebook that exercises the model end to end — stage and digest-verify the pinned snapshot including the Python it will execute, tokenize DNA into 6-mers, extract mean-pooled sequence embeddings, evaluate them with a nearest-centroid probe against two trivial baselines on a held-out split, and read one masked-token prediction:

- **DNA Sequence Representation Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/tutorials/nucleotide_transformer_colab.ipynb) [`nucleotide_transformer_colab.ipynb`](https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/tutorials/nucleotide_transformer_colab.ipynb)  
  *Standalone `TASK-INFERENCE` tutorial on pinned, digest-verified weights: 6-mer tokenization and its single-base fallback, 512-dimensional mean-pooled sequence embeddings, a nearest-centroid probe on a composition-matched synthetic task against majority-class and GC-content baselines, one masked-token prediction, and exported embeddings, metrics and provenance. The notebook states the non-commercial licence before it downloads anything and explains the remote-code trust boundary before it executes any.*

---

#### Description

`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` is the 50M-parameter multi-species member of the Nucleotide Transformer family (Dalla-Torre et al., bioRxiv 2023.01.11.523679), pinned here to revision `81b29e5786726d891dbf929404ef20adca5b36f1`. It is a transformer encoder trained with a masked-language-model objective on DNA: the upstream model card records pretraining on a collection of 850 genomes spanning model and non-model organisms, at a sequence length of 1,000 tokens. DNA is tokenized as **6-mers** — six consecutive `A`/`C`/`G`/`T` bases become one token, and any window containing another character (such as `N`) falls back to one token per base — over a 4,107-entry vocabulary. The pinned `config.json` names 12 hidden layers, hidden size 512, 16 attention heads, intermediate size 2,048 and rotary position embeddings; the checkpoint holds 55,904,972 parameters (the `model.safetensors` file stores 55,905,164 elements: the parameters plus 192 elements of rotary `inv_freq` buffers, 16 per layer). At inference the model returns a contextual 512-dimensional hidden state per token plus masked-token logits over the vocabulary — a representation, not a task prediction — and it is adapted by gradient training rather than by in-context conditioning.

This repository is deliberately **not** a DIMER pipeline package. It carries the model card you are reading, a standalone tutorial notebook, the pinned snapshot manifest (paths, byte sizes and SHA-256 for all 8 files) and the supporting documentation. It ships no `src/` module, no DIMER adapter and no release-asset validator, because the row is blocked on two decisions recorded below: the non-commercial licence and the remote-code requirement. What it adds over the upstream repository is therefore provenance and honesty rather than packaging: an immutable pin, digest verification that covers the two Python files the loader executes, an input contract with named ceilings, and a tutorial whose claims are bounded by what it measures.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is DNA sequence representation. Input is one or more DNA sequences as uppercase strings over `A`, `C`, `G`, `T` and `N`; output is either a contextual 512-dimensional vector per 6-mer token (pooled by the tutorial into one vector per sequence) or, through the masked-language-model head, a logit distribution over the 4,107-token vocabulary at a masked position. Envisioned application domains are genomic sequence analysis where a learned representation is wanted instead of hand-built k-mer features: extracting embeddings to feed a downstream classifier or regressor, transfer learning toward molecular-phenotype prediction tasks of the kind the upstream paper evaluates, similarity search or clustering over a sequence set, and research and educational use — including non-commercial experimentation and evaluation of genomic language models. In a larger system the model is a feature extractor or a fine-tuning starting point whose output feeds a task-specific model and expert review, never an annotation service in its own right.

Note what this repository provides against that: the tutorial demonstrates embeddings, a nearest-centroid probe and masked-token prediction. It does not implement fine-tuning, and no downstream biological application has been validated here.

###### Primary Intended Users

Intended users are computational biologists, bioinformaticians and machine-learning engineers evaluating genomic language models, in research and educational settings — and only for uses that satisfy the upstream licence's NonCommercial condition. The model assumes its users understand that a pretrained representation is not a validated predictor of any biological property; that the pinned vocabulary is a 6-mer vocabulary, so a single ambiguity code or a sequence length that is not a multiple of six changes the tokenization and shifts every downstream 6-mer boundary; that sequences from the same gene family, locus or assembly are not independent, so a random split leaks and inflates every metric; that masked-token logits are not calibrated probabilities; and that the licence question — is my use non-commercial? — is theirs to answer before they download the weights, not a checkbox inside a notebook.

###### Out-of-scope use cases

1. **Commercial use**, unless separately authorized by the relevant rights holder(s). The upstream licence's NonCommercial condition governs; nothing in this repository or in DIMER grants rights beyond it. Uses inconsistent with CC BY-NC-SA 4.0 — including distributing an adaptation of the weights under an incompatible licence, or redistributing without the required attribution — are out of scope regardless of technical merit.
2. **Capability boundary:** the checkpoint produces representations and masked-token logits. It does not itself predict promoters, enhancers, splice sites, chromatin state, variant effects or any other molecular phenotype; those require a task-specific head trained and validated on labelled data for that task. This repository implements no fine-tuning, no variant scoring and no published-benchmark reproduction, and it does not package the larger Nucleotide Transformer checkpoints.
3. **Input boundary:** sequences must be uppercase over `A`/`C`/`G`/`T`/`N`. The tutorial accepts 24 to 6,000 bases per sequence — 6,000 bases is 1,000 tokens, the length the upstream card says the model was trained at — and at most 64 sequences per call. The tokenizer's `model_max_length` is 2,048 tokens and the position embeddings allow 2,050, so longer inputs will run but leave the trained regime. Lowercase (repeat-masked) sequence, alignment gaps, amino-acid sequence, RNA with `U`, and non-human-readable formats are rejected rather than coerced.
4. **Decision boundary:** model output must not be treated as validated clinical or diagnostic advice, and must not drive patient management, therapeutic selection, genetic counselling, screening or any regulated decision, without task-specific validation, human expert review and whatever regulatory clearance the jurisdiction requires. Unsupported biological interpretation — reading a high embedding similarity or a confident masked-token prediction as evidence of function, causality or pathogenicity — is out of scope for this model and this repository.

#### Factors

###### Groups

The model is not human-centric in the demographic sense: its inputs are DNA strings with no attached identity, and no demographic attribute is available to it or evaluated here. Two things keep the question from being empty, though. First, the upstream family's larger models were trained partly on 3,200 diverse human genomes and this multi-species checkpoint on 850 genomes across species; neither the upstream authors nor this repository publishes a breakdown by ancestry, population or clade, so representation quality across human ancestries and across the tree of life is unknown rather than balanced, and genomic reference data is historically skewed toward European-ancestry individuals and toward well-studied model organisms. Second, human genomic sequence is inherently personal data: an operator who applies this model to human samples is processing information about people even though the model cannot see who they are. The factors the model's behaviour actually varies with are sequence-side: species and clade, GC content and repeat structure, the density of `N` and other ambiguity codes, sequence length relative to the 1,000-token trained window, and — for any downstream probe — the locus/family structure that a naive split will leak across. The downstream operator must stratify their own evaluation by those factors, and, where human samples are involved, by the population structure of their cohort; this repository performs no such audit and reports no group-level performance, because it measures no biological task.

###### Instrumentation

The instrument is DNA sequencing followed by assembly and annotation. The pretraining data are whole genomes as deposited in public references; the characteristics that matter are the sequencing platform and depth, the assembly quality (contig gaps appear as `N` runs), the reference build, and whether repeats were soft-masked — soft-masking writes lowercase, which this model's alphabet does not accept, so the operator's choice of masking convention silently determines what reaches the tokenizer. Assembly and basecalling errors enter as substituted or missing bases and shift 6-mer boundaries downstream of the error. At inference this repository's tutorial passes the caller's sequence verbatim: no case normalisation, no gap removal, no reverse-complement handling, no chunking of long sequences, no quality filtering. Defects upstream of the notebook therefore reach the model unchanged, and the only ones detected are the shape errors the validation stage checks — alphabet, length ceilings, sequence count, duplicate ids — plus the count of `N` bases, which it reports rather than dropping.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0` (the venv build is `2.14.0+cu130`), `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3` (`requirements.txt`); the tutorial loads to CPU and moves to `cuda:0` when CUDA is visible. Only float32 is exercised — no quantisation, autocast or compiled kernels. The recorded CPU run is in "Runtime" below: 8 files verified in 0.1 s, model load 0.4 s, 48 sequences of 300 bases embedded in 0.7 s, whole workflow 1.3 s on the reference machine after the weights were already present; a first run additionally downloads ~224 MB. Memory and time scale with tokens per sequence times batch size, because each batch is padded to its longest member; the tutorial's ceilings allow 64 sequences of 1,000 tokens, which was not measured. **Loading requires executing the checkpoint's own Python** (see *Mitigations* and *DIMER deployment notes*). Data environment: the model assumes genomic DNA resembling assembled reference genomes from the 850-genome pretraining mix. Its representations degrade, with no error and no signal, on clades far from that mix, on synthetic or heavily engineered constructs, on very short fragments, on sequence dense in `N`, and on anything that is not DNA. For a downstream probe the load-bearing assumption is that evaluation sequences are not homologous to, or drawn from the same loci as, the training ones.

#### Metrics

###### Performance Measures

This repository measures one thing, and it measures it on synthetic data: the **nearest-centroid probe accuracy** of the model's mean-pooled embeddings on a held-out split, reported next to a **majority-class baseline** and a **GC-content threshold baseline** fitted on the training half, plus the **mean and minimum cosine margin** between the two class centroids. Probe accuracy was chosen because an embedding has no intrinsic metric — the honest question about a representation is whether a downstream task can read it, and the cheapest such task that trains nothing is a centroid classifier. The two baselines are there because an accuracy without a floor is uninterpretable, and the GC baseline specifically because composition is the confounder that a genomic embedding must beat to have said anything. The margin is reported because a correct assignment with a margin in the third decimal place is a much weaker result than accuracy alone suggests, and on the recorded run the margin is exactly that small (±0.0036). No precision, recall, F1 or AUROC is reported, because with a two-class centroid rule on a balanced 24-sequence split they would add decimal places rather than information. **No task performance of any kind is claimed for this model**: the upstream paper reports downstream genomics benchmarks, and none of them is reproduced, quoted or implied here. For a real task the caller must supply labelled data from their own domain and fit and validate a task-specific model; the masked-token head, likewise, emits raw logits and no metric.

###### Decision thresholds

The tutorial applies exactly one decision rule, and it is an implicit one: **nearest centroid by cosine similarity**, which is an argmax over two similarity scores and is named as such. No probability threshold is set anywhere, because nothing in the workflow emits a calibrated probability. The masked-token head's output is ranked, never thresholded. The GC-content baseline does fit a threshold — the value that maximises accuracy on the training half only — and that value is reported with its training accuracy precisely so a reader can see it was fitted, not chosen to flatter the model. Thresholds deliberately not shipped: no embedding-similarity cutoff for "same class", no logit cutoff for "confident prediction", and no acceptance threshold on probe accuracy. A deployment that needs any of these owns calibrating it on its own labelled data, weighing the asymmetric cost of a false positive against a false negative in its setting — in a genomics triage workflow a missed candidate usually costs more than a spurious one, which argues for a permissive cutoff plus expert review. The constraints that *were* fixed during development are input-contract limits, not decision thresholds: 24 to 6,000 bases per sequence, at most 64 sequences per call, and the `A`/`C`/`G`/`T`/`N` alphabet.

###### Approaches to uncertainty and variability

The probe accuracy comes from a **single seeded stratified split** (half train, half held out; seed 42) of 48 generated sequences — 24 held out. There is no cross-validation, no bootstrap and no repeated-seed averaging, and therefore **no dispersion estimate is reported**; with 24 held-out items any interval would be dominated by the split itself. A caller who needs one repeats the workflow across seeds, or supplies enough labelled sequence for k-fold with locus-aware folds, and owns that design. Sources of variability: the sample generator and the split are both seeded explicitly, so the default path is reproducible on a fixed runtime; the forward pass runs under `torch.no_grad()` in `eval()` mode, so inference is deterministic for a fixed device and dtype; non-deterministic GPU kernel selection and float accumulation order are **not** seed-controlled and can move a cosine similarity in its low-order digits — which matters here, because the reported margin is itself of order 10⁻³. Masked-token logits are raw scores, not calibrated probabilities, and nothing in this repository calibrates them; a caller who needs a probability must fit a calibrator on held-out labelled data. Embeddings carry no uncertainty estimate at all.

#### Ethical considerations and biases

###### Data

Upstream discloses the pretraining data for this checkpoint as a collection of 850 genomes from a wide range of species, including model and non-model organisms, and lists `InstaDeepAI/multi_species_genome` and `InstaDeepAI/nucleotide_transformer_downstream_tasks` as associated datasets; the upstream card does not publish a per-species, per-assembly or per-population breakdown, and that is where disclosure ends. Whether human genomic sequence is included in the multi-species mix, and from which cohorts, is not established by the upstream card either way — unknown, not ruled out. That matters because human genomic sequence is personal data and is in principle re-identifying: a model trained on it stores statistical regularities rather than records, but an operator who *applies* it to human samples is processing personal and potentially health data. This repository distributes documentation and a notebook; it distributes **no weights, no model code and no sequence data** — the snapshot manifest records digests and the notebook stages the files from the pinned upstream revision at runtime. The operator owns the audit of whatever they submit: patient-derived or clinical sequence may be health data under their jurisdiction's law, unpublished assemblies may be confidential or third-party proprietary, consent for secondary use may not extend to the analysis they have in mind, and sequences of regulated agents may be export-controlled. Inference is local and transmits nothing, which bounds exposure but discharges none of those obligations.

###### Human Life

This model is not intended, in this repository, for decisions in health, safety, criminal justice, employment, credit or housing, and it has not been validated or certified for any of them by anyone. No clinical validation, no regulatory clearance, no external review board and no testing with any specific group has taken place here; the only evidence is one recorded CPU run on synthetic sequences. Use in a sensitive domain is nevertheless foreseeable — genomic language models are applied to variant interpretation, pathogen characterisation and clinical research pipelines — and would be admissible only with a task-specific model validated on data representative of the deployment, a domain expert reviewing every consequential output, locus- and cohort-aware evaluation, and the applicable clinical-laboratory and regulatory clearance. An embedding or a masked-token logit from this model is a hypothesis to test, never a result to act on, and no output of it should be reported to a patient or clinician as validated.

###### Mitigations

1. **Supply-chain integrity:** the model id and the 40-hex revision are constants in the notebook and in this card; the snapshot manifest records path, byte size and SHA-256 for all 8 files (223,752,120 bytes total); the notebook asserts that the inline manifest names the pinned identity, stages only absent entries from the Hub **at that revision** (never `main`), and re-hashes every file before anything is loaded. The `model.safetensors` digest equals the `oid sha256` of the Hub LFS pointer at the pinned revision.
2. **The remote-code files are inside the verified perimeter:** `modeling_esm.py` and `esm_config.py` are manifest entries, so the Python that `trust_remote_code=True` will import is digest-verified before it is imported, and an upstream change fails loudly instead of executing. `tools/check_assets.py` fails the build if those two files leave the manifest, or if the notebook's verification step stops preceding its `trust_remote_code=True` load. This is a narrowing of the trust boundary, not a removal of it — see the honest statement in *DIMER deployment notes*.
3. **No vendoring:** the repository ships neither weights nor the upstream Python (`weights/**/*.safetensors` and `weights/**/*.py` are git-ignored), so nothing executable is redistributed here and the licence question stays with the upstream files.
4. **Input integrity:** the tutorial's `validate_sequences` rejects non-string input, sequences shorter than 24 bases or longer than the 6,000-base trained ceiling, characters outside `A`/`C`/`G`/`T`/`N` (naming them, so lowercase and gaps fail loudly rather than being coerced), more than 64 sequences per call, and id lists that are the wrong length or not unique; it returns an input manifest recording the ceilings, the observed lengths and the ambiguity-base count.
5. **Evidence integrity:** the probe's split is stratified and seeded; both baselines are fitted on the training half only; the sample generator asserts that the two classes have an identical base multiset before any conclusion is drawn from the comparison; and the cosine margin is reported alongside the accuracy so the strength of the separation is visible.
6. **Refusals:** the tutorial exposes no fine-tuning, no variant scoring, no benchmark reproduction and no artifact export of model weights; the notebook's BYOD path is off by default and never runs on the default path.

Not implemented, and therefore not claimed: no security review of the upstream Python beyond reading what it does architecturally, no sandboxing of its execution, no differential-privacy or memorisation analysis, and no fairness audit of any kind.

###### Risks and harms

1. **Executing third-party code:** the default load path runs upstream Python in the user's runtime. Pinning and digest verification mean the code cannot change without detection, but they do not make it safe; a reader who skips Section 4 of the notebook may not realise this happened. The operator bears the risk, and it is the reason this row is not DIMER-qualified.
2. **Licence breach through inattention:** the weights are non-commercial, and the most likely harm from this repository is somebody embedding them in a commercial product because a notebook ran without complaint. The card, the README and the notebook all state the condition before the download; nothing enforces it, and nothing can.
3. **Composition confounding in a downstream probe:** GC content and repeat structure separate many naive genomic datasets, so an embedding-based classifier can score well while learning nothing about the biology of interest. The tutorial's sample is built to expose exactly this, and a user who omits the baseline on their own data will not see it.
4. **Locus and homology leakage:** related sequences on both sides of a split turn memorisation into apparent generalisation; the likelihood is high for any dataset assembled by a database query, and neither the model nor this repository can detect it.
5. **Tokenization surprises:** one `N`, a lowercase region, or a length that is not a multiple of six changes the token sequence — a single ambiguity code costs six bases of context and shifts every boundary after it. A user who filters sequences by base count rather than token count can silently compare sequences the model saw very differently.
6. **Over-reading a representation:** cosine similarity between two embeddings is not homology, function or pathogenicity. The magnitude of the harm ranges from a wasted experiment to a mistaken claim in a manuscript, and the recorded margins here are small enough to make over-reading easy.

###### Use cases

The model must not be used to infer, from a person's DNA, health status, disease risk, ancestry or any other personal characteristic used to make decisions about them — in employment, insurance, credit, education, immigration or healthcare access — nor to re-identify individuals from genomic data, nor for genetic surveillance or population profiling. It must not be used to produce clinical or diagnostic outputs presented to patients or clinicians as validated, nor to generate model-predicted annotations deposited into public sequence databases without marking them as predictions. It must not be used to design, screen or optimise sequences for toxins, pathogens or any agent intended to cause harm, nor to circumvent biosecurity screening of synthesised sequences. It must not be used to process sequence the operator has no right to process — patient material outside the scope of its consent, third-party proprietary assemblies, or export-controlled material. And it must not be used for **any commercial purpose** while the upstream CC BY-NC-SA 4.0 licence governs, nor redistributed without its attribution and ShareAlike conditions. These prohibitions hold even where the model would produce an accurate answer.

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
| Parameters | 55,904,972 (file stores 55,905,164 elements: + 192 rotary buffer elements) | measured from the pinned checkpoint |
| Trained sequence length | 1,000 tokens (≈ 6,000 bases) | upstream model card |
| Architectural maximum | `model_max_length` 2,048 tokens; `max_position_embeddings` 2,050 | pinned tokenizer/model config |
| Pretraining data | 850 genomes, model and non-model organisms | upstream model card |

Two upstream-documentation discrepancies, recorded rather than smoothed over: the upstream card states a vocabulary size of **4,105**, while the pinned `vocab.txt` and `config.json` both give **4,107**; and it states a maximum tokenized length of **1,000**, while the pinned tokenizer config allows **2,048**. This card uses the pinned files for the vocabulary and the upstream card's trained length as the operating ceiling, which is the conservative reading of the pair.

## Inputs

- **Format:** DNA as an uppercase string over `A`, `C`, `G`, `T` and `N`. Lowercase (soft-masked repeats), alignment gaps (`-`, `.`), RNA `U`, amino-acid letters and whitespace are rejected by the tutorial's validation stage, which names the offending characters; the upstream model itself has no such guard and would map unknown characters to `<unk>`.
- **Tokenization:** the native Transformers `EsmTokenizer` (loaded **without** remote code) reading the checkpoint's `vocab.txt`. Sequences are consumed left to right in windows of six: a clean window of `A`/`C`/`G`/`T` becomes one 6-mer token; a window containing anything else, and any trailing remainder shorter than six, falls back to one token per base. A `<cls>` token is prepended.
- **Length:** the upstream trained length is 1,000 tokens ≈ 6,000 bases, which this repository adopts as its ceiling (`MAX_BASES = 6000`); the architecture allows 2,048 tokens. Sequences are not chunked automatically — a caller with longer sequence owns the windowing and the aggregation.
- **Batching:** batches are padded to the longest member, so mixing very short and very long sequences wastes compute; the tutorial caps a call at 64 sequences and pools over non-padding tokens only.
- Repository-specific, not upstream: the 24-base minimum, the 6,000-base ceiling, the 64-sequence cap, the alphabet rejection and the id-uniqueness rule are this repository's input contract. The upstream model imposes only the position-embedding limit.

## Outputs

- **Token representations:** the last hidden state, shape `(batch, tokens, 512)`. The tutorial mean-pools over non-padding tokens to one 512-dimensional vector per sequence. These are representations; they carry no label, no class and no calibrated score, and any quality claim about them requires a downstream labelled task.
- **Masked-token logits:** shape `(batch, tokens, 4107)` from the masked-language-model head. These are raw logits — rankable, not probabilities — and the base model does not perform any downstream phenotype prediction task directly.
- **What the base model does not output:** class labels, variant effect scores, per-base functional annotations, or any of the downstream genomics benchmark predictions reported in the upstream paper. Those require task-specific heads trained on labelled data, which this repository does not provide.

## Example usage

The tutorial notebook is the supported path; this is the same workflow condensed. It mirrors the upstream model card's own snippet, with the pin and the verification this repository adds. Requires the pins in `requirements.txt` (`torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`).

```python
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
MODEL_REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"

# 1. Stage the pinned revision (verify the digests against the repository manifest before use;
#    tutorials/nucleotide_transformer_colab.ipynb Section 3 does this file by file).
local = snapshot_download(MODEL_ID, revision=MODEL_REVISION)

# 2. Tokenizer: native EsmTokenizer, no remote code.
tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)

# 3. Model: requires the checkpoint's own Python (bias-free SwiGLU feed-forward; see above).
model = AutoModelForMaskedLM.from_pretrained(local, local_files_only=True, trust_remote_code=True).eval()

# 4. Prepare DNA and run inference.
sequences = ["ATTCCGATTCCGATTCCGACGTACGTACGT", "ACGTACGTACGTAAACCCGGGTTTACGTAC"]
batch = tokenizer(sequences, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)

# 5. Mean-pool the last hidden state over non-padding tokens -> one 512-d vector per sequence.
hidden = outputs["hidden_states"][-1]
mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
embeddings = (hidden * mask).sum(1) / mask.sum(1)
print(embeddings.shape)          # torch.Size([2, 512])
print(outputs["logits"].shape)   # torch.Size([2, tokens, 4107])
```

Executed with the pins above on CPU: `embeddings.shape` is `(2, 512)` and the tokenizer turns the first sequence into `['<cls>', 'ATTCCG', 'ATTCCG', 'ATTCCG', 'ACGTAC', 'GTACGT']`.

## Limitations

- A pretrained representation does not guarantee downstream predictive validity: the model was trained to predict masked DNA tokens, and nothing about that objective promises that its embeddings are informative for the phenotype you care about. Establishing that requires a labelled task and a validated task-specific model.
- Performance depends on how close your sequence is to the 850-genome pretraining mix. Species, clade, GC regime and repeat content all shift that distance, and the model reports no out-of-distribution signal.
- Genomic context and preprocessing change the input the model sees: 6-mer boundaries, `N` runs, soft-masking, reverse-complement orientation, and where a window was cut all alter the tokenization and therefore the representation.
- Downstream tasks require task-specific validation with locus- or homology-aware splits. A random split of related sequences produces metrics that will not survive contact with novel sequence.
- Pretrained model output is not causal biological evidence. Similarity in embedding space reflects sequence statistics the model learned; attributing function, regulation or pathogenicity to it requires experimental or orthogonal computational support.
- Clinical or medical use requires substantially more validation than general inference, including cohort-representative evaluation, expert review and regulatory clearance — none of which this repository provides.
- Repository-specific: this row implements no fine-tuning, measures only a nearest-centroid probe on synthetic data, and has been executed only on CPU with the pins recorded below.

## DIMER deployment notes

| Field | Status |
|---|---|
| **DIMER status** | **Planned / conditional** — not approved, not deployed, not live. Two gates are open. |
| Licence status | **Conditional** — CC BY-NC-SA 4.0 |
| Commercial use | **Not permitted** under the upstream licence |
| Redistribution | Permitted subject to the attribution, NonCommercial and ShareAlike conditions |
| DIMER deployment | Non-commercial use only, if and when approved |
| Weights | Would be redistributed unmodified; this repository redistributes none and stages them from the pinned revision |
| Remote code | **Required** — see below |
| Upload format | `model.safetensors` (223,642,688 bytes), plus the tokenizer files and the two Python files the loader executes |

**Gate 1 — licence.** The upstream weights are CC BY-NC-SA 4.0. Hosting them on DIMER is a decision about whether every use the platform enables satisfies the NonCommercial condition, and about how the attribution and ShareAlike conditions are discharged for anything derived from them. That decision has not been made, and nothing in this repository should be read as having made it.

**Gate 2 — remote code.** This checkpoint cannot be loaded by the native Transformers ESM classes: its `config.json` carries an `auto_map` instead of a `model_type`, and, materially, its feed-forward block is a bias-free SwiGLU that the native implementation does not have — so loading these weights into the native class would not reproduce this model. `trust_remote_code=True` is therefore an architectural requirement here rather than a shortcut around a loading error, which is exactly the distinction the fleet's asset rules draw. This repository narrows the boundary as far as it can — the two Python files are pinned, listed in the manifest, and digest-verified before they are imported — but a model that requires unresolved remote code is not ordinarily qualified for DIMER, and no security review of that code has been performed beyond reading what it does architecturally.

Because both gates are open, this repository deliberately ships **no DIMER pipeline module, no adapter and no release-asset validator**. It is a model card and a tutorial, and the MODEL_MATRIX row remains **HOLD**.

## Attribution

- The model is developed by **InstaDeep, NVIDIA and TUM** (Technical University of Munich), as stated in the upstream model card ("Developed by: InstaDeep, NVIDIA and TUM").
- Original publication: Dalla-Torre, H., Gonzalez, L., Mendoza-Revilla, J., Lopez Carranza, N., Henryk Grzywaczewski, A., Oteri, F., Dallago, C., Trop, E., Sirelkhatim, H., Richard, G., Skwark, M., Beguir, K., Lopez, M., Pierrot, T. *The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics*, bioRxiv 2023.01.11.523679.
- Upstream Hugging Face model ID: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`.
- Upstream source repository: https://github.com/instadeepai/nucleotide-transformer.
- The checkpoint's `modeling_esm.py` and `esm_config.py` carry Meta Platforms and Hugging Face copyright headers and derive from the Transformers ESM implementation; they are modified by InstaDeep for this architecture.

## Repository and code licensing

Three licences apply to different things, and conflating them is the mistake this section exists to prevent:

- **Model weights** (`model.safetensors`): **CC BY-NC-SA 4.0** — attribution, NonCommercial, ShareAlike. This is the licence that governs what you may do with the model.
- **The checkpoint's supporting Python** (`modeling_esm.py`, `esm_config.py`): carries Meta/Hugging Face copyright headers and derives from the Apache-2.0 Transformers ESM implementation. A permissive licence on this code **does not** remove the NonCommercial condition from the weights.
- **This repository's own content** (this card, the notebook, the tooling): Apache-2.0, per `LICENSE`. It grants you nothing with respect to the upstream weights.

The upstream licence is not an OSI-style open-source software licence: CC BY-NC-SA 4.0 carries a NonCommercial restriction and should not be described as open source.

## Runtime

- Pins (`requirements.txt`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`. Python 3.12, Windows venv `dimer-next16`.
- Executed on CPU 2026-09-18 (`CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1`, float32): all 8 manifest entries verified in 0.1 s; `AutoModel.from_pretrained(..., trust_remote_code=False)` **refused** the checkpoint with a `ValueError` naming the custom code, confirming that remote code is required rather than assumed; `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)` loaded in 0.4 s with no warnings, reporting 55,904,972 parameters and 55,905,164 state-dict elements; the tokenizer loaded **without** remote code and turned `ATTCCGATTCCGATTCCGACGTACGTNACGT` into `['<cls>', 'ATTCCG', 'ATTCCG', 'ATTCCG', 'ACGTAC', 'G', 'T', 'N', 'A', 'C', 'G', 'T']`, showing the single-base fallback after the `N`; 48 generated 300-base sequences (51 tokens each) embedded to 512 dimensions in 0.7 s; the nearest-centroid probe scored **0.9583 accuracy (23 of 24 held-out sequences)** with a mean cosine margin of 0.0037 and a minimum of 0.0016, against a majority baseline of 0.5 and a GC-content baseline that scored 0.625 on the training half and **0.375** held out — at or below chance, as the composition-matched construction predicts; and masking one token in a repeated `ATTCCG` context returned `ATTCCG` as the top prediction (logit 16.633) ahead of `ATTCCC` (14.902), `ATTCGG` (14.308), `ATTCCA` (13.570) and `ATTCAG` (13.484). Whole notebook 33.2 s with the weights already staged.
- A pre-flight script with the same sample but a different stratified split scored 1.0 on its 24 held-out sequences. The difference between 1.0 and 0.9583 is one sequence and one split; it is recorded rather than averaged away, because with 24 held-out items and a cosine margin of order 10^-3 that is exactly the resolution this evidence has. These are sample-sanity observations on synthetic sequence, not a benchmark.
- Tutorial execution: the committed notebook blob `9f67399f7bcf` at commit `f9dc86f` executed 9/9 code cells in a clean Kaggle Tesla T4 runtime in 218.6 s, starting with an empty Hugging Face cache and no repository checkout. It downloaded and digest-verified all 8 files, including the two pinned Python files before remote-code import; the nearest-centroid probe scored 0.9583 accuracy (n = 24) against majority 0.5 and GC-threshold 0.375. The earlier local CPU pre-flight is retained in `docs/release-verification.md`. The hosted run establishes technical execution but does not clear either gate above.
- Not executed: the CUDA path, any run on real genomic data, any fine-tuning, any published downstream benchmark, and any measurement beyond the times above.

## References

- Dalla-Torre, H. et al. (2023). The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics. bioRxiv 2023.01.11.523679. https://doi.org/10.1101/2023.01.11.523679
- Upstream model card: https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species (pinned README, revision above)
- Upstream source repository: https://github.com/instadeepai/nucleotide-transformer
- Pretraining and downstream datasets as disclosed upstream: https://huggingface.co/datasets/InstaDeepAI/multi_species_genome and https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks
- Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International — https://creativecommons.org/licenses/by-nc-sa/4.0/ (legal code: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
