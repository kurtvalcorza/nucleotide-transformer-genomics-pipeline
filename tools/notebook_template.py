"""Template for the standalone E2E tutorial notebook (consumed by ``tools/build_notebook.py``).

2026-10-17 tutorial review (row 58, T58-1..T58-8): the trained depth is re-selected on validation by the grid in
docs/nucleotide-transformer-preregistration.md, the validation windows that overlapped training were re-drawn, the
held-out result is a fresh locus-disjoint 400-window test scored only under the pre-registered settings, the old
400 test windows are reported, labelled, as the exposed development test, the pre-training overlap (DAT9) and the
origin's locus overlap are stated, BYOD refusals are one message that invalidates the run (DAT20), and the notebook
declares NOTEBOOK_SPEC 2.5, its kind and the DOC17 use notice.

Numbers quoted in the prose name their source: the CPU build record of this revision (2026-10-17, workstation CPU,
pending the hosted T4 run) or the Colab T4 run of commit 7b53e06 (2026-10-08) for the pre-fix notebook.
"""


def _guarded(code: str) -> str:
    """DAT20: a cell after Section 4 runs only while Section 4's input is accepted.

    Form fields stay at top level so the form still renders; the rest of the cell moves under `else:`, so a refused
    BYOD input leaves the later cells printing `skipped`, not results from an earlier input."""
    lines = code.split("\n")
    params = [i for i, line in enumerate(lines) if "# @param" in line]
    head, body = (lines[: params[-1] + 1], lines[params[-1] + 1 :]) if params else ([], lines)
    while body and not body[0].strip():
        body.pop(0)
    indented = ["    " + line if line.strip() else line for line in body]
    skip = (
        "if RUN_STATE['refused']:\n"
        "    print({{'skipped': 'Section 4 refused the BYOD input, so this cell does not run on stale data', 'reason': RUN_STATE['refused'], 'what_to_do': 'fix the file (or set USE_BYOD = False), then select Section 4 and choose Runtime -> Run after'}})\n"
        "else:\n"
    )
    return ("\n".join(head) + "\n\n" if head else "") + skip + "\n".join(indented)



# ruff: noqa: E501

TEMPLATE = {
    "package": "nucleotide_transformer_genomics_pipeline",
    "repo_name": "nucleotide-transformer-genomics-pipeline",
    "stem": "nucleotide_transformer",
    "notebook_name": "nucleotide_transformer_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "kind": "TUTORIAL",
    "use_notice": (
        "> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,\n"
        "> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party\n"
        "> model weights and datasets under their own licenses (see the model card) and uses compute\n"
        "> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are\n"
        "> responsible for having the right to use it. Results here are tutorial evidence, not\n"
        "> benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum\n"
        "> extent permitted by law, the maintainers and the DIMER platform accept no liability for\n"
        "> its use."
    ),
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "NucleotideTransformerPipeline",
    "weights_key": "nt-v2-50m-multi-species",
    "modules": ["pipeline.py", "metrics.py", "samples.py", "protocol.py"],
    "runtime_imports": ["torch", "transformers"],
    "title": "Nucleotide Transformer v2 50M — DIMER E2E promoter fine-tuning tutorial (standalone)",
    "badges": [
        ("GitHub", "https://img.shields.io/badge/GitHub-nucleotide--transformer--genomics--pipeline-181717?logo=github", "https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline"),
        ("Model card", "https://img.shields.io/badge/Model%20card-MODEL__CARD.md-blue", "https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/MODEL_CARD.md"),
        ("Licence", "https://img.shields.io/badge/weights-CC%20BY--NC--SA%204.0-orange", "https://creativecommons.org/licenses/by-nc-sa/4.0/"),
        ("Upstream", "https://img.shields.io/badge/Hugging%20Face-InstaDeepAI%2Fnucleotide--transformer--v2--50m--multi--species-yellow", "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"),
    ],
    "model_cell_note": (
        "There is no fallback to a different download. **This checkpoint ships its own model code**, and the manifest lists "
        "the two Python files the loader executes (`modeling_esm.py`, `esm_config.py`), so `verify_snapshot` has re-hashed "
        "them before `from_pretrained` imports them with `trust_remote_code=True` — the tokenizer loads natively with "
        "`trust_remote_code=False`. Digest verification proves the executed code is the pinned upstream code byte for byte; "
        "it is not a safety claim about that code, and this is the one row of the fleet where remote code runs at all "
        "(accepted for this row on 2026-09-20)."
    ),
    # NTP-m6: replaces the generic getattr(pipe, 'source', ...) fallback with values the cell computed.
    "load_print": "print({'device': pipe.device, 'revision': snapshot['revision'], 'weights_dir': str(WEIGHTS_DIR), 'weight_sha256': pipe.weight_sha256[:16] + '...', 'remote_code_executed': pipe.remote_code_executed, 'encoder_is_base': pipe.encoder_is_base})",
    "capability": "DNA sequence representation (one 512-d mean-pooled vector per sequence) and bounded supervised fine-tuning of a promoter classifier — a mean-pooled head plus the last encoder blocks, their number chosen on validation by a pre-registered grid over 0, 2 and 12 — on labelled `{id, sequence, label}` records, using the pinned `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` weights (CC BY-NC-SA 4.0)",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated hash-locked environment from the pinned dependencies (nothing is installed into the notebook kernel, so no restart is needed), stages and digest-verifies the "
        "pinned `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` snapshot (a 224 MB `model.safetensors` and the two "
        "model-code files, all re-hashed before the code is imported), reads the human promoter sample and the fresh test "
        "carried **inline** in the package (no data download at all), validates the 1,600 / 200 / 400 split and checks it by exact "
        "sequence, reverse complement and genome locus, embeds a few sequences and predicts a masked 6-mer through the inference "
        "contract with an input manifest and a rejection probe, scores two non-neural baselines and a logistic probe on the "
        "**frozen** embeddings over the 400-window exposed development test, runs the pre-registered selection grid (the head plus "
        "0, 2 or 12 trained encoder blocks, each chosen epoch by validation MCC) and keeps the configuration its rule selects, scores "
        "the 400-window **fresh test** under the pre-registered settings (baselines, frozen probe and the adapted model), exports "
        "the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. "
        "The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no "
        "configuration edit (NOTEBOOK_SPEC 2.5 §5). The grid trains three configurations: about 2–3 min on a T4 (estimated from "
        "the 31.8 s two-block run of 2026-10-08) and 16 min on the build workstation's CPU (2026-10-17); every "
        "sequence is 47 tokens long, so CPU is workable — a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` and `BYOD_PATH` (a CSV already in the runtime; on Colab an empty path opens an upload dialog) in Section 4 and re-run from that cell to supply one "
        "CSV with a header naming `sequence` and `label` (0/1; an optional `id`) — at least 12 distinct sequences (the 65 / 15 / 20 split must leave 8 training records) of 12..6,000 "
        "A/C/G/T/N bases with both labels present. The records pass through the same validation, exact-sequence-disjoint split, "
        "baselines, frozen probe, fine-tuning, evaluation on your own test split, artifact export and reload-parity cells as the "
        "promoter sample. A BYOD run is **exploratory**: the fresh test is not scored, and the report is written as "
        "`outputs/nucleotide_transformer_evaluation_report_exploratory_<n>.json`. A file the notebook refuses is reported as one message, and the "
        "later cells skip until Section 4 accepts an input. Uploaded files stay inside this runtime. BYOD is optional and never part "
        "of the default path."
    ),
    "intro": (
        "`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` is the smallest Nucleotide Transformer v2 (Dalla-Torre et "
        "al., 2024): a 12-block, 512-wide encoder with rotary positions and a bias-free SwiGLU feed-forward, pre-trained by "
        "masked 6-mer prediction on 850 genomes (55,904,972 parameters as loaded with its masked-LM head; the encoder alone "
        "is 53,534,401). Its weights are published under **CC BY-NC-SA 4.0** — attribution, **non-commercial**, share-alike — "
        "and every embedding, probe and adapter below inherits those conditions. The checkpoint cannot be loaded by the "
        "native Transformers ESM classes (its `config.json` carries an `auto_map` and the native block has no bias-free "
        "SwiGLU), so it ships its own `modeling_esm.py`; Section 3 verifies that file's digest before it is imported.\n\n"
        "What this notebook adds to representation is **adaptation with labelled sequences**. The task is the human "
        "non-TATA promoter benchmark of Genomic Benchmarks: 251-base windows of the GRCh38 reference that either centre on "
        "a promoter from the Eukaryotic Promoter Database or contain none. Promoters are GC-rich, so the composition-only "
        "baseline is not trivial. The question is narrow: does a bounded fine-tuning of a small mean-pooled head plus a few "
        "encoder blocks on 1,600 windows move the held-out **MCC** past the frozen probe and both baselines?\n\n"
        "**How the answer is kept honest.** The default depth (two blocks under a mean-pooled head) was chosen in the build "
        "on comparisons whose split was not recorded; their accuracies fit the 400 test windows of this sample, so those "
        "windows are treated as **exposed** and reported only as the *exposed development test*. The depth is re-chosen here on "
        "validation by a grid fixed in advance (`docs/nucleotide-transformer-preregistration.md`): 0, 2 or 12 trained blocks, "
        "keep the frozen encoder unless beaten by more than 0.03 validation MCC, otherwise keep 2 unless beaten by at least "
        "0.03. The held-out answer comes from a **fresh test** of 400 windows that had never been scored, at least 251 bases from "
        "every other window on the genome, and it is scored only under those pre-registered settings. The fresh test is scored for the "
        "first time on the hosted T4 run of this revision; its numbers, and which run is the evaluation of record, are in "
        "`docs/release-verification.md`. Nothing here is a claim about your sequences: it is one split of one benchmark's "
        "labelling convention.\n\n"
        "**Pre-training overlap (DAT9).** The model was pre-trained, without labels, on 850 genomes "
        "(`InstaDeepAI/multi_species_genomes`); that list includes the human **GRCh38.p14** assembly every window here is read "
        "from (entry 784 of its `urls.txt`), as well as primate and rodent genomes. The published loader places human in its "
        "50-genome validation slice, and the upstream card does not say whether that slice was used for pre-training, so "
        "treat every window, including the fresh test, as possibly **seen as unlabelled sequence during pre-training** (never with "
        "promoter labels). That can make the frozen representation look better on these windows than on a genome the model never "
        "saw; its size is not measured. No human-reference window can avoid it.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (an 8-file manifest: weights, config, tokenizer "
        "files, README and the two model-code files); the upstream repository also hosts a `pytorch_model.bin` and a JAX "
        "checkpoint, which are not in the manifest and are never staged or loaded, so no pickle is opened anywhere. The "
        "adapter written in Section 9 is safetensors too."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter and knows what a DNA sequence and GC content are, and wants to see how a genomic language model turns sequences into representations, how to test honestly whether fine-tuning it beats simple baselines, and how to export the result under its licence. The audience is bioinformatics students and practitioners preparing their own labelled sequences; no prior experience with the Nucleotide Transformer, transformers or fine-tuning is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is workable (every window is 47 tokens); a T4 is faster.\n\n**Input → Model → Output.**\n\n| | Representation | Bounded fine-tuning |\n|---|---|---|\n| Input | DNA strings of A/C/G/T/N, 12..6,000 bases | labelled records `{{id, sequence, label}}`: 2,200 human promoter / non-promoter windows of 251 bases carried inline (1,600 train, 200 validation, 400 exposed development test) plus a 400-window fresh test, or your own CSV |\n| Model | the Nucleotide Transformer v2 50M encoder (6-mer tokens, 12 blocks, loaded with its digest-verified model code) | a fresh mean-pooled head plus the last 0, 2 or 12 encoder blocks (chosen on validation by a pre-registered grid) trained with cross-entropy; the rest frozen |\n| Output | one 512-d mean-pooled vector per sequence; a masked 6-mer distribution | a promoter label and probability per window, fresh-test accuracy and MCC beside two baselines and a frozen probe, and a CC BY-NC-SA 4.0 safetensors adapter |\n\n**How to use this notebook.** Choose a runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed (the recorded hosted run of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the verified snapshot with its model code — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer from the CPU build record of this notebook revision (2026-10-17, workstation CPU, seed 0; the hosted T4 run of this revision is pending, so a T4 may differ in the last digits). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the promoter sample, provenance, the split and where it overlaps on the genome *(evaluation practice)* → 5 representations and masked prediction *(core concept: a representation is not a prediction)* → 6 two baselines and the frozen probe *(evaluation practice)* → 7 bounded fine-tuning and the pre-registered depth grid *(core concept: what is trained, and choosing it on validation)* → 8 the fresh test and the exposed development test → 9 predictions, export and reload *(engineering)* → conclude."
    )]},
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees, including where its remote-code perimeter "
        "begins and ends; stage and digest-verify the immutable upstream snapshot; read a digest-pinned, licence-traced "
        "promoter sample carried inline, validate it, split it and measure where its splits overlap on the genome; embed sequences and predict a masked 6-mer "
        "through the public API and read the output contract correctly (a representation is not a prediction; the masked "
        "distribution is the pre-training objective, not a classifier); measure two non-neural baselines and a logistic "
        "probe on the frozen embeddings; run a bounded fine-tuning of a mean-pooled head plus the last encoder blocks with a "
        "stated loss, explicit hyperparameters, validation-based epoch selection and a pre-registered validation grid for the "
        "number of blocks; evaluate on a fresh, locus-disjoint test scored only under the pre-registered settings, beside a "
        "labelled exposed development test, with accuracy and MCC; and export a safetensors adapter that reloads against the pinned base with verified "
        "parity and carries the base licence with it."
    ),
    "exclusions": (
        "multi-class or multi-label genomics tasks, token-level prediction (splice sites, variant effects), sequences longer "
        "than 6,000 bases, generation, fine-tuning of the embeddings, LoRA, the other "
        "seventeen tasks of the Nucleotide Transformer benchmark or any benchmark proper (only one seeded draw of one "
        "Genomic Benchmarks task is scored here), calibration of the class probabilities, and any claim that a "
        "promoter classifier transfers to other regulatory elements or other species. The repository exposes none of these."
    ),
    "prerequisites": [
        '- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with the Nucleotide Transformer or fine-tuning. Tokens, embeddings, probes, MCC, epochs and adapters are explained where they are first used and again in the Glossary.',
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. The default path runs on CPU (float32) and uses CUDA automatically when available. Every sequence is 251 bases = 47 tokens, so the model is cheap per step: the build record measured 3 s for the frozen probe and 34 s for the six-epoch fine-tuning with per-epoch validation on an RTX 5070 Ti (558 MiB peak), and 213 s for the same fine-tuning on the build workstation's CPU (19 s for the probe). The pre-registered selection grid trains three configurations (0, 2 and 12 blocks); the CPU build record of 2026-10-17 took 16 min for the grid on that workstation, and about 2–3 min on a T4 is an estimate from the two-block run. No clean hosted CPU run is recorded: a few tens of minutes on a 2-vCPU hosted runtime is an **untested estimate** from the workstation timings, and the verified hosted runtime is a **T4 GPU** (see `docs/release-verification.md`). The pinned `torch==2.14.0` install and the 224 MB checkpoint are the large downloads of the run; the data is carried inline.",
        "- **Isolated environment location:** Section 1 builds the environment in `dimer_isolated_env_<lock digest>/` under the working directory. To put it elsewhere (a larger or persistent disk), set the environment variable `DIMER_ISOLATED_ENV` to a directory path **before** running Section 1 (for example `%env DIMER_ISOLATED_ENV=/content/dimer_env`); an existing complete environment at that path is reused.",
        "- **Knowledge:** basic Python; what a DNA sequence, GC content and a promoter are; what accuracy and Matthews correlation measure and why MCC is 0 for a constant predictor; why a probe on frozen embeddings and a fine-tuned model answer different questions.",
        "- **Data contract:** records are `{id, sequence, label}` — `sequence` a string of A/C/G/T/N of 12..6,000 bases (case-insensitive; `N` is tokenised base by base), `label` 0 or 1, `id` matching `[A-Za-z0-9_.:-]{1,64}` and unique; a dataset needs 8..20,000 records and a training split needs both labels; through the notebook's 65 / 15 / 20 split that means at least 12 distinct sequences; splitting de-duplicates by exact sequence so no sequence lands in two splits — but windows can still **overlap on the genome**, which this check cannot see (Section 4 measures it for the pinned sample). BYOD accepts one CSV with a header naming `sequence` and `label` (and optionally `id`).",
        "- **Validation is structural, not biological:** every sequence is checked for alphabet, length and label, but nothing checks that a label is right or that a window is a promoter — a mislabelled set is fine-tuned on without complaint.",
        "- **Licence:** the weights are **CC BY-NC-SA 4.0**. Copying, redistribution and adaptation are permitted for non-commercial purposes with attribution and under the same licence; **commercial use is not permitted under this licence**. The adapter this notebook writes is a derivative of the weights and carries the same conditions, recorded in its manifest.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** none. The 2,200 sample sequences and the 400 fresh-test sequences are carried inline in the carried `samples.py` with their digests. Their provenance is recorded there: interval lists from `ML-Bioinfo-CEITEC/genomic_benchmarks` (Apache-2.0) at commit `605d8539…` (four gzipped CSVs pinned by size and SHA-256) and bases from the GRCh38 human reference genome (Genome Reference Consortium, public domain) read through Ensembl REST; `tools/pin_sample.py` in the repository reproduces the draw, the validation re-draw and the fresh test from those sources and asserts the digests. Nothing is redistributed beyond 2,600 windows of a public-domain reference.",
    ],
    "cells": [
        {
            "md": (
                "## 4. The promoter sample, its provenance and the split\n\n"
                "`sample_dataset` returns the pinned draw from the inline block of the carried `samples.py` — 1,600 training, "
                "200 validation and 400 test windows, half promoter and half not in every split, the train and validation "
                "windows drawn from the origin's *train* lists and the test windows from its *test* lists — and refuses to "
                "return it unless the block still hashes to `SAMPLE_DIGEST`. Each record carries its genomic interval "
                "(chromosome, 0-based half-open start/end, strand) so any window can be looked up in the reference. "
                "`validate_dataset` checks every record against the contract (alphabet, 12..6,000 bases, label 0/1, unique "
                "ids, both labels in the training split) and reports counts, lengths, GC fractions and a digest; "
                "`check_split_disjoint` asserts that no sequence, or its reverse complement, appears in two splits; the training split's "
                "table is written to `outputs/{stem}_train.csv` (also a valid BYOD file).\n\n"
                "**Sequence-disjoint is not locus-disjoint.** Windows of two splits can overlap on the genome without sharing an "
                "exact sequence — shifted copies of one region, with the same label. The origin's own split is not locus-disjoint "
                "(4,449 of its 9,034 test intervals overlap one of its train intervals, and its negatives are tiled), and the seed-42 "
                "draw inherited that: 51 of the 200 validation windows overlapped a training window, and 113 of the 400 test windows "
                "overlapped a train or validation window (102 of them negatives). So the cell uses three sets, and `locus_overlaps` "
                "measures each on the genome:\n\n"
                "- **validation** — the 51 overlapping windows were replaced (all negatives, each at least 251 bases from every "
                "training and validation window); the check asserts **0** validation–train overlaps;\n"
                "- **exposed development test** — the 400 test windows of the draw, unchanged. They overlap training windows on the "
                "genome, and they were in view when the head type and depth were chosen in the build, so they are reported, "
                "labelled, for continuity only;\n"
                "- **fresh test** — 400 windows (200 / 200) from origin-test intervals never used before, each at least 251 bases "
                "from every other window here; the check asserts that, and that no sequence or reverse complement is shared. It is "
                "scored only under the pre-registered settings (Section 8).\n\n"
                "Look for: three split digests and the fresh-test digest, the locus check (0 validation–train overlaps, 0 fresh-test windows "
                "within 251 bases of any other window, and the exposed test's overlap count), promoter windows with a mean GC fraction around 0.62 against 0.48 for the "
                "negatives — the composition gap the GC baseline of Section 6 lives on — and four refusal probes (a "
                "duplicate id, a base outside A/C/G/T/N, a window too short, a training split with one label) each rejected "
                "before the model does anything. A BYOD file the notebook refuses is reported as **one message** naming the rule "
                "and what to do; the run is marked invalidated (`outputs/{stem}_run_status.json`), files already in `outputs/` "
                "are left alone and labelled as earlier, and the later cells skip until this cell accepts an input."
                '\n\n**Predict before running:** how many training, validation and test windows will the sample give? Can the same sequence end up in two splits — and can two windows of different splits still overlap on the genome?'
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "RUN_STATE = {{'refused': None}}\n\n\n"
                "class ByodInputError(ValueError):\n"
                "    \"\"\"A BYOD input this cell refuses (DAT20): shown as one message, not as a traceback.\"\"\"\n\n\n"
                'def byod_file(path, kind, suffixes=()):\n'
                '    """BYOD path first (works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog."""\n'
                '    if str(path).strip():\n'
                '        source = Path(str(path).strip()).expanduser()\n'
                '        if not source.is_file():\n'
                "            raise ByodInputError(f'BYOD path {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                '    else:\n'
                '        try:\n'
                '            from google.colab import files\n'
                '        except ImportError:\n'
                "            raise ByodInputError(f'BYOD is on but its path field is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set the path field.') from None\n"
                '        uploaded = files.upload()\n'
                '        if len(uploaded) != 1:\n'
                "            raise ByodInputError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '        name, payload = next(iter(uploaded.items()))\n'
                "        source = Path('work') / Path(name).name\n"
                '        source.parent.mkdir(parents=True, exist_ok=True)\n'
                '        source.write_bytes(payload)\n'
                '    if suffixes and not source.name.lower().endswith(tuple(suffixes)):\n'
                '        raise ByodInputError(f\'{{source.name}}: expected a {{kind}} ending in {{" or ".join(suffixes)}}.\')\n'
                '    return source\n'
                '\n\n'
                "# User-input refusals (DAT20): byod_file, load_byod_dataset, split_dataset and validate_dataset raise these for a bad\n"
                "# file or record (ValueError also covers malformed CSV text). On the default path nothing is caught.\n"
                "BYOD_REFUSALS = (ByodInputError, FileNotFoundError, UnicodeDecodeError, ValueError, TypeError)\n\n\n"
                "def refuse_byod(exc):\n"
                "    \"\"\"One actionable message; the run is invalidated so later cells skip, and files already on disk are labelled as earlier.\"\"\"\n"
                "    RUN_STATE['refused'] = f'{{BYOD_PATH or \"uploaded file\"}}: {{exc}}'\n"
                "    for name in ('splits', 'train_records', 'val_records', 'test_records', 'fresh_records', 'dataset_manifests', 'adapt_result', 'adapted_test', 'comparison', 'protocol_check', 'grid'):\n"
                "        globals().pop(name, None)\n"
                "    with open('outputs/{stem}_run_status.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'status': 'invalidated: Section 4 refused the BYOD input', 'reason': RUN_STATE['refused'], 'files_in_outputs': 'from an earlier run, not from this input'}}, handle, indent=2)\n"
                "    print({{'byod_refused': BYOD_PATH or 'uploaded file', 'reason': str(exc), 'what_to_do': 'fix the named file or record (or set USE_BYOD = False), then select this cell and choose Runtime -> Run after', 'later_sections': 'skipped until Section 4 accepts an input', 'earlier_outputs': 'files already in outputs/ are from an earlier run'}})\n\n\n"
                "BYOD_MIN_SEQUENCES = 12  # the smallest dataset whose 65 / 15 / 20 split leaves MIN_RECORDS (8) training records\n\n\n"
                "def validate_split(name, part):\n"
                "    # Training needs MIN_RECORDS and both labels; validation and test need one record, as linear_probe/evaluate accept.\n"
                "    try:\n"
                "        return validate_dataset(part, require_both_labels=(name == 'train'), min_records=MIN_RECORDS if name == 'train' else 1)\n"
                "    except ValueError as exc:\n"
                "        raise ValueError(f'{{name}} split ({{len(part)}} records): {{exc}}. A dataset needs at least {{BYOD_MIN_SEQUENCES}} distinct sequences, with both labels in the training split.') from None\n\n\n"
                "try:\n"
                "    if USE_BYOD:\n"
                "        byod_csv = byod_file(BYOD_PATH, 'labelled sequence CSV (sequence, label, optional id)', ('.csv',))\n"
                "        records = load_byod_dataset(byod_csv)\n"
                "        splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "        fresh_records = None  # the fresh test belongs to the pinned sample and is never scored in a BYOD run\n"
                "        data_source = 'BYOD (' + byod_csv.name + ')'\n"
                "        raw_rows = {{'byod': len(records)}}\n"
                "    else:\n"
                "        t0 = time.perf_counter()\n"
                "        splits = sample_dataset()\n"
                "        fresh_records = fresh_test_dataset()\n"
                "        data_source = f'{{CORPUS_NAME}}: {{CORPUS_REPO}}@{{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "        raw_rows = {{'origin_rows': CORPUS_ROWS, 'inline_records': sum(len(v) for v in splits.values()), 'pinned_digest': SAMPLE_DIGEST[:16] + '...', 'fresh_test_records': len(fresh_records), 'fresh_test_digest': FRESH_TEST_DIGEST[:16] + '...', 'seconds': round(time.perf_counter() - t0, 3)}}\n"
                "    dataset_manifests = {{name: validate_split(name, part) for name, part in splits.items()}}\n"
                "    splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "    disjoint = check_split_disjoint(splits if fresh_records is None else {{**splits, 'fresh_test': fresh_records}}, reverse_complement_too=fresh_records is not None)\n"
                "except BYOD_REFUSALS as exc:\n"
                "    if not USE_BYOD:\n"
                "        raise  # the pinned sample failing a check is a defect, not a user input: keep the traceback\n"
                "    refuse_byod(exc)\n\n"
                "if RUN_STATE['refused'] is None:\n"
                "    if os.path.exists('outputs/{stem}_run_status.json'):\n"
                "        os.remove('outputs/{stem}_run_status.json')  # an earlier refusal no longer applies\n"
                "    train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "    write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "    print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'reference': REFERENCE_GENOME, 'interval_bases': INTERVAL_BASES}})\n"
                "    for name, manifest in dataset_manifests.items():\n"
                "        by_label = {{label: round(sum(gc_content(r['sequence']) for r in splits[name] if r['label'] == label) / max(1, sum(1 for r in splits[name] if r['label'] == label)), 4) for label in (0, 1)}}\n"
                "        print({{name: {{'n': manifest['n_records'], 'labels': manifest['label_counts'], 'bases': manifest['bases'], 'gc_fraction': manifest['gc_fraction'], 'mean_gc_by_label': by_label, 'n_with_N': manifest['n_with_N'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "    if fresh_records is not None:\n"
                "        sets = {{'train': train_records, 'validation': val_records, 'test': test_records, 'fresh_test': fresh_records}}\n"
                "        check_locus_disjoint(sets, [('validation', 'train')])\n"
                "        check_locus_disjoint(sets, [('fresh_test', 'train'), ('fresh_test', 'validation'), ('fresh_test', 'test')], margin=LOCUS_MARGIN)\n"
                "        exposed_hits = {{hit[0] for hit in locus_overlaps(test_records, train_records + val_records)}}\n"
                "        fresh_manifest = validate_dataset(fresh_records, min_records=1)\n"
                "        print({{'locus_check': {{'validation_overlapping_train': 0, 'fresh_test_within_251_bases_of_another_window': 0, 'exposed_test_overlapping_train_or_validation': len(exposed_hits), 'of': len(test_records), 'of_them_negatives': sum(1 for r in test_records if r['id'] in exposed_hits and r['label'] == 0)}}}})\n"
                "        print({{'fresh_test': {{'n': fresh_manifest['n_records'], 'labels': fresh_manifest['label_counts'], 'digest': fresh_manifest['digest'][:16] + '...', 'chromosomes': len({{r['region'] for r in fresh_records}}), 'scored': 'only under the pre-registered settings (Section 8)'}}, 'exposed_development_test': {{'n': len(test_records), 'reported': 'labelled, for continuity: seen during the build and locus-overlapping'}}}})\n"
                "    else:\n"
                "        print({{'locus_check': 'BYOD records carry no coordinates: only exact duplicates were removed; if your windows overlap on the genome, split by chromosome or locus yourself', 'fresh_test': 'not used in a BYOD run (exploratory)'}})\n"
                "    example = train_records[0]\n"
                "    print({{'example': {{'id': example['id'], 'label': example['label'], 'region': example.get('region'), 'start': example.get('start'), 'end': example.get('end'), 'strand': example.get('strand'), 'sequence': example['sequence'][:60] + '...'}}}})\n\n"
                "    probes = {{\n"
                "        'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "        'base outside A/C/G/T/N': [{{**train_records[0], 'sequence': train_records[0]['sequence'][:-1] + 'U'}}, *train_records[1:8]],\n"
                "        'too short': [{{**train_records[0], 'sequence': 'ACGTACG'}}, *train_records[1:8]],\n"
                "        'one label only': [{{**r, 'label': 1}} for r in train_records[:8]],\n"
                "    }}\n"
                "    for name, probe in probes.items():\n"
                "        try:\n"
                "            validate_dataset(probe, require_both_labels=True)\n"
                "            print({{'probe': name, 'verdict': 'accepted'}})\n"
                "        except (TypeError, ValueError) as exc:\n"
                "            print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>1,600 / 200 / 400 plus the 400-window fresh test, each balanced between promoters and non-promoters (CPU build record, 2026-10-17). No sequence, and no reverse complement, appears in two sets. But yes, windows can still overlap on the genome: after the re-draw, validation overlaps training 0 times and the fresh test lies at least 251 bases from every other window, while 121 of the 400 exposed development test windows overlap a train or validation window (119 of them negatives). That is why the exposed set is not the held-out result. Every refusal probe was rejected before the model ran.</details>'
            ),
        },
        {
            "md": (
                "## 5. Representations and masked prediction through the inference contract\n\n"
                "The inference contract is exercised as the representation-only tutorial exercised it. A deterministic "
                "in-code pair of 300-base sequences with **exactly the same multiset of bases** — one carrying the 6-mer "
                "`ATTCCG` four times, the other the same letters permuted — is embedded beside the first four held-out "
                "windows; identical composition means no GC rule can tell the pair apart, so the cosine distance between "
                "their vectors is a small look at what the representation carries beyond composition. `validate_inputs` "
                "applies exactly the checks `embed` applies (a list of A/C/G/T/N strings of 12..6,000 bases) and returns an "
                "input manifest with per-sequence SHA-256s; a sequence with a `U` is validated too and its rejection "
                "recorded as a finding. `embed` returns one 512-d **mean-pooled** vector per sequence — a representation, "
                "not a prediction; `token_counts` shows the 6-mer tokenisation (251 bases → 47 tokens); `predict_masked` "
                "returns the model's distribution at one masked position — its pre-training objective, which is what "
                "`<mask>` inside a repeated `ATTCCG` context should recover — and `predict` is refused until an adapter "
                "exists, which is the contract's point: the checkpoint is not a classifier."
                '\n\n**Predict before running:** the pair of synthetic sequences has exactly the same bases in a different order. Will their embeddings be identical? And what will `predict` do before any adaptation?'
            ),
            "code": _guarded(
                '# NTP-M2: adapt() trains the last encoder blocks of `pipe` in place. On a re-run after Section 7 (BYOD, or an\n'
                '# experiment) put the pinned base back first, so this cell reads the frozen model it is labelled with.\n'
                'if not pipe.encoder_is_base or pipe.adapter is not None:\n'
                "    print({{'reset_to_base': len(pipe.reset_to_base()), 'reason': 'an earlier Section 7 had adapted the encoder in place'}})\n"
                '\n'
                "import csv\n"
                "import random\n\n"
                "rng = random.Random(7)\n"
                "MOTIF = 'ATTCCG'\n"
                "filler = ''.join(rng.choice('ACGT') for _ in range(300 - 4 * len(MOTIF)))\n"
                "positions = sorted(rng.sample(range(0, len(filler), 30), 4))\n"
                "with_motif = ''\n"
                "cursor = 0\n"
                "for p in positions:\n"
                "    with_motif += filler[cursor:p] + MOTIF\n"
                "    cursor = p\n"
                "with_motif += filler[cursor:]\n"
                "permuted = list(with_motif)\n"
                "rng.shuffle(permuted)\n"
                "permuted = ''.join(permuted)\n"
                "assert sorted(with_motif) == sorted(permuted) and len(with_motif) == 300\n"
                "probe_sequences = [with_motif, permuted] + [r['sequence'] for r in test_records[:4]]\n"
                "probe_names = ['motif_x4', 'same_bases_permuted'] + [r['id'] for r in test_records[:4]]\n"
                "print({{'ceilings': {{'MIN_BASES': MIN_BASES, 'MAX_BASES': MAX_BASES, 'MAX_TOKENS': MAX_TOKENS, 'MAX_SEQUENCES': MAX_SEQUENCES, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'device': pipe.device, 'remote_code_executed': pipe.remote_code_executed}}}})\n"
                "input_manifest = validate_inputs(probe_sequences, names=probe_names)\n"
                "input_manifest['findings'] = []\n"
                "try:\n"
                "    validate_inputs(['ACGUACGUACGU'])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'uracil-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'input_manifest': {{k: input_manifest[k] for k in ('n_sequences', 'bases', 'gc_fraction', 'n_with_N')}}, 'findings': len(input_manifest['findings'])}})\n\n"
                "t0 = time.perf_counter()\n"
                "vectors = pipe.embed(probe_sequences)\n"
                "embed_seconds = round(time.perf_counter() - t0, 3)\n"
                "counts = pipe.token_counts(probe_sequences)\n\n\n"
                "def cosine(a, b):\n"
                "    dot = sum(x * y for x, y in zip(a, b, strict=True))\n"
                "    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))\n\n\n"
                "pair_cosine = cosine(vectors[0], vectors[1])\n"
                "held_out_cosines = [round(cosine(vectors[0], v), 4) for v in vectors[2:]]\n"
                "checks = {{'one_vector_per_sequence': len(vectors) == len(probe_sequences), 'dimension': all(len(v) == HIDDEN_SIZE for v in vectors), 'pooling': POOLING}}\n"
                "if not all(v is True or isinstance(v, str) for v in checks.values()):\n"
                "    raise RuntimeError(f'embed output failed a sanity check: {{checks}}')\n"
                "print({{'embed': {{'n': len(vectors), 'dimension': len(vectors[0]), 'seconds': embed_seconds, 'tokens': dict(zip(probe_names, counts, strict=True)), 'checks': checks}}}})\n"
                "print({{'motif_vs_permuted_cosine': round(pair_cosine, 4), 'motif_vs_held_out_cosines': held_out_cosines, 'reading': 'same bases, different order: the distance between the pair is what the model carries beyond composition (sample-sanity, not a measurement)'}})\n"
                "with open('outputs/{stem}_embeddings.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['name', 'bases', 'tokens'] + [f'dim_{{k}}' for k in range(HIDDEN_SIZE)])\n"
                "    for name, sequence, count, vector in zip(probe_names, probe_sequences, counts, vectors, strict=True):\n"
                "        writer.writerow([name, len(sequence), count] + [f'{{x:.6f}}' for x in vector])\n\n"
                "masked = pipe.predict_masked(MOTIF * 3 + '<mask>' + MOTIF * 3, top_k=5)\n"
                "print({{'masked_prediction': masked, 'reading': 'the pre-training objective: the distribution over 6-mers at the masked position'}})\n"
                "try:\n"
                "    pipe.predict(probe_sequences[:1])\n"
                "except RuntimeError as exc:\n"
                "    print({{'predict_before_adapt': 'refused', 'message': str(exc)}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Not identical: identical composition does not mean identical representation, because the encoder reads the order of the 6-mers, so the cosine distance between the pair is above zero. `predict` is **refused** until an adapter exists — the checkpoint is a masked-language model, not a classifier; `predict_masked` returns its pre-training objective instead.</details>'
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the exposed development test\n\n"
                "This section scores the 400 **exposed development test** windows only: they were in view when the build chose the "
                "head and depth, and many overlap training windows on the genome, so read these numbers as continuity with earlier "
                "runs, not as the held-out result. The same three references are scored on the fresh test in Section 8, and only "
                "under the pre-registered settings.\n\n"
                "Three references frame the adaptation, each scored by `classification_metrics` (carried in `metrics.py`): "
                "**accuracy** and **MCC** — the Matthews correlation, 0 for any constant or chance predictor and symmetric "
                "in the two classes, which is the headline metric here — with per-class precision, recall and F1. The "
                "**majority** baseline answers the training majority label (0.5 on a balanced test split, MCC 0). The "
                "**GC-threshold** baseline is fitted on the training split alone — the midpoint threshold and direction "
                "that maximise training accuracy — and reads only composition: on the exposed development test it scores **0.715** / "
                "0.4308 with the rule *GC ≥ 0.560 → promoter* (Colab T4 run of commit 7b53e06, 2026-10-08; the CPU build record of "
                "this revision matches), which is the honest floor for a sequence model on this task. The "
                "**frozen probe** (`pipe.linear_probe`) fits an L2-penalised logistic regression on the standardised "
                "mean-pooled embeddings of the training windows and scores the test windows; the model's weights are "
                "untouched, so this is what the pre-trained representation alone knows — 0.815 / 0.6315 on the exposed "
                "development test in the same two records. Expect the probe well above the GC rule: the representation carries more than "
                "composition. The cell reports whether that ordering holds as a verdict (it does not stop the notebook, so a BYOD run still "
                "exports). If an earlier run of Section 7 adapted the encoder in place, the cell first calls `pipe.reset_to_base()`, which "
                "puts the pinned base blocks back and drops the adapter, so the probe always reads the frozen representation (and "
                "`linear_probe` refuses an adapted encoder outright)."
                '\n\n**Predict before running:** write down the test accuracy and MCC you expect for the majority answer, the GC rule and the frozen probe.'
            ),
            "code": _guarded(
                '# NTP-M2: adapt() trains the last encoder blocks of `pipe` in place. On a re-run after Section 7 (BYOD, or an\n'
                '# experiment) put the pinned base back first, so this cell reads the frozen model it is labelled with.\n'
                'if not pipe.encoder_is_base or pipe.adapter is not None:\n'
                "    print({{'reset_to_base': len(pipe.reset_to_base()), 'reason': 'an earlier Section 7 had adapted the encoder in place'}})\n"
                '\n'
                "METRICS = ('accuracy', 'mcc')\n"
                "EXPOSED_LABEL = 'your BYOD test split' if USE_BYOD else 'exposed development test (400 windows; in view during the build, locus-overlapping)'\n"
                "print({{'scored_set': EXPOSED_LABEL}})\n"
                "baseline_majority = majority_baseline(train_records, test_records)\n"
                "baseline_gc = gc_threshold_baseline(train_records, test_records)\n"
                "print({{'majority_baseline': {{k: baseline_majority[k] for k in METRICS}}, 'label': baseline_majority['label'], 'n': baseline_majority['n']}})\n"
                "print({{'gc_threshold_baseline': {{k: baseline_gc[k] for k in METRICS}}, 'threshold': baseline_gc['threshold'], 'rule': baseline_gc['rule'], 'train_accuracy': baseline_gc['train_accuracy']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_probe = pipe.linear_probe(train_records, test_records)\n"
                "print({{'frozen_probe_test': {{k: frozen_probe[k] for k in METRICS}}, 'positive': frozen_probe['positive'], 'negative': frozen_probe['negative'], 'train_accuracy': frozen_probe['train_accuracy'], 'probe': frozen_probe['probe'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "frozen_ordering = 'probe > GC rule > majority' if frozen_probe['mcc'] > baseline_gc['mcc'] > baseline_majority['mcc'] else 'the expected ordering does not hold'\n"
                "print({{'frozen_verdict': frozen_ordering}})"

            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>On the exposed development test the majority scored 0.5 / MCC 0.0 (a constant answer always has MCC 0), the GC rule 0.715 / 0.4308 and the frozen probe 0.815 / 0.6315 (Colab T4 run of commit 7b53e06, 2026-10-08; the CPU build record of this revision gives the same figures, since nothing in Sections 4–6 that they depend on changed). Promoters are GC-rich, so composition alone is a real floor; the probe well above it shows the representation carries more than composition.</details>'
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning: the pre-registered depth grid\n\n"
                "`pipe.adapt` builds a fresh **mean-pooled head** (a 512→512 `tanh` layer and a 512→2 output, 263,682 "
                "parameters, initialised from `SEED` so two runs start alike) and trains it together with the **last "
                "`layers` encoder blocks** while the embeddings and the other blocks stay frozen. Cross-entropy over the two "
                "labels, AdamW without weight decay at a fixed learning rate, gradient clipping at 1.0, batches of 16 in seeded "
                "order, no scheduler. Epoch 0 records the untrained head's validation metrics (chance); every epoch is scored on "
                "the 200 validation windows and the epoch with the highest **validation MCC** is kept (ties: accuracy, then the "
                "earlier epoch); on any exception the encoder returns to its state on entry. Every call starts from the **pinned "
                "base** blocks, so a re-run never continues from an earlier adaptation.\n\n"
                "**How many blocks to train is chosen here, on validation only.** The build chose two blocks under a mean-pooled head "
                "from comparisons whose split was not recorded (revision `26879e0`; their accuracies fit the 400-window exposed test), "
                "so that choice is re-made by the grid fixed in `docs/nucleotide-transformer-preregistration.md` before any run: "
                "`TRAINED_LAYERS` ∈ {{0 (head only, frozen encoder), 2, 12 (whole encoder)}}, everything else fixed. The rule "
                "(`select_trained_layers`): keep the frozen encoder unless another configuration beats its validation MCC by **more "
                "than 0.03**; otherwise keep 2 unless the whole encoder beats it by **at least 0.03**; within the tolerance the smaller "
                "trainable scope wins. The 0.03 is a practical tolerance (about the noise of a 200-window validation MCC), not a "
                "significance test. The head type stays mean-pooled; its choice is disclosed, not re-tested. The grid trains 0, 12 "
                "and then 2 blocks, so when 2 is kept its model is already in place.\n\n"
                "**Pre-registered or exploratory.** After training, `preregistration_check` compares every setting this run used "
                "with the pre-registration — data source and digests, split sizes and seeds, `EPOCHS`, `LEARNING_RATE`, batch size, "
                "`SEED`, the grid flag and the selected depth — and prints `preregistered` and `settings_differing`. Leave the form "
                "fields at their defaults (`TRAINED_LAYERS = -1` means *use the grid's selection*) for the pre-registered run. "
                "Changing any field, turning the grid off or using BYOD makes the run **exploratory**. An exploratory run is compared "
                "on validation and the exposed development test only; the fresh test is not scored.\n\n"
                "Watch each configuration's training loss fall while the validation MCC peaks early: 1,600 windows are enough to "
                "overfit, which is what the epoch rule is for. Only one seed is recorded: the spread across seeds has **not been measured**, "
                "and; a different `SEED` may move these numbers by an unknown amount."
                '\n\n**Predict before running:** which configuration will the rule keep: the frozen encoder, two blocks or the whole encoder? Will the validation MCC rise every epoch?'
            ),
            "code": _guarded(
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 3e-5  # @param {{type:\"number\"}}\n"
                "RUN_SELECTION_GRID = True  # @param {{type:\"boolean\"}}\n"
                "TRAINED_LAYERS = -1  # @param {{type:\"integer\"}}\n"
                "SEED = 0  # @param {{type:\"integer\"}}\n"
                "# TRAINED_LAYERS = -1 uses the grid's selection (or DEFAULT_TRAINED_LAYERS with the grid off); 0..12 forces a depth (exploratory unless it equals the selection).\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_accuracy': entry['val']['accuracy'], 'val_mcc': entry['val']['mcc']}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "GRID_ORDER = (0, 12, 2)  # GRID_LAYERS, the incumbent last so its model stays in place when it is kept\n"
                "grid, grid_results, last_run = None, {{}}, None\n"
                "if RUN_SELECTION_GRID:\n"
                "    for layers in GRID_ORDER:\n"
                "        print({{'grid_configuration': {{'trained_layers': layers}}}})\n"
                "        t0 = time.perf_counter()\n"
                "        last_run = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, layers=layers, seed=SEED, progress=report)\n"
                "        kept = last_run['history'][last_run['best_epoch']]['val']\n"
                "        grid_results[layers] = {{'mcc': kept['mcc'], 'accuracy': kept['accuracy'], 'best_epoch': last_run['best_epoch'], 'trainable_parameters': last_run['n_trainable'], 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "    grid = select_trained_layers(grid_results)\n"
                "    grid['results'] = {{str(k): v for k, v in sorted(grid_results.items())}}\n"
                "    print({{'grid_selection': grid}})\n"
                "layers_used = (grid['selected_layers'] if grid else DEFAULT_TRAINED_LAYERS) if TRAINED_LAYERS == -1 else TRAINED_LAYERS\n"
                "if grid and layers_used == GRID_ORDER[-1]:\n"
                "    adapt_result, adapt_seconds = last_run, grid_results[layers_used]['seconds']  # already trained and in place\n"
                "else:\n"
                "    print({{'final_configuration': {{'trained_layers': layers_used}}}})\n"
                "    t0 = time.perf_counter()\n"
                "    adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, layers=layers_used, seed=SEED, progress=report)\n"
                "    adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "run_settings = {{\n"
                "    'data_source': 'BYOD' if USE_BYOD else 'pinned promoter sample',\n"
                "    'sample_digest': dataset_digest(train_records + val_records + test_records),\n"
                "    'split_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'split_sizes': {{name: len(part) for name, part in splits.items()}},\n"
                "    'sample_seed': SAMPLE_SEED,\n"
                "    'split_seed': SPLIT_SEED,\n"
                "    'fresh_test_digest': dataset_digest(fresh_records) if fresh_records is not None else None,\n"
                "    'fresh_test_size': len(fresh_records) if fresh_records is not None else 0,\n"
                "    'epochs': EPOCHS,\n"
                "    'learning_rate': LEARNING_RATE,\n"
                "    'batch_size': adapt_result['batch_size'],\n"
                "    'seed': SEED,\n"
                "    'head': PREREGISTERED_SETTINGS['head'],\n"
                "    'optimizer': PREREGISTERED_SETTINGS['optimizer'],\n"
                "    'epoch_selection': adapt_result['selection'],\n"
                "    'selection_grid': bool(RUN_SELECTION_GRID),\n"
                "    'grid_layers': list(GRID_LAYERS) if grid else [],\n"
                "    'trained_layers': 'grid selection' if grid and adapt_result['layers'] == grid['selected_layers'] else adapt_result['layers'],\n"
                "}}\n"
                "protocol_check = preregistration_check(run_settings)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'head_parameters': adapt_result['n_head'], 'total_parameters': adapt_result['n_total'], 'layers': adapt_result['layers'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'train_accuracy': adapt_result['train_accuracy'], 'seconds': adapt_seconds}})\n"
                "print({{'preregistered': protocol_check['preregistered'], 'settings_differing': protocol_check['settings_differing']}})\n"
                "if not protocol_check['preregistered']:\n"
                "    print('This run is EXPLORATORY (settings differ from docs/nucleotide-transformer-preregistration.md): Section 8 reports validation and the exposed development test only, does not score the fresh test, and writes an *_exploratory_<n>.json report.')"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>In the CPU build record of this revision (2026-10-17, seed 0) the best-epoch validation MCC was 0 blocks 0.5417 (epoch 6), 2 blocks 0.6679 (epoch 2), 12 blocks 0.6628 (epoch 4), so the rule kept **the head plus the last two blocks** (no larger configuration beats 2 blocks by at least 0.03 validation MCC). The kept run\'s validation curve went 0.505 / 0.0709 → 0.805 / 0.6335 → 0.825 / 0.6679 → 0.815 / 0.6543 → 0.8 / 0.6250 → 0.78 / 0.5618 → 0.82 / 0.6593, so **epoch 2** was kept while the training loss kept falling: 1,600 windows are enough to start overfitting, which is what validation selection is for. These are validation numbers on the re-drawn 200 windows, so they differ from the pre-fix notebook\'s (0.805 / 0.621 at epoch 2 on the old validation windows, Colab T4 run of commit 7b53e06).</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation: the fresh test and the exposed development test\n\n"
                "Two sets are scored here, and they answer different questions.\n\n"
                "- The **fresh test** is the held-out result: 400 windows that had never been scored, locus-disjoint from every "
                "other window here. It is scored **only** when Section 7 printed `preregistered: True`. Then the two baselines, the "
                "frozen probe and the adapted model are all scored on it. The probe needs the frozen encoder, so the cell parks "
                "the adapter in `work/`, calls `pipe.reset_to_base()`, fits the probe and overlays the adapter again, checking that "
                "its fresh-test predictions come back unchanged. Before that, `adapted_state_check` confirms the model in memory is the selected "
                "adapted model (its depth, kept epoch and settings, and its validation scores re-measured now); if not, the fresh test "
                "is not scored. The report carries `test_look`: *pre-registered evaluation (this runtime)* for the first scoring in "
                "this runtime, *reproduction (this runtime)* for an identical later one; a runtime cannot know about other runs. Scored only under the pre-registered settings; see "
                "`docs/release-verification.md` for the recorded evaluation and its reproductions.\n"
                "- The **exposed development test** (the old 400 test windows) is scored on every run for continuity, labelled. It "
                "is not a held-out number: it was in view when the build chose the head and depth, and many of its windows overlap "
                "training windows on the genome.\n\n"
                "A pre-registered run writes `outputs/{stem}_evaluation_report.json`. An exploratory run writes "
                "`outputs/{stem}_evaluation_report_exploratory_<n>.json` instead, so it never replaces the pre-registered report. "
                "Read the fresh-test table in this order: the **MCC** first, then accuracy, then the per-class rows (the fresh-test "
                "numbers of record are in `docs/release-verification.md`; this build's CPU check kept the fresh test masked). Four hundred windows "
                "from one seeded draw give **no dispersion estimate**: the deltas are sample-sanity evidence that the adaptation "
                "contract works, not a benchmark. A gain on this benchmark's window convention says nothing about promoters in "
                "*your* sequences until you measure it."
                '\n\n**Predict before running:** will the adapted model beat the frozen probe on the fresh test, and will its margin be larger or smaller than on the exposed development test?'
            ),
            "code": _guarded(
                "import shutil\n\n"
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{metric: {{'majority': baseline_majority[metric], 'gc_threshold': baseline_gc[metric], 'frozen_probe': frozen_probe[metric], 'adapted': adapted_test[metric]}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen_probe'] = {{metric: round(adapted_test[metric] - frozen_probe[metric], 4) for metric in METRICS}}\n"
                "comparison['delta_vs_gc_threshold'] = {{metric: round(adapted_test[metric] - baseline_gc[metric], 4) for metric in METRICS}}\n"
                "comparison['per_class'] = {{'frozen_probe': {{'positive': frozen_probe['positive'], 'negative': frozen_probe['negative']}}, 'adapted': {{'positive': adapted_test['positive'], 'negative': adapted_test['negative']}}}}\n"
                "comparison['confusion_adapted'] = adapted_test['confusion']\n"
                "print({{'set': EXPOSED_LABEL, 'validation': {{k: adapted_val[k] for k in METRICS}}}})\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "verdict = evaluation_report(adapted_test, frozen_probe, [baseline_majority, baseline_gc], sample_kind='BYOD' if USE_BYOD else 'exposed development test (in view during the build; locus-overlapping)')\n"
                "print({{'verdict_exposed_development_test': verdict}})\n\n"
                "PREREGISTERED_SCORINGS = globals().get('PREREGISTERED_SCORINGS', 0)  # pre-registered fresh-test scorings earlier in this session\n"
                "look = test_look(protocol_check['preregistered'] and adapted_state_check(pipe.adapter, adapt_result, adapted_val)['matches'], PREREGISTERED_SCORINGS)\n"
                "fresh_result = None\n"
                "state = adapted_state_check(pipe.adapter, adapt_result, adapted_val)  # the model in memory is the selected one\n"
                "if protocol_check['preregistered'] and not state['matches']:\n"
                "    print({{'fresh_test': 'not scored: the model in memory is not the selected adapted model', 'mismatches': state['mismatches'], 'what_to_do': 're-run Sections 7-8'}})\n"
                "if protocol_check['preregistered'] and state['matches']:\n"
                "    fresh_adapted = pipe.evaluate(fresh_records)\n"
                "    parked = Path('work/{stem}_parked_adapter')\n"
                "    shutil.rmtree(parked, ignore_errors=True)\n"
                "    pipe.save_artifact(parked)\n"
                "    pipe.reset_to_base()\n"
                "    fresh_frozen = pipe.linear_probe(train_records, fresh_records)\n"
                "    pipe.load_artifact(parked)\n"
                "    shutil.rmtree(parked, ignore_errors=True)\n"
                "    if pipe.evaluate(fresh_records)['confusion'] != fresh_adapted['confusion']:\n"
                "        raise RuntimeError('the adapter did not come back unchanged after the frozen probe; re-run Sections 7-8')\n"
                "    fresh_majority = majority_baseline(train_records, fresh_records)\n"
                "    fresh_gc = gc_threshold_baseline(train_records, fresh_records)\n"
                "    fresh_result = {{\n"
                "        'test_look': look,\n"
                "        'n': fresh_adapted['n'],\n"
                "        'digest': FRESH_TEST_DIGEST,\n"
                "        'comparison': {{metric: {{'majority': fresh_majority[metric], 'gc_threshold': fresh_gc[metric], 'frozen_probe': fresh_frozen[metric], 'adapted': fresh_adapted[metric]}} for metric in METRICS}},\n"
                "        'per_class': {{'frozen_probe': {{'positive': fresh_frozen['positive'], 'negative': fresh_frozen['negative']}}, 'adapted': {{'positive': fresh_adapted['positive'], 'negative': fresh_adapted['negative']}}}},\n"
                "        'confusion_adapted': fresh_adapted['confusion'],\n"
                "        'gc_rule': {{'threshold': fresh_gc['threshold'], 'rule': fresh_gc['rule']}},\n"
                "        'verdict': evaluation_report(fresh_adapted, fresh_frozen, [fresh_majority, fresh_gc], sample_kind='fresh test (pre-registered settings)'),\n"
                "    }}\n"
                "    PREREGISTERED_SCORINGS += 1\n"
                "    print({{'fresh_test': look}})\n"
                "    for key in ('comparison', 'per_class', 'confusion_adapted', 'verdict'):\n"
                "        print({{key: fresh_result[key]}})\n"
                "    if not (fresh_adapted['mcc'] > fresh_frozen['mcc'] and fresh_result['verdict']['adapted_beats_baselines']):\n"
                "        print('On the fresh test the adapted model did not beat the frozen probe and both baselines on MCC; record that as the finding.')\n"
                "else:\n"
                "    print({{'fresh_test': look, 'settings_differing': protocol_check['settings_differing']}})\n\n"
                "SUFFIX = '' if fresh_result is not None else exploratory_suffix('{stem}', os.listdir('outputs'))\n"
                "evaluation_report_payload = {{\n"
                "    'test_look': look,\n"
                "    'preregistered': protocol_check['preregistered'],\n"
                "    'settings_differing': protocol_check['settings_differing'],\n"
                "    'model_state_check': state,\n"
                "    'record_of_evaluations': RECORD_OF_EVALUATIONS,\n"
                "    'preregistration': PREREGISTRATION_DOC,\n"
                "    'run_settings': run_settings,\n"
                "    'grid': grid,\n"
                "    'fresh_test': fresh_result,\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY, 'license': MODEL_LICENSE}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'frozen_ordering': frozen_ordering,\n"
                "    'exposed_development_test': {{'label': EXPOSED_LABEL, 'baselines': {{'majority': baseline_majority, 'gc_threshold': baseline_gc}}, 'frozen_probe': frozen_probe, 'adapted': adapted_test, 'comparison': comparison, 'verdict': verdict}},\n"
                "    'validation_metrics': adapted_val,\n"
                "    'adapted_beats_frozen_probe': adapted_test['mcc'] > frozen_probe['mcc'],\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "report_path = 'outputs/{stem}_evaluation_report.json' if not SUFFIX else f'outputs/{stem}_evaluation_report{{SUFFIX}}.json'\n"
                "with open(report_path, 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "if not (adapted_test['mcc'] > frozen_probe['mcc'] and verdict['adapted_beats_baselines']):\n"
                "    print('On the exposed development test the adapted model did not beat the frozen probe and both baselines on MCC; the report, export and reload still run.')\n"
                "print({{'report': report_path, 'test_look': look}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>The fresh-test answer is recorded in `docs/release-verification.md` from the hosted T4 run of this revision (the CPU build check kept the fresh test masked). On the exposed development test the same model scored 0.8425 / 0.6934 against the probe\'s 0.815 / 0.6315; the pre-fix notebook\'s adapted 0.8425 / 0.6934 there came from epoch selection on the old validation windows (Colab T4 run of commit 7b53e06). Seed-to-seed spread is **untested** (no multi-seed run is recorded), and on 400 windows the sampling error of MCC is of a similar size to these margins, so treat them as small.</details>'
            ),
        },
        {
            "md": (
                "## 9. Look at the predictions, export the adapter and reload it\n\n"
                "Eight held-out windows are printed with their GC fraction, the GC rule's answer, the adapted "
                "classifier's label and promoter probability, and the truth — so the numbers can be checked by eye: look "
                "for windows the GC rule gets wrong that the classifier gets right, and for the probabilities near 0.5 "
                "that the accuracy hides. The pair from Section 5 is classified too (both are synthetic, so there is no "
                "truth — only the observation that identical composition can receive different probabilities).\n\n"
                "`pipe.save_artifact` writes the trained tensors — the head and the last two blocks, about 35 MB — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, **the licence the adapter "
                "inherits (`cc-by-nc-sa-4.0`)**, the base model id and revision, the digest of the base `model.safetensors`, "
                "the two remote-code files the base executes, the tensor names, the file size and SHA-256, the training "
                "configuration and the epoch history (OUT8). `NucleotideTransformerPipeline.from_artifact` re-verifies "
                "the base snapshot (including the model code), checks the artifact manifest, its digest, its licence and "
                "its exact tensor set **before** deserialising, refuses any encoder tensor outside the recorded blocks, and "
                "overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). "
                "The cell asserts identical labels and near-identical probabilities on 64 exposed-development-test windows (VER4; "
                "prediction parity needs no labels, so this does not score anything). The adapter and the result file of an "
                "exploratory run carry the same `_exploratory_<n>` suffix as its report."
                '\n\n**Predict before running:** the reloaded pipeline is rebuilt from the adapter file and the pinned base. Will its 64 predictions match the in-memory model exactly?'
            ),
            "code": _guarded(
                "import shutil\n\n"
                "shown = test_records[:8]  # exposed development test windows, for reading by eye\n"
                "shown_predictions = pipe.predict([r['sequence'] for r in shown])\n"
                "for record, label, probabilities in zip(shown, shown_predictions['labels'], shown_predictions['probabilities'], strict=True):\n"
                "    gc = gc_content(record['sequence'])\n"
                "    gc_answer = int(gc >= baseline_gc['threshold']) if baseline_gc['rule'].startswith('GC >=') else int(gc < baseline_gc['threshold'])\n"
                "    print({{'id': record['id'], 'region': f\"{{record.get('region')}}:{{record.get('start')}}-{{record.get('end')}}{{record.get('strand', '')}}\", 'gc': round(gc, 3), 'gc_rule': gc_answer, 'adapted': label, 'p_promoter': probabilities[1], 'truth': record['label']}})\n"
                "pair = pipe.predict(probe_sequences[:2])\n"
                "print({{'synthetic_pair': dict(zip(probe_names[:2], [p[1] for p in pair['probabilities']], strict=True)), 'reading': 'same bases, different order; no truth exists for these'}})\n\n"
                "artifact_dir = Path(f'outputs/{stem}_adapter{{SUFFIX}}')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'license': artifact_manifest['license'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'base_remote_code_files': artifact_manifest['base']['remote_code_files']}})\n\n"
                "reloaded = NucleotideTransformerPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "parity_sequences = [r['sequence'] for r in test_records[:64]]\n"
                "before = pipe.predict(parity_sequences)\n"
                "after = reloaded.predict(parity_sequences)\n"
                "parity = {{'identical_labels': sum(a == b for a, b in zip(before['labels'], after['labels'], strict=True)), 'of': len(parity_sequences), 'max_probability_difference': max(abs(x - y) for pa, pb in zip(before['probabilities'], after['probabilities'], strict=True) for x, y in zip(pa, pb, strict=True))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch'], 'reloaded_layers': reloaded.adapter['layers']}})\n"
                "assert parity['identical_labels'] == parity['of'] and parity['max_probability_difference'] < 1e-4\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'remote_code': {{'executed': pipe.remote_code_executed, 'files': list(REMOTE_CODE_FILES), 'verified_before_import': True}},\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'dataset': CORPUS_DATASET, 'license': CORPUS_LICENSE, 'files': {{k: {{'bytes': v[0], 'sha256': v[1]}} for k, v in CORPUS_FILES.items()}}, 'reference': REFERENCE_GENOME, 'sample_digest': SAMPLE_DIGEST, 'split_digests': SAMPLE_SPLIT_DIGESTS}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'embedding_dimension': HIDDEN_SIZE, 'pooling': POOLING, 'motif_vs_permuted_cosine': round(pair_cosine, 6), 'masked_prediction': masked, 'output_files': ['outputs/{stem}_embeddings.csv', 'outputs/{stem}_input_manifest.json']}},\n"
                "    'test_look': look,\n"
                "    'preregistered': protocol_check['preregistered'],\n"
                "    'settings_differing': protocol_check['settings_differing'],\n"
                "    'grid': grid,\n"
                "    'fresh_test': fresh_result,\n"
                "    'fresh_test_digest': FRESH_TEST_DIGEST,\n"
                "    'exposed_development_test': {{'comparison': comparison, 'verdict': verdict}},\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors']), 'license': artifact_manifest['license']}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json' if not SUFFIX else f'outputs/{stem}_result{{SUFFIX}}.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Yes: the CPU build record of this revision reported 64 of 64 identical labels and a maximum probability difference of 0, and the adapter (34,645,456 bytes, 32 tensors for the head plus the last two blocks) carried the licence `cc-by-nc-sa-4.0` in its manifest.</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The held-out result is the fresh test (400 never-scored windows, locus-disjoint from every other window, scored "
        "under the pre-registered settings): the GC threshold, the frozen probe and the adapted model the pre-registered grid "
        "kept (the head plus the last two blocks) side by side, with the numbers of record in `docs/release-verification.md`, and an "
        "adapter that reloads prediction-for-prediction and carries the base licence in its manifest. That is the claim: the "
        "adaptation contract works end to end on a real, licence-traced labelled set behind a verified remote-code perimeter, "
        "and its numbers are read against two non-neural baselines and the frozen representation, on windows that played no "
        "part in choosing the configuration.\n\n"
        "The limits are specific. The fresh test is 400 windows from one seeded draw and the validation split that picks the "
        "epoch and the depth is 200, so neither gives a dispersion estimate. The gain over the frozen probe comes from one "
        "seed, and its size against seed-to-seed variation is **untested**. The task is one benchmark's convention: 251-base "
        "windows centred on EPD non-TATA promoters against windows that contain none. A gain here does not show that the "
        "classifier finds promoters in arbitrary sequence, handles TATA promoters or other species, or that its probabilities "
        "are calibrated (they are not). The head type was chosen in the build on an unrecorded split and is not re-tested. "
        "Every window, the fresh test included, is GRCh38 sequence that the model may have seen, unlabelled, during "
        "pre-training (DAT9), and that effect is not measured. The **exposed development test** numbers (adapted "
        "0.8425 / 0.6934 against the probe's 0.815 / 0.6315) are continuity only. The benchmark's own split is not "
        "locus-disjoint: 4,449 of its 9,034 test intervals overlap a training interval on the genome. Published scores on "
        "`human_nontata_promoters` therefore include that overlap, and they are not directly comparable with the fresh-test "
        "numbers here.\n\n"
        "Three things to carry to real data. **Baselines first:** fit the GC rule and the frozen probe on *your* labels before "
        "reading any fine-tuned number; if the probe already matches the fine-tuning, the representation was the answer. "
        "**Labelling convention:** the windows the head learns from define what \"promoter\" means; keep the window length "
        "and the negative-sampling rule fixed between training and use, or the number means nothing. **Leakage:** keep every "
        "sequence in one split (the contract de-duplicates by exact bases), and split by chromosome or locus when windows "
        "overlap. Exact-sequence disjointness is not enough: the seed-42 draw shared no sequence across splits, yet 113 of its "
        "400 test windows overlapped training windows on the genome with the same label. Here that is handled by the "
        "validation re-draw and the 251-base margin of the fresh test; a BYOD CSV gets neither.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot including the model code it executes, validate the demonstrated "
        "dataset contract on a licence-traced inline sample with exact-sequence-disjoint splits and a locus-disjoint fresh test, "
        "execute the inference contract, a pre-registered depth grid and a bounded fine-tuning, evaluate against two "
        "non-neural baselines and the frozen representation on that fresh test under the pre-registered settings, "
        "and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish "
        "benchmark superiority, classification quality on any other task or window convention, calibration of the "
        "probabilities, or production fitness; and nothing it produces may be used commercially under the weights' "
        "licence.\n\n"
        "**Optional experiments (exploratory; they do not affect the default path):** turn `RUN_SELECTION_GRID` off and set "
        "`TRAINED_LAYERS` to 0, 2 or 12 to see one configuration on its own; raise `EPOCHS` and watch the selection rule keep an "
        "early epoch while the training loss keeps falling; change `SEED` to see how much one seed moves the validation curve; or "
        "bring your own CSV through BYOD and read the two baselines before the adapted number. To run one, change the field in "
        "Section 7 and run Sections 7–9 again (for BYOD, Sections 4–9). `adapt` restores the pinned base blocks before it trains, "
        "and Sections 5, 6 and 8 call `pipe.reset_to_base()` when they need the frozen encoder, so every experiment starts from "
        "the pinned weights. **Every one of these runs is exploratory:** Section 7 prints `preregistered: False` with the "
        "settings that differ, the fresh test is not scored, and the report, result and adapter are written with an "
        "`_exploratory_<n>` suffix, so the pre-registered files are never replaced. Restoring every default and running "
        "Sections 7–9 again gives a pre-registered run, labelled a *reproduction* when it is not the session's first.\n\n"
        '## Troubleshooting\n\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch (weights or one of the two model-code files)** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again; never edit the model-code files.\n'
        '- **A verdict line says the adapted model did not beat the probe or a baseline** — on the default path that is a finding worth recording (the recorded run gained 0.06 MCC); on your data read the baselines first.\n'
        '- **CUDA out of memory in Section 7** — turn `RUN_SELECTION_GRID` off, set `TRAINED_LAYERS = 1` and run Sections 6–9 again. The run becomes **exploratory**: the fresh test is not scored and the outputs carry an `_exploratory_<n>` suffix.\n'
        '- **Section 4 printed `byod_refused`** — the message names the file or record and the rule. Fix the file (or set `USE_BYOD = False`), select Section 4 and choose **Runtime → Run after**; until then the later cells print `skipped`, and the files already in `outputs/` are from an earlier run.\n'
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one …"** — set `BYOD_PATH` to a CSV in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog is reported as a refusal.\n'
        '- **BYOD: a `load_byod_dataset` or `validate_dataset` refusal** — it names the row and the rule (alphabet, length 12..6,000, label 0/1, duplicate id, both labels needed for training).\n\n'
        '## Glossary\n\n'
        '- **6-mer token** — the unit the tokenizer reads: six bases at a time (251 bases → 47 tokens); `N` is tokenised base by base.\n'
        "- **Mean-pooled embedding** — the average of the encoder's token vectors: one 512-d representation per sequence, not a prediction.\n"
        '- **Masked prediction** — the pre-training objective: a distribution over 6-mers at a masked position; not a classifier.\n'
        "- **Remote code** — the checkpoint's own `modeling_esm.py` and `esm_config.py`, digest-verified before they are imported.\n"
        '- **GC-threshold baseline** — a single cut on GC fraction fitted on the training split; reads composition only.\n'
        '- **Frozen probe** — a logistic regression on the frozen embeddings: what the pretrained representation alone knows.\n'
        '- **Accuracy / MCC** — share correct; the Matthews correlation, 0 for any constant or chance predictor and symmetric in the two classes.\n'
        '- **Epoch / validation selection** — one pass over the training windows; keeping the epoch with the highest validation MCC.\n'
        '- **Fresh test** — 400 windows never used for training, selection or any earlier score, at least 251 bases from every other window on the genome; scored only under the pre-registered settings.\n'
        '- **Exposed development test** — the old 400 test windows: seen while the build chose the head and depth, and partly overlapping training windows on the genome; reported for continuity only.\n'
        '- **Locus-disjoint** — no window of one split overlaps a window of another on the genome; stricter than exact-sequence disjointness.\n'
        '- **Pre-registered / exploratory** — a run whose every setting matches `docs/nucleotide-transformer-preregistration.md` / any other run (validation and the exposed development test only).\n'
        '- **Adapter / reload parity** — the head and trained blocks only, overlaid on the pinned base; the reloaded model gives the same labels.\n'
        "- **CC BY-NC-SA 4.0** — the weights' licence: attribution, non-commercial, share-alike; the adapter inherits it.\n"
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: your labelled sequences through the same cells.\n\n'
        '## Conclusion (your notes)\n\nOptional — fill in from **your** run:\n\n'
        '- Majority ___, GC rule ___, frozen probe ___, adapted ___ (test MCC); the verdict was ___.\n'
        '- The kept epoch was ___; the later epochs were ___ on validation.\n'
        '- One window the GC rule got wrong and the classifier got right: ___.\n'
        '- One reason not to trust this gain on my own sequences yet: ___.\n\n'
        "## References\n\n"
        "[1] H. Dalla-Torre *et al.*, \"Nucleotide Transformer: building and evaluating robust foundation models for human "
        "genomics,\" *Nature Methods*, vol. 22, pp. 287–297, 2025, doi: 10.1038/s41592-024-02523-z.\n\n"
        "[2] K. Grešová, V. Martinek, D. Čechák, P. Šimeček, and P. Alexiou, \"Genomic benchmarks: a collection of datasets "
        "for genomic sequence classification,\" *BMC Genomic Data*, vol. 24, art. 25, 2023, doi: 10.1186/s12863-023-01123-8. "
        "[Online]. Available: https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks (Apache-2.0; human non-TATA promoters "
        "from the Eukaryotic Promoter Database)\n\n"
        "[3] InstaDeep, \"nucleotide-transformer-v2-50m-multi-species,\" Hugging Face model repository. [Online]. Available: "
        "https://huggingface.co/{MODEL_ID}\n\n"
        "[4] InstaDeep, \"multi_species_genomes\" (the 850-genome pre-training list; `urls.txt` entry 784 is GRCh38.p14), "
        "Hugging Face dataset repository. [Online]. Available: https://huggingface.co/datasets/InstaDeepAI/multi_species_genomes\n\n"
        "[5] InstaDeep, \"nucleotide-transformer,\" GitHub repository (CC BY-NC-SA 4.0). [Online]. Available: "
        "https://github.com/instadeepai/nucleotide-transformer\n\n"
        "[6] Ensembl, \"Ensembl REST API: sequence/region (GRCh38),\" [Online]. Available: https://rest.ensembl.org/\n\n"
        "[7] K. Valcorza, \"nucleotide-transformer-genomics-pipeline: README, model card, weight provenance and the "
        "pre-registration,\" GitHub repository. [Online]. Available: "
        "https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/MODEL_CARD.md, "
        "https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/README.md, "
        "https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/docs/WEIGHTS.md and "
        "docs/nucleotide-transformer-preregistration.md\n\n"
        "[8] DIMER, \"Notebook Specification 2.5 and Model Card Specification 1.1,\" fleet specifications in the ml-worker "
        "repository."
    ),
}
