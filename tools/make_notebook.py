#!/usr/bin/env python3
"""Emit the standalone Nucleotide Transformer tutorial notebook.

This is **not** the fleet's `build_notebook.py` generator: this repository ships no DIMER pipeline
package (the row is on licence HOLD, see `MODEL_CARD.md`), so the tutorial's helper code lives in
the notebook's own cells rather than being carried from `src/`. This script exists so that the
notebook is reproducible from reviewable Python and so that the inline manifest and dependency pins
cannot drift from `weights/.../dimer-base-manifest.json` and `requirements.txt`.

Usage (from the repository root):
    python tools/make_notebook.py            # write tutorials/<notebook>
    python tools/make_notebook.py --check    # exit 1 if the committed notebook differs
"""

# ruff: noqa: E501  -- learner-facing prose is kept on single lines so the rendered markdown stays readable
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "nucleotide_transformer_colab.ipynb"
MODEL_KEY = "nt-v2-50m-multi-species"
MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
MODEL_REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"
MODEL_LICENSE = "cc-by-nc-sa-4.0"
NOTEBOOK_SPEC = "2.0"
PROFILE = "TASK-INFERENCE"
MODE = "GUIDED"
GENERATOR = "make_notebook.py/1"


def pins() -> list[str]:
    lines = [ln.strip() for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    values = [ln for ln in lines if ln and not ln.startswith("#")]
    unpinned = [p for p in values if "==" not in p]
    if unpinned:
        raise SystemExit(f"unpinned runtime dependency (ENV2): {unpinned}")
    return values


def manifest() -> dict[str, Any]:
    return json.loads((ROOT / "weights" / MODEL_KEY / "dimer-base-manifest.json").read_text(encoding="utf-8"))


def _md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": "", "metadata": {}, "source": source.rstrip("\n")}


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "",
        "metadata": {},
        "outputs": [],
        "source": source.rstrip("\n"),
    }


BADGES = " ".join(
    [
        "[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline)",
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/tutorials/nucleotide_transformer_colab.ipynb)",
        "[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-InstaDeepAI%2Fnucleotide--transformer--v2--50m--multi--species-ffcc4d?style=flat)](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)",
        "[![Upstream GitHub](https://img.shields.io/badge/Upstream-instadeepai%2Fnucleotide--transformer-181717?style=flat&logo=github&logoColor=white)](https://github.com/instadeepai/nucleotide-transformer)",
        "[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)",
    ]
)


def build() -> dict[str, Any]:
    mf = manifest()
    pin_list = pins()
    total_mb = mf["totalBytes"] / 1e6
    cells: list[dict[str, Any]] = []

    def add(cell: dict[str, Any]) -> None:
        cell["id"] = f"nt-{len(cells):02d}"
        cells.append(cell)

    add(
        _md(
            "# Nucleotide Transformer v2 50M Multi-Species — DNA sequence representations (standalone tutorial)\n\n"
            + BADGES
            + "\n\n"
            f"**Profile:** `{PROFILE}`  \n"
            f"**Mode:** `{MODE}`  \n"
            f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)  \n"
            "**Capability:** DNA sequence embeddings and masked-token prediction from a pinned genomic language model\n\n"
            "> [!IMPORTANT]\n"
            "> **Non-commercial model licence.** The Nucleotide Transformer v2 50M Multi-Species model "
            "(`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`) is developed by InstaDeep, NVIDIA and TUM and distributed "
            "under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) licence. The licence "
            "permits copying, redistribution and adaptation for non-commercial purposes, subject to its attribution and ShareAlike "
            "requirements. **Commercial use is not permitted under this licence.** Running this notebook downloads those weights to "
            "your own runtime; you are responsible for ensuring your use is non-commercial and otherwise complies with the licence.\n\n"
            "> [!WARNING]\n"
            "> **This notebook executes remote model code.** This checkpoint ships its own `modeling_esm.py` and `esm_config.py` and "
            "cannot be loaded by the native Transformers ESM classes — its feed-forward block is a bias-free SwiGLU that the native "
            "implementation does not have. The notebook therefore loads the model with `trust_remote_code=True`, pinned to one "
            "immutable revision whose Python files are digest-verified in Section 3 before they are imported. Section 4 explains "
            "exactly what that means. Do not run this notebook on data or in an environment where executing third-party Python "
            "would be unacceptable.\n\n"
            "**This notebook is standalone.** It carries its own helper code, the pinned model identity and the per-file SHA-256 "
            f"manifest in Section 3, and the exact runtime pins in Section 1, so it keeps working after export even if the repository "
            f"changes or disappears. Its only external dependencies are the pinned PyPI distributions and the Hugging Face Hub at the "
            f"immutable revision `{MODEL_REVISION}` (~{total_mb:.0f} MB, digest-verified before loading). It was emitted by "
            f"`tools/make_notebook.py` ({GENERATOR}); edit the repository and regenerate rather than editing cells.\n\n"
            "**Run all:** Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and "
            "digest-verifies the pinned snapshot (8 files, including the two Python files it will execute), loads the tokenizer "
            "without remote code and the model with it, shows how DNA is tokenized into 6-mers, generates a deterministic 48-sequence "
            "sample in code, extracts mean-pooled sequence embeddings, fits a nearest-centroid probe on a held-out split against two "
            "trivial baselines, runs one masked-token prediction, and exports embeddings, metrics and provenance. The default path "
            "needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
            "(NOTEBOOK_SPEC 2.0 §5). It takes a couple of minutes, almost all of it the model download.\n\n"
            "**Bring Your Own Data:** After the sample workflow completes, set `USE_BYOD = True` in Section 6 and re-run from that "
            "cell to supply your own DNA sequences as FASTA or as a CSV with `id,sequence,label` columns. They pass through the same "
            "validation, embedding, probe and export cells as the sample. The expected format, the alphabet and the ceilings are "
            "stated in the Prerequisites and in Section 6, and uploaded files stay inside this runtime. BYOD is optional and never "
            "part of the default path.\n\n"
            "The Nucleotide Transformer family are transformer language models pretrained on DNA. This checkpoint — the 50M-parameter "
            "multi-species model — was pretrained on 850 genomes spanning model and non-model organisms. DNA is tokenized as "
            "**6-mers**: every six consecutive `A`/`C`/`G`/`T` bases become one token, and any window containing another character "
            "(such as `N`) falls back to one token per base. The model returns a 512-dimensional hidden state per token, so a "
            "sequence becomes a matrix of token representations rather than a prediction. This tutorial pools those into one vector "
            "per sequence and then asks the only honest question about an embedding: does a downstream task read it? The sample task "
            "is deliberately built so that nucleotide composition cannot answer it — the two classes contain **exactly the same "
            "multiset of bases**, differing only in whether a fixed 6-mer motif is present or its letters are permuted.\n\n"
            "**Learning objectives:** install the pinned runtime; stage and digest-verify an immutable snapshot including the code it "
            "will execute; read what `trust_remote_code=True` actually buys and costs here; see DNA turned into 6-mer tokens and "
            "watch the fallback to single bases; extract mean-pooled sequence embeddings; evaluate them with a nearest-centroid probe "
            "against a GC-content baseline and a majority baseline on a held-out split; read one masked-token prediction; and export "
            "embeddings, metrics and provenance.\n\n"
            "**This notebook does not demonstrate:** fine-tuning or LoRA adaptation, variant-effect prediction, any of the published "
            "downstream genomics benchmarks, the larger Nucleotide Transformer checkpoints, or DIMER packaging — this row is on "
            "licence HOLD and ships no DIMER pipeline module."
        )
    )

    add(
        _md(
            "## Prerequisites\n\n"
            "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is enough; the whole default path "
            "is about a second of model time once the weights are downloaded. CUDA is used automatically when present.\n"
            "- **Knowledge:** DNA as a string over `A`/`C`/`G`/`T`, what an embedding is, and what a held-out split protects against.\n"
            "- **Licence:** the model weights are **CC BY-NC-SA 4.0 (non-commercial)**. This is your decision to make before you run "
            "the notebook, not a checkbox inside it.\n"
            "- **Remote code:** the default path executes the checkpoint's own `modeling_esm.py` and `esm_config.py` after verifying "
            "their SHA-256 against the inline manifest. See Section 4.\n"
            "- **Data contract:** sequences are uppercase strings over `A`, `C`, `G`, `T` and `N`; 24 to 6,000 bases each (6,000 bases "
            "is 1,000 6-mer tokens, the length this checkpoint was trained at); at most 64 sequences per call. BYOD accepts FASTA "
            "(`>id` headers) or a CSV with `id,sequence,label`.\n"
            "- **Privacy:** do not upload confidential or restricted data — patient-derived or unpublished sequences included — to a "
            "hosted runtime unless you are authorized to process it there. The default path uploads nothing.\n"
            f"- **External access:** the Hugging Face Hub only, to fetch the pinned `{MODEL_ID}` snapshot (~{total_mb:.0f} MB) at "
            f"revision `{MODEL_REVISION[:12]}…`. No GitHub access and no credentials are required; nothing is installed from this repository."
        )
    )

    pins_literal = "PINS = [\n" + "".join(f"    {p!r},\n" for p in pin_list) + "]"
    add(
        _md(
            "## 1. Install the pinned runtime\n\n"
            "The dependency set is pinned exactly (the same `==` pins as the repository's `requirements.txt`) and installed directly — "
            "there is no repository clone and no package install. If a pin replaces a distribution this runtime has already imported, "
            "the cell stops with a restart instruction rather than continuing with mixed versions. Look for a dictionary reporting the "
            "notebook's source revision, Python, `torch` and `transformers` versions, and whether CUDA is available."
        )
    )
    add(
        _code(
            "import importlib\n"
            "import importlib.metadata\n"
            "import os\n"
            "import platform\n"
            "import subprocess\n"
            "import sys\n\n"
            f"{pins_literal}\n"
            "NOTEBOOK_SOURCE = {\n"
            "    'repository': 'nucleotide-transformer-genomics-pipeline',\n"
            "    'repository_revision': '062e7a43fe01bbb3b8856d6cdd296e6bacfcd88d',\n"
            f"    'generator': {GENERATOR!r},\n"
            f"    'notebook_spec': {NOTEBOOK_SPEC!r},\n"
            f"    'notebook_profile': {PROFILE!r},\n"
            "}\n"
            "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'\n\n"
            "def _installed_version(distribution):\n"
            "    try:\n"
            "        return importlib.metadata.version(distribution)\n"
            "    except importlib.metadata.PackageNotFoundError:\n"
            "        return None\n\n"
            "if not SKIP_INSTALL:\n"
            "    _module_dists = importlib.metadata.packages_distributions()\n"
            "    _loaded = sorted({d for m in list(sys.modules) for d in _module_dists.get(m.partition('.')[0], ())})\n"
            "    loaded = {distribution: _installed_version(distribution) for distribution in _loaded}\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PINS], check=True)\n"
            "    importlib.invalidate_caches()\n"
            "    stale = []\n"
            "    for distribution, before in loaded.items():\n"
            "        installed = _installed_version(distribution)\n"
            "        if before is not None and before != installed:\n"
            "            stale.append(f'{distribution}: loaded={before}, installed={installed}')\n"
            "    if stale:\n"
            "        raise RuntimeError('Core dependencies changed while older modules were loaded: ' + '; '.join(stale) + '. Restart the runtime, then rerun from the top.')\n\n"
            "import torch, transformers\n"
            "print({'notebook_source': NOTEBOOK_SOURCE, 'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.cuda.is_available()})"
        )
    )

    add(
        _md(
            "## 2. The licence, before anything is downloaded\n\n"
            "This model is not permissively licensed, and that is the first thing to establish because the next cell fetches its "
            "weights onto your runtime.\n\n"
            "- **Weights licence:** CC BY-NC-SA 4.0 — Attribution, **NonCommercial**, ShareAlike. Copying, redistribution and "
            "adaptation are permitted for non-commercial purposes; **commercial use is not permitted under this licence**. If you "
            "adapt the weights and distribute the adaptation, the ShareAlike condition requires you to distribute it under the same "
            "or a compatible licence.\n"
            "- **What NonCommercial means here:** the test is whether the use is primarily intended for or directed toward commercial "
            "advantage or monetary compensation. It is a question about *your use*, not about your organisation's type — being a "
            "university, a government body or a non-profit does not by itself make a use non-commercial, and hosted inference is not "
            "categorically prohibited either. If your intended use might fail that test, obtain separate permission from the rights "
            "holders rather than relying on this notebook.\n"
            "- **Supporting code:** the checkpoint's `modeling_esm.py` and `esm_config.py` carry Meta and Hugging Face copyright "
            "headers and derive from the Apache-2.0 Transformers ESM implementation. A permissive licence on that code does **not** "
            "remove the NonCommercial condition from the weights.\n"
            "- **This repository** adds documentation and this notebook under its own licence, redistributes no weights, and grants "
            "no rights beyond those the upstream licensor gives you.\n\n"
            "The next cell prints the licence string recorded in the snapshot manifest so the value you rely on is the one that was "
            "pinned, not one copied into prose."
        )
    )

    manifest_literal = json.dumps(mf, indent=2, ensure_ascii=False)
    add(
        _md(
            "## 3. Pin, stage and verify the snapshot\n\n"
            f"The identity is carried twice — the constants below and the `{len(mf['files'])}`-file manifest (paths, byte sizes, "
            "SHA-256) — and the cell asserts they agree before fetching anything. `stage_missing_files` then fetches exactly the "
            f"entries that are absent, from the Hugging Face Hub **at revision `{MODEL_REVISION[:12]}…`** (never `main`), and "
            "`verify_snapshot` re-hashes every file and raises on the first size or digest mismatch.\n\n"
            "Two of those eight files are Python — `modeling_esm.py` and `esm_config.py` — and they are in the manifest for the same "
            "reason the weights are: Section 5 will execute them, so their bytes are pinned and checked first. There is no fallback "
            "to a different download and no unpinned revision anywhere in this notebook."
        )
    )
    add(
        _code(
            "import hashlib\n"
            "import json\n"
            "from pathlib import Path\n\n"
            f"MODEL_ID = {MODEL_ID!r}\n"
            f"MODEL_REVISION = {MODEL_REVISION!r}\n"
            f"MODEL_KEY = {MODEL_KEY!r}\n"
            f"MODEL_LICENSE = {MODEL_LICENSE!r}\n"
            "MANIFEST_NAME = 'dimer-base-manifest.json'\n"
            f"MANIFEST = {manifest_literal}\n\n"
            "if (MANIFEST['modelId'], MANIFEST['revision']) != (MODEL_ID, MODEL_REVISION):\n"
            "    raise RuntimeError('inline manifest does not name the pinned identity; the notebook was not regenerated after a change')\n\n"
            "WEIGHTS_DIR = Path.cwd() / 'weights' / MODEL_KEY\n"
            "WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "with open(WEIGHTS_DIR / MANIFEST_NAME, 'w', encoding='utf-8') as handle:\n"
            "    json.dump(MANIFEST, handle, indent=2)\n\n"
            "def stage_missing_files(root, allow_download=True):\n"
            '    """Fetch only the manifest entries that are absent, only at the pinned revision."""\n'
            "    missing = [entry['path'] for entry in MANIFEST['files'] if not (root / entry['path']).is_file()]\n"
            "    if missing and not allow_download:\n"
            "        raise FileNotFoundError(f'snapshot at {root} is missing {missing}')\n"
            "    if missing:\n"
            "        from huggingface_hub import hf_hub_download\n"
            "        for relative_path in missing:\n"
            "            hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))\n"
            "    return missing\n\n"
            "def verify_snapshot(root):\n"
            '    """Re-hash every manifest entry; raise naming the first mismatch."""\n'
            "    for entry in MANIFEST['files']:\n"
            "        path = root / entry['path']\n"
            "        if not path.is_file():\n"
            "            raise FileNotFoundError(f'snapshot file missing: {path}')\n"
            "        size = path.stat().st_size\n"
            "        if size != entry['bytes']:\n"
            "            raise ValueError(f\"{entry['path']}: size {size} != manifest {entry['bytes']}\")\n"
            "        digest = hashlib.sha256()\n"
            "        with open(path, 'rb') as handle:\n"
            "            for chunk in iter(lambda: handle.read(1 << 20), b''):\n"
            "                digest.update(chunk)\n"
            "        if digest.hexdigest() != entry['sha256']:\n"
            "            raise ValueError(f\"{entry['path']}: sha256 mismatch against the manifest\")\n"
            "    return MANIFEST\n\n"
            "print({'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'license': MODEL_LICENSE, 'files': len(MANIFEST['files']), 'total_bytes': MANIFEST['totalBytes']})\n"
            "fetched = stage_missing_files(WEIGHTS_DIR)\n"
            "print({'weights_dir': str(WEIGHTS_DIR), 'fetched': len(fetched)})\n"
            "verify_snapshot(WEIGHTS_DIR)\n"
            "print({'verified_files': len(MANIFEST['files']), 'python_files_verified': [e['path'] for e in MANIFEST['files'] if e['path'].endswith('.py')]})"
        )
    )

    add(
        _md(
            "## 4. What `trust_remote_code=True` means here\n\n"
            "Most checkpoints in this fleet load with `trust_remote_code=False`, and that is the default everywhere else. This one "
            "cannot, and it is worth being precise about why rather than flipping the flag and moving on.\n\n"
            "The checkpoint's `config.json` has no `model_type` and instead carries an `auto_map` pointing at its own "
            "`esm_config.EsmConfig` and `modeling_esm.EsmForMaskedLM`. That alone would only be an inconvenience — but the code is "
            "not cosmetic. Its feed-forward block projects to twice the intermediate width, splits the result in half and gates one "
            "half with SiLU (a **SwiGLU**), with `bias=False` throughout; the native Transformers ESM implementation uses a plain "
            "GELU feed-forward **with** bias. Loading these weights into the native class would not reproduce this model. So remote "
            "code here is a genuine architectural requirement, not a workaround for a loading error.\n\n"
            "What the notebook does about it: the two Python files are pinned to one immutable revision and their SHA-256 verified in "
            "Section 3 **before** this cell imports them, so the code that runs is the code that was reviewed, and a changed upstream "
            "file fails loudly rather than executing. What it does **not** do: claim that verification makes third-party code safe. "
            "Executing it is a trust decision about InstaDeep's published code, and it is the reason this row ships no DIMER pipeline "
            "module — an unresolved remote-code dependency is an explicit gate in the fleet's asset rules.\n\n"
            "The cell first shows the native loader refusing the checkpoint, so the requirement is visible rather than asserted, then "
            "loads the tokenizer (native, no remote code) and the model (remote code, pinned)."
        )
    )
    add(
        _code(
            "from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer\n\n"
            "try:\n"
            "    AutoModel.from_pretrained(str(WEIGHTS_DIR), local_files_only=True, trust_remote_code=False)\n"
            "    print({'native_load': 'unexpectedly succeeded'})\n"
            "except Exception as exc:\n"
            "    print({'native_load_refused': type(exc).__name__, 'reason': str(exc).split('.')[0][:160]})\n\n"
            "tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR), local_files_only=True, trust_remote_code=False)\n"
            "model = AutoModelForMaskedLM.from_pretrained(str(WEIGHTS_DIR), local_files_only=True, trust_remote_code=True)\n"
            "DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'\n"
            "model = model.to(DEVICE).eval()\n"
            "print({\n"
            "    'tokenizer_class': type(tokenizer).__name__,\n"
            "    'tokenizer_remote_code': False,\n"
            "    'model_class': type(model).__name__,\n"
            "    'model_remote_code': True,\n"
            "    'parameters': sum(p.numel() for p in model.parameters()),\n"
            "    'state_dict_elements': sum(v.numel() for v in model.state_dict().values()),\n"
            "    'hidden_size': model.config.hidden_size,\n"
            "    'layers': model.config.num_hidden_layers,\n"
            "    'device': DEVICE,\n"
            "})"
        )
    )

    add(
        _md(
            "## 5. How DNA becomes tokens\n\n"
            "The tokenizer is the native `EsmTokenizer` (no remote code) reading the checkpoint's own `vocab.txt`. That vocabulary has "
            "4,107 entries: six special tokens, all 4,096 six-letter combinations of `A`/`C`/`G`/`T`, and the five single bases "
            "`A`, `T`, `C`, `G`, `N`. Sequences are consumed left to right in windows of six: a window of four standard bases becomes "
            "one 6-mer token, and a window containing anything else — an `N`, or a trailing remainder shorter than six — falls back to "
            "one token per base.\n\n"
            "Watch that fallback in the demonstration below: the first eighteen bases become three clean 6-mer tokens, and everything "
            "after the `N` is emitted base by base. It matters in practice, because a single ambiguity code costs you six tokens' worth "
            "of context and shifts every 6-mer boundary after it.\n\n"
            "Two numbers the rest of the notebook depends on: `MAX_BASES = 6000`, which is 1,000 tokens — the sequence length this "
            "checkpoint was **trained** at, as stated in the upstream model card — and `MAX_SEQUENCES = 64` per call. The tokenizer's "
            "own `model_max_length` is 2,048 tokens and the position embeddings allow 2,050, so longer inputs will run; they are simply "
            "beyond the trained regime, which is why this notebook stops at the trained length rather than at the architectural one."
        )
    )
    add(
        _code(
            "MAX_BASES = 6000          # 1,000 six-mer tokens: the length this checkpoint was trained at\n"
            "MAX_SEQUENCES = 64        # per call, in this tutorial\n"
            "ALPHABET = set('ACGTN')\n\n"
            "def validate_sequences(sequences, names=None):\n"
            '    """Raise naming the first violated rule; return (sequences, ids) and an input manifest."""\n'
            "    if isinstance(sequences, str) or not isinstance(sequences, (list, tuple)):\n"
            "        raise TypeError('sequences must be a list of DNA strings')\n"
            "    if not 1 <= len(sequences) <= MAX_SEQUENCES:\n"
            "        raise ValueError(f'sequences must hold 1..{MAX_SEQUENCES} items, got {len(sequences)}')\n"
            "    checked = []\n"
            "    for i, seq in enumerate(sequences):\n"
            "        if not isinstance(seq, str):\n"
            "            raise TypeError(f'sequences[{i}] must be str, got {type(seq).__name__}')\n"
            "        if len(seq) < 24:\n"
            "            raise ValueError(f'sequences[{i}] has {len(seq)} bases; at least 24 (four 6-mers) are required')\n"
            "        if len(seq) > MAX_BASES:\n"
            "            raise ValueError(f'sequences[{i}] has {len(seq)} bases; the trained ceiling is {MAX_BASES}')\n"
            "        bad = sorted(set(seq) - ALPHABET)\n"
            "        if bad:\n"
            "            raise ValueError(f'sequences[{i}] contains characters outside A/C/G/T/N: {bad} (uppercase your sequence and strip gaps first)')\n"
            "        checked.append(seq)\n"
            "    ids = list(names) if names is not None else [f'seq-{i}' for i in range(len(checked))]\n"
            "    if len(ids) != len(checked) or len(set(ids)) != len(ids):\n"
            "        raise ValueError('names must provide one unique id per sequence')\n"
            "    manifest = {\n"
            "        'n_sequences': len(checked),\n"
            "        'bases': {'min': min(len(s) for s in checked), 'max': max(len(s) for s in checked)},\n"
            "        'ambiguous_bases': sum(s.count('N') for s in checked),\n"
            "        'ceilings': {'max_bases': MAX_BASES, 'max_sequences': MAX_SEQUENCES, 'alphabet': 'ACGTN'},\n"
            "        'verdict': 'accepted',\n"
            "    }\n"
            "    return checked, ids, manifest\n\n"
            "demo = 'ATTCCGATTCCGATTCCGACGTACGTNACGT'\n"
            "demo_ids = tokenizer(demo)['input_ids']\n"
            "print({'sequence': demo, 'bases': len(demo)})\n"
            "print({'tokens': tokenizer.convert_ids_to_tokens(demo_ids), 'n_tokens': len(demo_ids)})\n"
            "print({'vocab_size': tokenizer.vocab_size, 'model_max_length': tokenizer.model_max_length, 'trained_length_tokens': MAX_BASES // 6})\n"
            "for probe in ['acgtacgtacgtacgtacgtacgt', 'ACGT', 'ACGTACGTACGTACGT-ACGTACGT']:\n"
            "    try:\n"
            "        validate_sequences([probe])\n"
            "        print({'probe': probe[:24], 'verdict': 'accepted'})\n"
            "    except (TypeError, ValueError) as exc:\n"
            "        print({'probe': probe[:24], 'rejected': str(exc)[:96]})"
        )
    )

    add(
        _md(
            "## 6. Sample sequences\n\n"
            "The default sample is generated in code with a fixed seed: 24 pairs of 300-base sequences. Each pair starts from the same "
            "random background; in the `motif` member a fixed 6-mer (`ACGTAC`) is written into six positions, and in the `shuffled` "
            "member the **same six letters in a different order** (`GTCAAC`) is written into the same positions. The consequence is the "
            "point of this tutorial: the two classes have an identical multiset of bases — identical GC content, identical length, "
            "identical everything a composition statistic can see — and differ only in the *order* of six letters at six places.\n\n"
            "That is what makes the probe in Section 8 meaningful. A classifier that only counts bases cannot do better than chance "
            "here, by construction, and the cell asserts that composition equality rather than asking you to take it on trust.\n\n"
            "To use your own sequences, set `USE_BYOD = True` and re-run from this cell. FASTA (`>id` headers) and CSV "
            "(`id,sequence,label`) are both accepted; labels are optional for embedding but required for the probe."
        )
    )
    add(
        _code(
            "import csv\n"
            "import io\n"
            "import random\n\n"
            'USE_BYOD = False  # @param {type:"boolean"}\n'
            'SEED = 42  # @param {type:"integer"}\n'
            "SAMPLE_SEED = 20260918\n"
            "SAMPLE_SIZE = 48\n"
            "MOTIF = 'ACGTAC'\n"
            "DECOY = 'GTCAAC'   # the same six letters, permuted\n"
            "SEQ_LENGTH = 300\n"
            "MOTIF_COPIES = 6\n\n"
            "os.makedirs('outputs', exist_ok=True)\n\n"
            "def generate_sample(seed=SAMPLE_SEED, size=SAMPLE_SIZE):\n"
            '    """Pairs of sequences with an identical base multiset; only the motif\'s letter order differs."""\n'
            "    rng = random.Random(seed)\n"
            "    records = []\n"
            "    for i in range(size // 2):\n"
            "        background = [rng.choice('ACGT') for _ in range(SEQ_LENGTH)]\n"
            "        slots = sorted(rng.sample(range(0, SEQ_LENGTH - len(MOTIF), len(MOTIF) * 2), MOTIF_COPIES))\n"
            "        for label, word in (('motif', MOTIF), ('shuffled', DECOY)):\n"
            "            seq = list(background)\n"
            "            for start in slots:\n"
            "                seq[start:start + len(word)] = list(word)\n"
            "            records.append({'id': f'{label}-{i:03d}', 'sequence': ''.join(seq), 'label': label})\n"
            "    return records\n\n"
            "def parse_fasta(text):\n"
            "    records, name, chunks = [], None, []\n"
            "    for line in text.splitlines():\n"
            "        line = line.strip()\n"
            "        if line.startswith('>'):\n"
            "            if name is not None:\n"
            "                records.append({'id': name, 'sequence': ''.join(chunks).upper(), 'label': ''})\n"
            "            name, chunks = line[1:].split()[0] or f'seq-{len(records)}', []\n"
            "        elif line:\n"
            "            chunks.append(line)\n"
            "    if name is not None:\n"
            "        records.append({'id': name, 'sequence': ''.join(chunks).upper(), 'label': ''})\n"
            "    if not records:\n"
            "        raise ValueError('no FASTA records found: expected at least one \">id\" header followed by sequence lines')\n"
            "    return records\n\n"
            "def gc_fraction(seq):\n"
            "    return sum(ch in 'GC' for ch in seq) / len(seq)\n\n"
            "if USE_BYOD:\n"
            "    from google.colab import files\n"
            "    uploaded = files.upload()\n"
            "    file_name, payload = next(iter(uploaded.items()))\n"
            "    text = payload.decode('utf-8-sig')\n"
            "    if file_name.lower().endswith('.csv'):\n"
            "        reader = csv.DictReader(io.StringIO(text))\n"
            "        header = [h.strip() for h in (reader.fieldnames or [])]\n"
            "        if 'id' not in header or 'sequence' not in header:\n"
            "            raise ValueError(f\"CSV header {header} must contain at least 'id' and 'sequence' columns\")\n"
            "        records = [{'id': r['id'].strip(), 'sequence': (r['sequence'] or '').strip().upper(), 'label': (r.get('label') or '').strip()} for r in reader]\n"
            "    else:\n"
            "        records = parse_fasta(text)\n"
            "    data_source = 'BYOD (' + file_name + ')'\n"
            "else:\n"
            "    records = generate_sample()\n"
            "    data_source = f'synthetic composition-matched motif dataset (seed {SAMPLE_SEED}, {SAMPLE_SIZE} sequences)'\n\n"
            "sequences, ids, input_manifest = validate_sequences([r['sequence'] for r in records], [r['id'] for r in records])\n"
            "labels = [r['label'] for r in records]\n"
            "print({'data_source': data_source, **input_manifest})\n"
            "print({'classes': sorted({l for l in labels if l}), 'labelled': sum(1 for l in labels if l)})\n\n"
            "if not USE_BYOD:\n"
            "    pairs = [(records[2 * i], records[2 * i + 1]) for i in range(len(records) // 2)]\n"
            "    same_multiset = all(sorted(a['sequence']) == sorted(b['sequence']) for a, b in pairs)\n"
            "    same_gc = all(abs(gc_fraction(a['sequence']) - gc_fraction(b['sequence'])) < 1e-12 for a, b in pairs)\n"
            "    print({'identical_base_multiset_within_pair': same_multiset, 'identical_gc_within_pair': same_gc})\n"
            "    print({'motif_occurrences': {'motif': records[0]['sequence'].count(MOTIF), 'shuffled': records[1]['sequence'].count(MOTIF)}})\n"
            "    assert same_multiset and same_gc, 'the sample is only meaningful if the two classes share their composition'\n"
            "print({'example_id': records[0]['id'], 'label': records[0]['label'], 'first_60_bases': records[0]['sequence'][:60]})"
        )
    )

    add(
        _md(
            "## 7. Sequence embeddings\n\n"
            "The model returns one 512-dimensional hidden state per token. To get one vector per sequence the notebook takes the mean "
            "of the last hidden layer over the real (non-padding) tokens — the same pooling the upstream model card demonstrates.\n\n"
            "Be clear about what this is: a **representation**, not a prediction. It carries no label and no metric of its own. The "
            "number printed below tells you the shape of the output and nothing about its quality; Section 8 is where a downstream "
            "task gives it meaning. Embeddings and their ids are written to `outputs/nucleotide_transformer_embeddings.csv`."
        )
    )
    add(
        _code(
            "import time\n\n"
            'BATCH_SIZE = 8  # @param {type:"integer"}\n\n'
            "def embed(sequence_list, batch_size=BATCH_SIZE):\n"
            '    """Mean-pooled last-hidden-state vector per sequence (padding excluded)."""\n'
            "    vectors = []\n"
            "    for start in range(0, len(sequence_list), batch_size):\n"
            "        batch = tokenizer(sequence_list[start:start + batch_size], return_tensors='pt', padding=True)\n"
            "        batch = {k: v.to(DEVICE) for k, v in batch.items()}\n"
            "        with torch.no_grad():\n"
            "            hidden = model(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)['hidden_states'][-1]\n"
            "        mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)\n"
            "        pooled = (hidden * mask).sum(1) / mask.sum(1)\n"
            "        vectors.extend(pooled.float().cpu().tolist())\n"
            "    return vectors\n\n"
            "started = time.perf_counter()\n"
            "vectors = embed(sequences)\n"
            "embed_seconds = round(time.perf_counter() - started, 2)\n"
            "tokens_per_sequence = [len(tokenizer(s)['input_ids']) for s in sequences]\n"
            "print({'n_sequences': len(vectors), 'dimension': len(vectors[0]), 'pooling': 'mean of the last hidden state over non-padding tokens', 'unit': 'one vector per sequence; a representation, not a prediction', 'tokens_per_sequence': {'min': min(tokens_per_sequence), 'max': max(tokens_per_sequence)}, 'seconds': embed_seconds, 'device': DEVICE})\n\n"
            "with open('outputs/nucleotide_transformer_embeddings.csv', 'w', encoding='utf-8', newline='') as handle:\n"
            "    writer = csv.writer(handle)\n"
            "    writer.writerow(['id', 'label'] + [f'dim_{k}' for k in range(len(vectors[0]))])\n"
            "    for rid, label, vector in zip(ids, labels, vectors):\n"
            "        writer.writerow([rid, label] + [f'{x:.6f}' for x in vector])\n"
            "print('wrote outputs/nucleotide_transformer_embeddings.csv')"
        )
    )

    add(
        _md(
            "## 8. Does a downstream task read the embedding?\n\n"
            "An embedding with no task attached cannot be scored, so this section attaches the smallest honest one: a "
            "**nearest-centroid probe**. The labelled sequences are split in half, stratified by class and seeded; the mean embedding "
            "of each class is computed on the training half only; and every held-out sequence is assigned to the nearer centroid by "
            "cosine similarity. Nothing about the model is trained — this measures the representation, not a fine-tuned classifier.\n\n"
            "Two baselines sit next to it on the same held-out split. **Majority class** is 0.5 on a balanced split. **GC content** "
            "fits one threshold on the training half and applies it to the held-out half; because the two classes share their base "
            "composition exactly, it should land at chance, and if it does not, suspect the sample rather than the model.\n\n"
            "These are tutorial sample-sanity numbers from a single seeded split of 24 held-out synthetic sequences, with no dispersion "
            "estimate — not a genomics benchmark, and not a claim about any published task. Read the cosine margin too: it is reported "
            "because a correct assignment with a margin in the third decimal place is a much weaker statement than the accuracy alone "
            "suggests."
        )
    )
    add(
        _code(
            "import math\n\n"
            "labelled = [(i, labels[i]) for i in range(len(ids)) if labels[i]]\n"
            "classes = sorted({label for _, label in labelled})\n"
            "if len(classes) < 2 or len(labelled) < 8:\n"
            "    probe_report = {'verdict': 'not-measurable', 'reason': 'the probe needs at least two labelled classes and 8 labelled sequences', 'needs': 'supply a label column (CSV) or use the default sample; embeddings alone carry no intrinsic metric'}\n"
            "    print(probe_report)\n"
            "else:\n"
            "    rng = random.Random(SEED)\n"
            "    by_class = {c: [i for i, label in labelled if label == c] for c in classes}\n"
            "    train_idx, test_idx = [], []\n"
            "    for c in classes:\n"
            "        rows = list(by_class[c])\n"
            "        rng.shuffle(rows)\n"
            "        cut = max(1, len(rows) // 2)\n"
            "        train_idx += rows[:cut]\n"
            "        test_idx += rows[cut:]\n\n"
            "    def cosine(a, b):\n"
            "        dot = sum(x * y for x, y in zip(a, b))\n"
            "        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))\n\n"
            "    centroids = {}\n"
            "    for c in classes:\n"
            "        rows = [vectors[i] for i in train_idx if labels[i] == c]\n"
            "        centroids[c] = [sum(col) / len(rows) for col in zip(*rows)]\n\n"
            "    hits, margins, per_class = 0, [], {c: {'hits': 0, 'support': 0} for c in classes}\n"
            "    for i in test_idx:\n"
            "        scores = {c: cosine(vectors[i], centroids[c]) for c in classes}\n"
            "        best = max(scores, key=scores.get)\n"
            "        ordered = sorted(scores.values(), reverse=True)\n"
            "        margins.append(ordered[0] - ordered[1])\n"
            "        per_class[labels[i]]['support'] += 1\n"
            "        if best == labels[i]:\n"
            "            hits += 1\n"
            "            per_class[labels[i]]['hits'] += 1\n"
            "    probe_accuracy = hits / len(test_idx)\n\n"
            "    train_gc = [(gc_fraction(sequences[i]), labels[i]) for i in train_idx]\n"
            "    best_rule = (-1.0, 0.0, True)\n"
            "    for threshold, _ in train_gc:\n"
            "        for high_is_first in (True, False):\n"
            "            acc = sum(((classes[0] if (x >= threshold) == high_is_first else classes[-1]) == y) for x, y in train_gc) / len(train_gc)\n"
            "            if acc > best_rule[0]:\n"
            "                best_rule = (acc, threshold, high_is_first)\n"
            "    _, gc_threshold, gc_high_is_first = best_rule\n"
            "    gc_hits = sum(((classes[0] if (gc_fraction(sequences[i]) >= gc_threshold) == gc_high_is_first else classes[-1]) == labels[i]) for i in test_idx)\n"
            "    majority = max(classes, key=lambda c: sum(1 for i in train_idx if labels[i] == c))\n"
            "    majority_hits = sum(1 for i in test_idx if labels[i] == majority)\n\n"
            "    probe_report = {\n"
            "        'verdict': 'sample-sanity',\n"
            "        'task': 'nearest-centroid probe on mean-pooled embeddings (no model training)',\n"
            "        'estimation': f'single stratified split, seed {SEED}, no dispersion estimate',\n"
            "        'classes': classes,\n"
            "        'splits': {'train': len(train_idx), 'test': len(test_idx)},\n"
            "        'probe_accuracy': round(probe_accuracy, 4),\n"
            "        'per_class': {c: {**v, 'accuracy': round(v['hits'] / v['support'], 4) if v['support'] else None} for c, v in per_class.items()},\n"
            "        'mean_cosine_margin': round(sum(margins) / len(margins), 5),\n"
            "        'min_cosine_margin': round(min(margins), 5),\n"
            "        'baselines': {\n"
            "            'majority_class': {'label': majority, 'accuracy': round(majority_hits / len(test_idx), 4)},\n"
            "            'gc_content_threshold': {'threshold': round(gc_threshold, 4), 'train_accuracy': round(best_rule[0], 4), 'accuracy': round(gc_hits / len(test_idx), 4)},\n"
            "        },\n"
            "        'note': 'tutorial sample-sanity evidence on synthetic sequences; not a genomics benchmark',\n"
            "    }\n"
            "    for key in ('probe_accuracy', 'per_class', 'mean_cosine_margin', 'min_cosine_margin', 'baselines'):\n"
            "        print({key: probe_report[key]})\n\n"
            "with open('outputs/nucleotide_transformer_probe_report.json', 'w', encoding='utf-8') as handle:\n"
            "    json.dump(probe_report, handle, indent=2)\n"
            "print('wrote outputs/nucleotide_transformer_probe_report.json')"
        )
    )

    add(
        _md(
            "## 9. Masked-token prediction\n\n"
            "The checkpoint is a masked language model, so its other output is a distribution over the 4,107 vocabulary entries at each "
            "position. Masking one token in a repetitive sequence makes the behaviour legible: the model should prefer the repeated "
            "6-mer, and its runners-up should be near neighbours of it.\n\n"
            "The values printed are raw **logits**, not probabilities. Ranking them is meaningful; reading them as confidence is not."
        )
    )
    add(
        _code(
            "masked = 'ATTCCG' * 3 + tokenizer.mask_token + 'ATTCCG' * 3\n"
            "encoded = tokenizer(masked, return_tensors='pt')\n"
            "position = int((encoded['input_ids'][0] == tokenizer.mask_token_id).nonzero()[0, 0])\n"
            "with torch.no_grad():\n"
            "    logits = model(encoded['input_ids'].to(DEVICE), attention_mask=encoded['attention_mask'].to(DEVICE))['logits']\n"
            "top = torch.topk(logits[0, position], 5)\n"
            "masked_predictions = [(tokenizer.convert_ids_to_tokens([int(i)])[0], round(float(v), 3)) for v, i in zip(top.values, top.indices)]\n"
            "print({'masked_sequence': masked, 'masked_token_position': position})\n"
            "print({'top_5_tokens_and_logits': masked_predictions, 'note': 'logits, not calibrated probabilities'})"
        )
    )

    add(
        _md(
            "## 10. Export and provenance\n\n"
            "The last output, `outputs/nucleotide_transformer_result.json`, gathers what a reader needs to interpret the files above: "
            "the notebook source revision, the model id, immutable revision and **licence**, the fact that remote code was executed and "
            "which files were verified first, the sample identity, the probe report, the masked-token result, and the runtime versions "
            "and device. No credential is involved anywhere in this notebook, so none can leak into it."
        )
    )
    add(
        _code(
            "result_payload = {\n"
            "    'task': 'DNA sequence representation (Nucleotide Transformer v2 50M multi-species)',\n"
            "    'model_id': MODEL_ID,\n"
            "    'model_revision': MODEL_REVISION,\n"
            "    'model_license': MODEL_LICENSE,\n"
            "    'license_conditions': 'CC BY-NC-SA 4.0: attribution, NonCommercial, ShareAlike; commercial use is not permitted under this licence',\n"
            "    'remote_code_executed': True,\n"
            "    'remote_code_files': [e['path'] for e in MANIFEST['files'] if e['path'].endswith('.py')],\n"
            "    'snapshot_verified_files': len(MANIFEST['files']),\n"
            "    'notebook_source': NOTEBOOK_SOURCE,\n"
            "    'data_source': data_source,\n"
            "    'input_manifest': input_manifest,\n"
            "    'embedding': {'n_sequences': len(vectors), 'dimension': len(vectors[0]), 'pooling': 'mean of the last hidden state over non-padding tokens', 'seconds': embed_seconds},\n"
            "    'probe_report': probe_report,\n"
            "    'masked_token_prediction': {'sequence': masked, 'top_5': masked_predictions, 'values': 'logits'},\n"
            "    'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': DEVICE, 'precision': 'float32'},\n"
            "}\n"
            "with open('outputs/nucleotide_transformer_result.json', 'w', encoding='utf-8') as handle:\n"
            "    json.dump(result_payload, handle, indent=2)\n\n"
            "print('outputs/:')\n"
            "for path in sorted(Path('outputs').rglob('*')):\n"
            "    if path.is_file():\n"
            "        print(f'  - {path.as_posix()} ({path.stat().st_size / 1024:.1f} KB)')"
        )
    )

    add(
        _md(
            "## Interpretation and limits\n\n"
            "The probe separates two classes of DNA that contain exactly the same bases in the same proportions, differing only in the "
            "order of six letters at six positions — which the GC-content baseline cannot do, by construction. That is the whole claim: "
            "the embedding retains order information that composition statistics discard. It is not a claim that this model predicts "
            "promoters, enhancers, splice sites, variant effects or any published benchmark, and the cosine margin between the two "
            "centroids is small enough that you should treat the separation as detectable rather than robust.\n\n"
            "On real genomic data the same workflow needs much more care than this sample shows. Sequences from the same gene family, "
            "the same genomic neighbourhood or the same assembly are not independent, so a random split leaks and a chromosome- or "
            "locus-aware split is required. `N` runs and repeat-masked (lowercase) regions change the tokenization before the model sees "
            "anything. The model is multi-species and its pretraining mix determines where its representations are informative; a "
            "clade far from those 850 genomes is out of distribution with no error to tell you so. And a probe on 24 held-out sequences "
            "is an existence check, not an estimate — for a real task, fine-tune or fit a proper classifier, use several seeds, and "
            "report dispersion.\n\n"
            "Successful execution proves that this notebook can acquire and digest-verify the pinned snapshot including the code it "
            "executes, tokenize DNA, produce embeddings whose held-out probe beats two trivial baselines, and emit the shown "
            "machine-readable outputs — without the repository being reachable. It does **not** establish benchmark performance, "
            "production fitness, or biological validity.\n\n"
            "**Licence, once more, because it governs what you may do with everything above:** the weights are CC BY-NC-SA 4.0. Your "
            "use must be non-commercial; attribution is required; a distributed adaptation must carry the same or a compatible licence. "
            "This notebook and this repository grant you no rights beyond those the upstream licensor gives you, and hosting the model "
            "on DIMER has not been approved — see `MODEL_CARD.md` for the two open gates (licence and remote code).\n\n"
            "**Optional experiments (they do not affect the default path):** raise `MOTIF_COPIES` and watch the cosine margin grow; "
            "replace `DECOY` with a different permutation of the same letters; insert an `N` into one sequence and compare its token "
            "count with its neighbour's; or bring your own labelled FASTA through BYOD and read the GC baseline first — if it already "
            "separates your classes, your labels may be predictable from composition alone.\n\n"
            "## References\n\n"
            "- Repository model card: https://github.com/kurtvalcorza/nucleotide-transformer-genomics-pipeline/blob/main/MODEL_CARD.md\n"
            f"- Upstream model: https://huggingface.co/{MODEL_ID}\n"
            "- Upstream code: https://github.com/instadeepai/nucleotide-transformer\n"
            "- Dalla-Torre, H. et al. (2023). The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human "
            "Genomics. bioRxiv 2023.01.11.523679. https://doi.org/10.1101/2023.01.11.523679\n"
            "- Licence: https://creativecommons.org/licenses/by-nc-sa/4.0/"
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "dimer": {
                "notebook_profile": PROFILE,
                "notebook_mode": MODE,
                "notebook_spec": NOTEBOOK_SPEC,
                "standalone": True,
                "remote_code_executed": True,
                "model_license": MODEL_LICENSE,
                "generated_from": {
                    "repository": "nucleotide-transformer-genomics-pipeline",
                    "generator": GENERATOR,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                },
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def to_bytes(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed notebook differs")
    args = parser.parse_args(argv)
    rendered = to_bytes(build())
    if args.check:
        current = NOTEBOOK.read_bytes().replace(b"\r\n", b"\n") if NOTEBOOK.exists() else b""
        if current != rendered:
            print(f"STALE: {NOTEBOOK} differs from tools/make_notebook.py output", file=sys.stderr)
            return 1
        print(f"OK: {NOTEBOOK.name} is up to date ({len(rendered)} bytes)")
        return 0
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_bytes(rendered)
    print(f"wrote {NOTEBOOK} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
