"""DNA sequence representation and bounded promoter fine-tuning over the pinned
``InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`` checkpoint.

**Remote code, inside a verified perimeter.** This checkpoint cannot be loaded by the native Transformers ESM
classes: its ``config.json`` carries an ``auto_map`` and its feed-forward block is a bias-free SwiGLU the native
implementation does not have, so ``trust_remote_code=True`` is an architectural requirement. The two Python
files the loader executes (``modeling_esm.py``, ``esm_config.py``) are entries of the snapshot manifest, so
``from_pretrained`` refuses to run without a manifest, stages any absent file at the pinned revision, re-hashes
every file (``verify_snapshot``) and only then imports the model code. The tokenizer is the native
``EsmTokenizer`` and loads with ``trust_remote_code=False``. Digest verification proves the executed code is
the pinned upstream code byte for byte; it is not a safety claim about that code.

Weights are **CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike). Everything derived from them -
embeddings, probes, adapters saved by ``save_artifact`` - carries the same conditions.

Two capabilities: ``embed`` (mean-pooled 512-d representations; the ``TASK-INFERENCE`` contract) and, after
``adapt`` or ``load_artifact``, ``predict`` / ``evaluate`` (binary sequence classification through a small
mean-pooled head trained together with the last encoder blocks). ``linear_probe`` scores the frozen model
on the same task without touching its weights - the honest reference an adaptation has to beat.
"""

from __future__ import annotations

# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width
import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
MODEL_REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"
MODEL_LICENSE = "cc-by-nc-sa-4.0"
MODEL_KEY = "nt-v2-50m-multi-species"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHT_FILE = "model.safetensors"
REMOTE_CODE_FILES = ("modeling_esm.py", "esm_config.py")  # executed by the loader; manifest entries, verified before import
HIDDEN_SIZE = 512
NUM_LAYERS = 12
KMER = 6
MAX_TOKENS = 1_000  # the length the checkpoint was trained at (6,000 bases as 6-mers)
MAX_BASES = KMER * MAX_TOKENS
MIN_BASES = 12
MAX_SEQUENCES = 20_000  # per call
DEFAULT_BATCH_SIZE = 16
PARAMETER_COUNT = 55_904_972  # EsmForMaskedLM as loaded (encoder + masked-LM head)
ENCODER_PARAMETERS = 53_534_401  # the `esm` encoder the head reads
DEFAULT_TRAINED_LAYERS = 2  # last encoder blocks trained by `adapt` alongside the head
HEAD_HIDDEN = 512
NUM_LABELS = 2
LABEL_NAMES = ("no_promoter", "promoter")
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = "1.0"
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
HEAD_PREFIX = "head."
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
MAX_EVAL_RECORDS = 20_000
POOLING = "mean of the last hidden state over non-padding tokens"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    missing = [name for name in (WEIGHT_FILE, *REMOTE_CODE_FILES) if name not in listed]
    if missing:
        raise ValueError(f"manifest does not list {missing}; the remote code must be inside the verified perimeter")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch. The manifest must
    list the weight file and both remote-code files, so a passing check means the code about to be imported
    is byte-identical to the pinned revision."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "remote_code_files": list(REMOTE_CODE_FILES),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights and the model code). Returns the relative paths fetched; `verify_snapshot`
    still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def validate_sequences(sequences: Any) -> list[str]:
    """Upper-cased A/C/G/T/N strings within the base limits; raises ValueError before any model import."""
    if isinstance(sequences, str | bytes) or not isinstance(sequences, Sequence):
        raise ValueError("sequences must be a list of str")
    if not 1 <= len(sequences) <= MAX_SEQUENCES:
        raise ValueError(f"{len(sequences)} sequences; 1..{MAX_SEQUENCES} per call")
    checked = []
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, str):
            raise ValueError(f"sequences[{index}] must be a str")
        upper = sequence.strip().upper()
        if not MIN_BASES <= len(upper) <= MAX_BASES:
            raise ValueError(f"sequences[{index}]: {len(upper)} bases; {MIN_BASES}..{MAX_BASES} are required")
        bad = sorted(set(upper) - set("ACGTN"))
        if bad:
            raise ValueError(f"sequences[{index}]: characters outside A/C/G/T/N: {bad[:5]}")
        checked.append(upper)
    return checked


def validate_inputs(sequences: Sequence[str], *, names: Sequence[str] | None = None) -> dict[str, Any]:
    """The input manifest the tutorial records before inference (DAT24): counts, lengths, GC, N content."""
    checked = validate_sequences(sequences)
    if names is not None and len(names) != len(checked):
        raise ValueError(f"{len(names)} names for {len(checked)} sequences")
    lengths = [len(s) for s in checked]
    gc = [(s.count("G") + s.count("C")) / len(s) for s in checked]
    return {
        "n_sequences": len(checked),
        "bases": {"min": min(lengths), "max": max(lengths), "total": sum(lengths)},
        "gc_fraction": {"min": round(min(gc), 4), "max": round(max(gc), 4)},
        "n_with_N": sum("N" in s for s in checked),
        "tokenization": f"{KMER}-mers over the checkpoint vocabulary, single bases around N and at a trailing remainder; ceiling {MAX_TOKENS} tokens = {MAX_BASES} bases",
        "names": list(names) if names is not None else None,
        "sequence_sha256": [hashlib.sha256(s.encode()).hexdigest() for s in checked],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    adapted: Mapping[str, Any],
    frozen: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
    *,
    sample_kind: str = "pinned promoter sample",
) -> dict[str, Any]:
    """Structured verdict (EVAL21): the adapted model against the frozen probe and the non-neural baselines
    on the same held-out records, with the small-sample flag and no claim beyond the numbers."""
    n = int(adapted["n"])
    if n != int(frozen["n"]) or any(int(b["n"]) != n for b in baselines):
        raise ValueError("adapted, frozen and baseline metrics must score the same records")
    best_baseline = max(baselines, key=lambda b: b["mcc"])
    return {
        "n": n,
        "sample_kind": sample_kind,
        "small_sample": n < MIN_SCORED_RECORDS,
        "headline_metric": "mcc",
        "adapted": {"accuracy": adapted["accuracy"], "mcc": adapted["mcc"]},
        "frozen_probe": {"accuracy": frozen["accuracy"], "mcc": frozen["mcc"]},
        "best_baseline": {"name": best_baseline.get("baseline"), "accuracy": best_baseline["accuracy"], "mcc": best_baseline["mcc"]},
        "adapted_beats_frozen": adapted["mcc"] > frozen["mcc"],
        "adapted_beats_baselines": all(adapted["mcc"] > b["mcc"] for b in baselines),
        "frozen_beats_baselines": all(frozen["mcc"] > b["mcc"] for b in baselines),
        "mcc_gain_over_frozen": round(adapted["mcc"] - frozen["mcc"], 4),
        "note": "sample-sanity evidence on one seeded draw with no dispersion estimate; not a benchmark reproduction",
    }


def _trainable_names(model: Any, layers: int) -> list[str]:
    """Encoder tensors `adapt` trains: the last `layers` blocks of the 12-block encoder (0 = head only)."""
    if isinstance(layers, bool) or not isinstance(layers, int) or not 0 <= layers <= NUM_LAYERS:
        raise ValueError(f"layers must be an int in 0..{NUM_LAYERS}")
    wanted = {f"esm.encoder.layer.{NUM_LAYERS - 1 - k}." for k in range(layers)}
    names = [name for name, _ in model.named_parameters() if any(name.startswith(prefix) for prefix in wanted)]
    if layers == NUM_LAYERS:  # the whole encoder also trains its final layer norm (embeddings stay frozen)
        names += [name for name, _ in model.named_parameters() if name.startswith("esm.encoder.emb_layer_norm_after")]
    return names


def _check_artifact_manifest(manifest: Mapping[str, Any], artifact_dir: Path, base_sha256: str) -> None:
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base", {})
    if (base.get("model_id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("artifact was trained on a different base model or revision")
    if base_sha256 and base.get("weight_sha256") and base["weight_sha256"] != base_sha256:
        raise ValueError("artifact base weight digest does not match the loaded snapshot")
    if manifest.get("license") != MODEL_LICENSE:
        raise ValueError(f"artifact licence {manifest.get('license')!r} != {MODEL_LICENSE!r}; the adapter inherits the base licence")
    files = {entry["path"]: entry for entry in manifest.get("files", [])}
    if ADAPTER_WEIGHTS not in files:
        raise ValueError(f"artifact manifest does not list {ADAPTER_WEIGHTS}")
    weights = artifact_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights}")
    if weights.stat().st_size != files[ADAPTER_WEIGHTS]["bytes"]:
        raise ValueError("artifact weights size does not match the manifest")
    if _sha256(weights) != files[ADAPTER_WEIGHTS]["sha256"]:
        raise ValueError("artifact weights digest does not match the manifest")


def _build_head(seed: int) -> Any:
    """The mean-pooled classification head, initialised from a fixed seed so two fresh pipelines agree."""
    import torch

    class PromoterHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = torch.nn.Linear(HIDDEN_SIZE, HEAD_HIDDEN)
            self.out = torch.nn.Linear(HEAD_HIDDEN, NUM_LABELS)

        def forward(self, hidden: Any, attention_mask: Any) -> Any:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            return self.out(torch.tanh(self.dense(pooled)))

    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        head = PromoterHead()
    finally:
        torch.random.set_rng_state(generator_state)
    return head


@dataclass
class NucleotideTransformerPipeline:
    """Sequence representations (`embed`) and, once adapted, promoter classification (`predict`) over the
    pinned Nucleotide Transformer v2 50M checkpoint loaded through its digest-verified remote code."""

    device: str
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)
    _head: Any = field(default=None, repr=False)
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None
    remote_code_executed: bool = True

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> NucleotideTransformerPipeline:
        """Stage (if allowed) and verify the snapshot, then load the tokenizer natively and the model through
        the verified remote code. There is no manifest-less path: without a digest check nothing is imported."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        stage_missing_files(root, allow_download=allow_download)
        snapshot = verify_snapshot(root)  # every file, including the two .py files, before any import
        with open(root / MANIFEST_NAME, encoding="utf-8") as handle:
            entries = json.load(handle).get("files", [])
        weight_sha256 = next((e["sha256"] for e in entries if e["path"] == WEIGHT_FILE), None)

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
        model = AutoModelForMaskedLM.from_pretrained(str(root), local_files_only=True, trust_remote_code=True, dtype=torch.float32)
        if type(model).__module__.split(".")[-1] != "modeling_esm":
            raise RuntimeError(f"expected the snapshot's modeling_esm module, loaded {type(model).__module__}")
        model = model.to(resolved_device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        pipe = cls(resolved_device, model, tokenizer, None, weight_sha256)
        pipe.snapshot = snapshot  # type: ignore[attr-defined]
        return pipe

    # ------------------------------------------------------------------ representation (TASK-INFERENCE)

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("no model loaded: construct with from_pretrained or from_artifact")
        return self._model, self._tokenizer

    def _encode(self, sequences: Sequence[str]) -> Any:
        _model, tokenizer = self._require_model()
        return tokenizer(list(sequences), return_tensors="pt", padding=True).to(self.device)

    def _hidden(self, encoded: Any) -> Any:
        model, _tokenizer = self._require_model()
        return model.esm(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).last_hidden_state

    def token_counts(self, sequences: Sequence[str]) -> list[int]:
        """Tokens per sequence (6-mers plus single-base fallbacks, with the special tokens)."""
        checked = validate_sequences(sequences)
        _model, tokenizer = self._require_model()
        return [len(tokenizer(s)["input_ids"]) for s in checked]

    def embed(self, sequences: Sequence[str], *, batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
        """One mean-pooled 512-d vector per sequence (padding excluded) from the frozen encoder."""
        checked = validate_sequences(sequences)
        self._require_model()
        import torch

        vectors: list[list[float]] = []
        with torch.no_grad():  # not inference_mode: the remote code caches rotary tables on first use, and inference tensors cannot feed a later backward pass
            for start in range(0, len(checked), batch_size):
                encoded = self._encode(checked[start : start + batch_size])
                hidden = self._hidden(encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                vectors.extend(pooled.float().cpu().tolist())
        return vectors

    def predict_masked(self, sequence: str, *, top_k: int = 5) -> dict[str, Any]:
        """Masked-token prediction for one sequence containing the tokenizer's mask token (the checkpoint's
        pre-training objective); returns the top-k tokens and probabilities at the masked position."""
        model, tokenizer = self._require_model()
        if not isinstance(sequence, str) or tokenizer.mask_token not in sequence:
            raise ValueError(f"sequence must contain the mask token {tokenizer.mask_token!r}")
        import torch

        encoded = tokenizer(sequence, return_tensors="pt").to(self.device)
        position = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero()
        if len(position) != 1:
            raise ValueError("exactly one mask token is required")
        with torch.no_grad():  # not inference_mode: the remote code caches rotary tables on first use, and inference tensors cannot feed a later backward pass
            logits = model(**encoded).logits[0, int(position[0, 0])]
        probabilities = torch.softmax(logits.float(), dim=-1)
        values, indices = probabilities.topk(top_k)
        return {
            "tokens": tokenizer.convert_ids_to_tokens(indices.tolist()),
            "probabilities": [round(float(v), 6) for v in values.tolist()],
            "position": int(position[0, 0]),
        }

    # ------------------------------------------------------------------ frozen reference

    def linear_probe(
        self,
        train: Sequence[Mapping[str, Any]],
        test: Sequence[Mapping[str, Any]],
        *,
        l2: float = 1e-2,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict[str, Any]:
        """Logistic regression on standardised frozen mean-pooled embeddings (L-BFGS, L2-penalised, no seed
        dependence): the model's weights are untouched, so this is what the pre-trained representation alone
        knows about the task. Returns the test metrics plus the fitted probe's train accuracy."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        train_checked = validate_dataset(train, require_both_labels=True)["records"]
        test_checked = validate_dataset(test, min_records=1)["records"]
        self._require_model()
        import torch

        started = time.perf_counter()
        x_train = torch.tensor(self.embed([r["sequence"] for r in train_checked], batch_size=batch_size))
        x_test = torch.tensor(self.embed([r["sequence"] for r in test_checked], batch_size=batch_size))
        mean, std = x_train.mean(0), x_train.std(0) + 1e-6
        x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
        y_train = torch.tensor([float(r["label"]) for r in train_checked])
        weight = torch.zeros(HIDDEN_SIZE, requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([weight, bias], max_iter=300, line_search_fn="strong_wolfe")

        def closure() -> Any:
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x_train @ weight + bias, y_train) + l2 * (weight * weight).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            train_predictions = ((x_train @ weight + bias) > 0).int().tolist()
            test_predictions = ((x_test @ weight + bias) > 0).int().tolist()
        scored = classification_metrics(test_predictions, [r["label"] for r in test_checked])
        return {
            "probe": "logistic regression on frozen mean-pooled embeddings",
            "l2": l2,
            "train_accuracy": classification_metrics(train_predictions, [r["label"] for r in train_checked])["accuracy"],
            "seconds": round(time.perf_counter() - started, 3),
            **scored,
        }

    # ------------------------------------------------------------------ classification (after adapt)

    def predict(self, sequences: Sequence[str], *, batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, Any]:
        """Labels and class probabilities for sequences; requires an adapter (`adapt` or `load_artifact`)."""
        checked = validate_sequences(sequences)
        self._require_model()
        if self._head is None or self.adapter is None:
            raise RuntimeError("no promoter head: call adapt() or load an artifact before predict()")
        import torch

        labels: list[int] = []
        probabilities: list[list[float]] = []
        self._head.eval()
        with torch.no_grad():  # not inference_mode: the remote code caches rotary tables on first use, and inference tensors cannot feed a later backward pass
            for start in range(0, len(checked), batch_size):
                encoded = self._encode(checked[start : start + batch_size])
                logits = self._head(self._hidden(encoded), encoded["attention_mask"])
                probs = torch.softmax(logits.float(), dim=-1)
                labels.extend(int(v) for v in probs.argmax(-1).tolist())
                probabilities.extend([round(float(p), 6) for p in row] for row in probs.tolist())
        return {"labels": labels, "label_names": [LABEL_NAMES[v] for v in labels], "probabilities": probabilities, "n": len(checked)}

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Classification metrics of the adapted model over labelled records (validated first)."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        predictions: list[int] = []
        for start in range(0, len(checked), batch_size):
            predictions.extend(self.predict([r["sequence"] for r in checked[start : start + batch_size]], batch_size=batch_size)["labels"])
            if progress is not None:
                progress(min(start + batch_size, len(checked)), len(checked))
        return classification_metrics(predictions, [r["label"] for r in checked])

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 6,
        lr: float = 3e-5,
        layers: int = DEFAULT_TRAINED_LAYERS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = 0,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning: a fresh mean-pooled head (seeded) plus the last `layers` encoder blocks are
        trained with cross-entropy, AdamW (no weight decay), gradient clipping at 1.0, seeded shuffling and no
        scheduler; embeddings and the other blocks stay frozen. Epoch 0 records the untrained head's
        validation metrics; the epoch with the highest validation MCC (ties: accuracy, then the earlier epoch)
        is kept, or the final one without a validation split. On any exception the frozen weights are
        restored and the previous adapter, if any, is kept."""
        model, _tokenizer = self._require_model()  # refuse before importing torch
        from .metrics import classification_metrics
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-2:
            raise ValueError("lr must be in (0, 1e-2]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be an int in 1..256")
        names = _trainable_names(model, layers)
        train_checked = validate_dataset(train, require_both_labels=True)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        import torch

        name_set = set(names)
        frozen_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        previous_head, previous_adapter = self._head, self.adapter
        cudnn_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # repeatable on one device
        history: list[dict[str, Any]] = []
        started = time.perf_counter()

        def _val() -> dict[str, Any] | None:
            if val_checked is None:
                return None
            scored = self.evaluate(val_checked, batch_size=max(batch_size, 32))
            return {"accuracy": scored["accuracy"], "mcc": scored["mcc"], "n": scored["n"]}

        def _score(entry: Mapping[str, Any]) -> tuple[float, float]:
            return (entry["val"]["mcc"], entry["val"]["accuracy"]) if entry["val"] else (0.0, 0.0)

        try:
            head = _build_head(seed).to(self.device)
            self._head = head
            self.adapter = {"provisional": True}  # lets evaluate() run during training
            for param in model.parameters():
                param.requires_grad_(False)
            params = list(head.parameters())
            for name, param in model.named_parameters():
                if name in name_set:
                    param.requires_grad_(True)
                    params.append(param)
            n_trainable = sum(p.numel() for p in params)
            entry = {"epoch": 0, "train_loss": None, "val": _val(), "note": "untrained head on the frozen encoder"}
            history.append(entry)
            if progress is not None:
                progress(entry)
            best_epoch, best_score = 0, _score(entry)
            best_state = ({k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}, {k: v.detach().clone() for k, v in head.state_dict().items()})
            optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            for epoch in range(1, epochs + 1):
                model.train()
                head.train()
                order = list(train_checked)
                rng.shuffle(order)
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = order[start : start + batch_size]
                    encoded = self._encode([r["sequence"] for r in batch])
                    logits = head(self._hidden(encoded), encoded["attention_mask"])
                    targets = torch.tensor([r["label"] for r in batch], device=self.device)
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                head.eval()
                entry = {"epoch": epoch, "train_loss": round(sum(losses) / len(losses), 6), "val": _val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                if val_checked is None or _score(entry) > best_score:
                    best_epoch, best_score = epoch, _score(entry)
                    best_state = ({k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}, {k: v.detach().clone() for k, v in head.state_dict().items()})
            model.load_state_dict(best_state[0], strict=False)
            head.load_state_dict(best_state[1])
            for param in model.parameters():
                param.requires_grad_(False)
            for param in head.parameters():
                param.requires_grad_(False)
            model.eval()
            head.eval()
        except BaseException:
            model.load_state_dict(frozen_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            self._head, self.adapter = previous_head, previous_adapter
            raise
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
        train_metrics = classification_metrics(self.predict([r["sequence"] for r in train_checked], batch_size=max(batch_size, 32))["labels"], [r["label"] for r in train_checked])
        self.adapter = {
            "task": "binary promoter classification",
            "pooling": POOLING,
            "layers": layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_head": sum(p.numel() for p in head.parameters()),
            "n_total": sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation MCC (ties: accuracy, earlier epoch)" if val_checked is not None else "final epoch (no validation split)",
            "loss": "cross-entropy over the two labels",
            "lr": float(lr),
            "batch_size": batch_size,
            "seed": seed,
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "train_accuracy": train_metrics["accuracy"],
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)

    # ------------------------------------------------------------------ artifacts

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the head and the trained encoder tensors as safetensors plus a manifest naming the base, the
        digests, the licence the adapter inherits and the training configuration. Requires a prior `adapt`."""
        model, _tokenizer = self._require_model()  # refuse before importing torch
        if self.adapter is None or self._head is None or self.adapter.get("provisional"):
            raise RuntimeError("nothing to save: call adapt() first")
        import torch
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = list(self.adapter["trainable_names"])
        state = model.state_dict()
        tensors = {name: state[name].detach().cpu().contiguous() for name in names}
        tensors.update({HEAD_PREFIX + k: v.detach().cpu().contiguous() for k, v in self._head.state_dict().items()})
        weights = out / ADAPTER_WEIGHTS
        save_file(tensors, str(weights), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "license": MODEL_LICENSE,
            "license_note": "derived from CC BY-NC-SA 4.0 weights: attribution, non-commercial use and share-alike apply to this adapter",
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHT_FILE, "weight_sha256": self.weight_sha256, "remote_code_files": list(REMOTE_CODE_FILES)},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "label_names": list(LABEL_NAMES),
            "files": [{"path": ADAPTER_WEIGHTS, "bytes": weights.stat().st_size, "sha256": _sha256(weights)}],
            "torch": torch.__version__,
            "metadata": dict(metadata or {}),
        }
        with open(out / ADAPTER_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Overlay a saved adapter onto this (freshly loaded) pipeline after checking its manifest, digest,
        licence and exact tensor set. Refuses encoder tensors outside the recorded trained blocks."""
        model, _tokenizer = self._require_model()  # refuse before importing safetensors
        from safetensors.torch import load_file

        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _check_artifact_manifest(manifest, artifact, self.weight_sha256 or "")
        layers = manifest.get("adapter", {}).get("layers")
        expected_encoder = _trainable_names(model, layers)
        head = _build_head(int(manifest.get("adapter", {}).get("seed", 0))).to(self.device)
        expected = sorted(expected_encoder + [HEAD_PREFIX + k for k in head.state_dict()])
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor set does not match its recorded configuration")
        tensors = load_file(str(artifact / ADAPTER_WEIGHTS))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from the manifest")
        state = model.state_dict()
        head_state = head.state_dict()
        for name, tensor in tensors.items():
            target = head_state[name[len(HEAD_PREFIX) :]] if name.startswith(HEAD_PREFIX) else state[name]
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(f"artifact tensor {name} has shape {tuple(tensor.shape)}, base has {tuple(target.shape)}")
        model.load_state_dict({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items() if not k.startswith(HEAD_PREFIX)}, strict=False)
        head.load_state_dict({k[len(HEAD_PREFIX) :]: v.to(self.device) for k, v in tensors.items() if k.startswith(HEAD_PREFIX)})
        for param in head.parameters():
            param.requires_grad_(False)
        model.eval()
        head.eval()
        self._head = head
        self.adapter = {**manifest["adapter"], "trainable_names": expected_encoder, "history": manifest.get("history", [])}
        return dict(self.adapter)

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> NucleotideTransformerPipeline:
        """Load the verified base snapshot, then overlay the adapter (verified before deserialising)."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
