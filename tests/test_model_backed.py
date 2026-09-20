"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): the identity of
what the remote code loads, embeddings and masked prediction, the frozen linear probe, a one-epoch adaptation
of the head plus one block on a slice of the pinned sample, the artifact round trip, the loader's scope and
licence checks, the transactional guarantee and - where CUDA is visible - the same path on the accelerator.
Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import json

import pytest

from nucleotide_transformer_genomics_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    ENCODER_PARAMETERS,
    HIDDEN_SIZE,
    MODEL_LICENSE,
    PARAMETER_COUNT,
    REMOTE_CODE_FILES,
    WEIGHT_FILE,
    NucleotideTransformerPipeline,
    sample_dataset,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)


@pytest.fixture(scope="module")
def data():
    splits = sample_dataset()
    return {"train": splits["train"][:16] + splits["train"][-16:], "val": splits["validation"][:8] + splits["validation"][-8:], "test": splits["test"][:8] + splits["test"][-8:]}


@pytest.fixture(scope="module")
def pipe():
    return NucleotideTransformerPipeline.from_pretrained(device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_identity_of_what_the_remote_code_loads(pipe):
    model = pipe._model
    assert type(model).__name__ == "EsmForMaskedLM" and type(model).__module__.endswith("modeling_esm")
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT
    assert sum(p.numel() for n, p in model.named_parameters() if n.startswith("esm.")) == ENCODER_PARAMETERS
    assert model.config.hidden_size == HIDDEN_SIZE and model.config.add_bias_fnn is False
    assert pipe.weight_sha256 is not None and len(pipe.weight_sha256) == 64
    assert pipe.snapshot["remote_code_files"] == list(REMOTE_CODE_FILES)
    assert pipe.remote_code_executed is True
    assert type(pipe._tokenizer).__name__ == "EsmTokenizer"


def test_embeddings_tokens_and_masked_prediction(pipe, data):
    sequences = [r["sequence"] for r in data["test"][:4]]
    vectors = pipe.embed(sequences, batch_size=2)
    assert len(vectors) == 4 and all(len(v) == HIDDEN_SIZE for v in vectors)
    again = pipe.embed(sequences, batch_size=4)
    assert max(abs(a - b) for va, vb in zip(vectors, again, strict=True) for a, b in zip(va, vb, strict=True)) < 1e-5, "batching does not change the pooled vectors beyond float drift"
    counts = pipe.token_counts(sequences)
    assert counts == [47] * 4, "251 bases = 41 six-mers + 5 single bases + <cls>"
    masked = pipe.predict_masked("ATTCCG" * 3 + pipe._tokenizer.mask_token + "ATTCCG" * 3, top_k=3)
    assert len(masked["tokens"]) == 3 and abs(sum(masked["probabilities"])) <= 1.0 + 1e-6
    with pytest.raises(ValueError, match="mask token"):
        pipe.predict_masked("ACGT" * 10)
    with pytest.raises(RuntimeError, match="no promoter head"):
        pipe.predict(sequences)


def test_linear_probe_scores_the_frozen_representation(pipe, data):
    probe = pipe.linear_probe(data["train"], data["test"])
    assert set(probe) >= {"accuracy", "mcc", "train_accuracy", "n", "probe", "seconds"}
    assert probe["n"] == 16 and 0.0 <= probe["accuracy"] <= 1.0
    assert pipe.adapter is None, "the probe does not touch the pipeline's weights or state"


def test_adapt_one_epoch_then_round_trip_the_artifact(pipe, data, tmp_path):
    before = {k: v.detach().clone() for k, v in pipe._model.state_dict().items() if k.startswith("esm.encoder.layer.11.")}
    seen = []
    result = pipe.adapt(data["train"], data["val"], epochs=1, lr=5e-5, layers=1, batch_size=8, seed=3, progress=seen.append)
    assert result["best_epoch"] in (0, 1) and len(result["history"]) == 2 and len(seen) == 2
    assert result["layers"] == 1 and result["n_head"] == HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + 2 * HIDDEN_SIZE + 2
    assert all(n.startswith("esm.encoder.layer.11.") for n in result["trainable_names"])
    assert result["history"][0]["note"].startswith("untrained head")
    predictions = pipe.predict([r["sequence"] for r in data["test"]])
    assert len(predictions["labels"]) == 16 and set(predictions["labels"]) <= {0, 1}
    scored = pipe.evaluate(data["test"])
    assert scored["n"] == 16
    if result["best_epoch"] == 1:
        changed = any(not torch.equal(before[k], pipe._model.state_dict()[k]) for k in before)
        assert changed, "epoch 1 was kept, so the last block must have moved"

    artifact = pipe.save_artifact(tmp_path / "adapter", metadata={"test": True})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["license"] == MODEL_LICENSE and manifest["base"]["remote_code_files"] == list(REMOTE_CODE_FILES)
    assert any(t.startswith("head.") for t in manifest["tensors"]) and any(t.startswith("esm.encoder.layer.11.") for t in manifest["tensors"])
    reloaded = NucleotideTransformerPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    again = reloaded.predict([r["sequence"] for r in data["test"]])
    assert again["labels"] == predictions["labels"]
    assert max(abs(a - b) for pa, pb in zip(again["probabilities"], predictions["probabilities"], strict=True) for a, b in zip(pa, pb, strict=True)) < 1e-5
    assert reloaded.adapter["layers"] == 1 and reloaded.adapter["seed"] == 3

    # a tampered tensor set is refused before anything is overlaid
    tampered = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    tampered["tensors"].append("esm.encoder.layer.0.output.dense.weight")
    (tmp_path / "adapter" / "manifest.json").write_text(json.dumps(tampered), encoding="utf-8")
    fresh = NucleotideTransformerPipeline.from_pretrained(device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    with pytest.raises(ValueError, match="tensor set"):
        fresh.load_artifact(artifact)
    assert fresh.adapter is None and fresh._head is None


def test_adapt_restores_the_frozen_weights_when_training_fails(data):
    fresh = NucleotideTransformerPipeline.from_pretrained(device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    before = {k: v.detach().clone() for k, v in fresh._model.state_dict().items() if k.startswith("esm.encoder.layer.11.")}

    def explode(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        fresh.adapt(data["train"], data["val"], epochs=2, layers=1, batch_size=8, progress=explode)
    assert all(torch.equal(before[k], fresh._model.state_dict()[k]) for k in before)
    assert fresh.adapter is None and fresh._head is None
    assert all(not p.requires_grad for p in fresh._model.parameters())
    with pytest.raises(ValueError, match="both labels"):
        fresh.adapt([{**r, "label": 1} for r in data["train"]], epochs=1)
    with pytest.raises(ValueError, match="epochs"):
        fresh.adapt(data["train"], epochs=0)
    with pytest.raises(ValueError, match="lr"):
        fresh.adapt(data["train"], lr=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_path_matches_the_contract(data, tmp_path):
    pipe = NucleotideTransformerPipeline.from_pretrained(device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    assert pipe.device == "cuda:0"
    vectors = pipe.embed([r["sequence"] for r in data["test"][:2]])
    assert len(vectors[0]) == HIDDEN_SIZE
    result = pipe.adapt(data["train"], data["val"], epochs=1, layers=2, batch_size=8)
    assert len(result["history"]) == 2
    artifact = pipe.save_artifact(tmp_path / "cuda-adapter")
    reloaded = NucleotideTransformerPipeline.from_artifact(artifact, device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    sequences = [r["sequence"] for r in data["test"]]
    assert reloaded.predict(sequences)["labels"] == pipe.predict(sequences)["labels"]
