"""Offline tests for the adaptation contract: dataset validation and digests, the pinned inline sample, the
BYOD readers, the metrics and baselines, the artifact manifest checks and the trainable-tensor selection."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import random

import pytest

from nucleotide_transformer_genomics_pipeline import (
    ARTIFACT_FORMAT,
    CORPUS_FILES,
    CORPUS_REVISION,
    INTERVAL_BASES,
    MAX_RECORDS,
    MIN_RECORDS,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    NUM_LAYERS,
    SAMPLE_DIGEST,
    SAMPLE_SPLIT,
    SAMPLE_SPLIT_DIGESTS,
    NucleotideTransformerPipeline,
    check_split_disjoint,
    classification_metrics,
    dataset_digest,
    evaluation_report,
    gc_content,
    gc_threshold_baseline,
    load_byod_dataset,
    majority_baseline,
    sample_csv_text,
    sample_dataset,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from nucleotide_transformer_genomics_pipeline import pipeline as pl
from nucleotide_transformer_genomics_pipeline import samples as sm


def _seq(i: int, n: int = 40) -> str:
    rng = random.Random(i)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _records(n: int = 12) -> list[dict]:
    return [{"id": f"r{i:02d}", "sequence": _seq(i), "label": i % 2} for i in range(n)]


# --- dataset validation -----------------------------------------------------------------------------------------------


def test_validate_dataset_accepts_records_and_reports_counts_and_digest():
    report = validate_dataset(_records())
    assert report["n_records"] == 12
    assert report["label_counts"] == {"0": 6, "1": 6}
    assert report["bases"] == {"min": 40, "max": 40}
    assert 0.0 <= report["gc_fraction"]["min"] <= report["gc_fraction"]["mean"] <= report["gc_fraction"]["max"] <= 1.0
    assert report["digest"] == dataset_digest(_records())
    assert report["model_id"] == MODEL_ID
    assert report["records"][0]["sequence"] == _seq(0)


def test_validate_dataset_upper_cases_and_keeps_provenance():
    record = {"id": "x", "sequence": "acgt" * 5, "label": "1", "region": "chr1", "start": 10, "end": 30, "strand": "-"}
    item = validate_dataset([record], min_records=1)["records"][0]
    assert item["sequence"] == "ACGT" * 5 and item["label"] == 1
    assert item["region"] == "chr1" and item["strand"] == "-"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("label"), "missing 'label'"),
        (lambda r: r.update(id="bad id!"), "id must match"),
        (lambda r: r.update(sequence=123), "sequence must be a str"),
        (lambda r: r.update(sequence="ACGTU" * 4), "outside A/C/G/T/N"),
        (lambda r: r.update(sequence="ACG"), "bases"),
        (lambda r: r.update(label=2), "label must be 0 or 1"),
        (lambda r: r.update(label=True), "label must be 0 or 1"),
    ],
)
def test_validate_dataset_refuses_malformed_records(mutate, message):
    records = _records()
    mutate(records[3])
    with pytest.raises(ValueError, match=message):
        validate_dataset(records)


def test_validate_dataset_enforces_bounds_unique_ids_and_both_labels():
    with pytest.raises(ValueError, match="records; "):
        validate_dataset(_records(MIN_RECORDS - 1))
    with pytest.raises(ValueError, match="records; "):
        validate_dataset(_records(3), min_records=1, max_records=2)
    duplicated = _records()
    duplicated[5]["id"] = duplicated[4]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(duplicated)
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})
    one_class = [{**r, "label": 1} for r in _records()]
    validate_dataset(one_class)
    with pytest.raises(ValueError, match="both labels"):
        validate_dataset(one_class, require_both_labels=True)
    assert sum(SAMPLE_SPLIT.values()) <= MAX_RECORDS


def test_validate_dataset_refuses_before_importing_model_libraries(forbid_model_imports):
    with pytest.raises(ValueError):
        validate_dataset([{"id": "a", "sequence": "ACGT" * 10, "label": 3}] * MIN_RECORDS)


def test_digests_split_and_disjointness():
    records = _records(20)
    digest = dataset_digest(records)
    assert len(digest) == 64 and digest == dataset_digest(list(reversed(records)))
    assert digest != dataset_digest(records[:-1])
    lowered = [{**r, "sequence": r["sequence"].lower()} for r in records]
    assert dataset_digest(lowered) == digest, "digest is case-insensitive like the validator"
    splits = split_dataset(records + [{**records[0], "id": "dup"}], seed=1)
    assert sum(len(v) for v in splits.values()) == 20, "duplicate sequences are dropped before the split"
    assert check_split_disjoint(splits) == {k: len(v) for k, v in splits.items()}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:5], "test": records[4:6]})
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="too few"):
        split_dataset(_records(MIN_RECORDS), test_fraction=0.5, val_fraction=0.4)


# --- the pinned inline sample --------------------------------------------------------------------------------------------


def test_inline_sample_matches_its_pinned_digest_and_split_sizes():
    splits = sample_dataset()
    assert {k: len(v) for k, v in splits.items()} == SAMPLE_SPLIT
    every = [r for part in splits.values() for r in part]
    assert dataset_digest(every) == SAMPLE_DIGEST
    for name, part in splits.items():
        assert dataset_digest(part) == SAMPLE_SPLIT_DIGESTS[name]
        assert sum(r["label"] for r in part) * 2 == len(part), f"{name} is not balanced"
        assert {len(r["sequence"]) for r in part} == {INTERVAL_BASES}
        assert all(r["end"] - r["start"] == INTERVAL_BASES for r in part)
        assert all(r["source_split"] == ("test" if name == "test" else "train") for r in part)
    assert check_split_disjoint(splits) == SAMPLE_SPLIT
    for part in splits.values():
        validate_dataset(part, require_both_labels=True)


def test_inline_sample_is_refused_when_edited(monkeypatch):
    monkeypatch.setattr(sm, "SAMPLE_DIGEST", "0" * 64)
    with pytest.raises(ValueError, match="edited"):
        sample_dataset()
    assert len(sample_dataset(verify=False)["train"]) == SAMPLE_SPLIT["train"]


def test_corpus_pins_are_well_formed():
    assert len(CORPUS_REVISION) == 40
    assert set(CORPUS_FILES) == {"train/positive.csv.gz", "train/negative.csv.gz", "test/positive.csv.gz", "test/negative.csv.gz"}
    for size, digest in CORPUS_FILES.values():
        assert size > 0 and len(digest) == 64


def test_sample_csv_text_is_a_valid_byod_file(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(sample_csv_text(), encoding="utf-8")
    records = load_byod_dataset(path)
    assert len(records) == sum(SAMPLE_SPLIT.values())
    assert dataset_digest(validate_dataset(records)["records"]) == SAMPLE_DIGEST


# --- BYOD readers ---------------------------------------------------------------------------------------------------------


def test_load_byod_dataset_reads_csv_with_and_without_ids(tmp_path):
    path = tmp_path / "byod.csv"
    path.write_text("Sequence,Label\nACGTACGTACGTACGT,1\nttttttttttttcccc,0\n", encoding="utf-8")
    records = load_byod_dataset(path)
    assert [r["id"] for r in records] == ["row00001", "row00002"]
    checked = validate_dataset(records, min_records=1)["records"]
    assert checked[1]["sequence"] == "TTTTTTTTTTTTCCCC"
    with pytest.raises(ValueError, match="header"):
        load_byod_dataset(tmp_path.joinpath("bad.csv").write_text("seq,y\nACGT,1\n", encoding="utf-8") and tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="not a file"):
        load_byod_dataset(tmp_path / "missing.csv")
    out = write_dataset_csv(checked, tmp_path / "out" / "table.csv")
    again = load_byod_dataset(out)
    assert [r["sequence"] for r in again] == [r["sequence"] for r in checked]


# --- metrics and baselines -----------------------------------------------------------------------------------------------


def test_classification_metrics_and_gc_content():
    scored = classification_metrics([1, 1, 0, 0, 1], [1, 0, 0, 0, 1])
    assert scored["n"] == 5 and scored["accuracy"] == 0.8
    assert scored["confusion"] == {"tp": 2, "tn": 2, "fp": 1, "fn": 0}
    assert scored["positive"]["precision"] == round(2 / 3, 4) and scored["positive"]["recall"] == 1.0
    assert abs(scored["mcc"] - 0.6667) < 1e-3
    assert classification_metrics([1, 1, 1], [1, 0, 1])["mcc"] == 0.0, "a constant predictor scores 0"
    with pytest.raises(ValueError, match="predictions for"):
        classification_metrics([1], [1, 0])
    with pytest.raises(ValueError, match="0 or 1"):
        classification_metrics([2], [1])
    assert gc_content("GGCCAATT") == 0.5 and gc_content("ggNN") == 0.5 and gc_content("") == 0.0


def test_baselines_are_fit_on_train_and_scored_on_test():
    train = [{"sequence": "G" * 20, "label": 1}] * 6 + [{"sequence": "A" * 20, "label": 0}] * 4
    test = [{"sequence": "GGGGGGGGGGAAAAAAAAAA", "label": 1}, {"sequence": "A" * 20, "label": 0}, {"sequence": "C" * 20, "label": 1}]
    majority = majority_baseline(train, test)
    assert majority["label"] == 1 and majority["accuracy"] == round(2 / 3, 4) and majority["mcc"] == 0.0
    gc = gc_threshold_baseline(train, test)
    assert gc["rule"] == "GC >= threshold -> positive" and gc["train_accuracy"] == 1.0 and gc["accuracy"] == 1.0
    inverted = gc_threshold_baseline([{**r, "label": 1 - r["label"]} for r in train], [{**r, "label": 1 - r["label"]} for r in test])
    assert inverted["rule"] == "GC < threshold -> positive" and inverted["accuracy"] == 1.0


def test_evaluation_report_compares_on_the_same_records():
    adapted = {"n": 100, "accuracy": 0.83, "mcc": 0.66}
    frozen = {"n": 100, "accuracy": 0.78, "mcc": 0.55}
    baselines = [{"baseline": "majority", "n": 100, "accuracy": 0.5, "mcc": 0.0}, {"baseline": "gc_threshold", "n": 100, "accuracy": 0.72, "mcc": 0.43}]
    report = evaluation_report(adapted, frozen, baselines)
    assert report["adapted_beats_frozen"] and report["adapted_beats_baselines"] and report["frozen_beats_baselines"]
    assert report["best_baseline"]["name"] == "gc_threshold" and report["mcc_gain_over_frozen"] == 0.11
    assert report["small_sample"] is False
    assert evaluation_report({**adapted, "n": 10}, {**frozen, "n": 10}, [{**b, "n": 10} for b in baselines])["small_sample"] is True
    with pytest.raises(ValueError, match="same records"):
        evaluation_report(adapted, {**frozen, "n": 99}, baselines)


# --- adaptation guards -----------------------------------------------------------------------------------------------------


def test_adapt_predict_and_artifacts_require_a_loaded_model(forbid_model_imports):
    pipe = NucleotideTransformerPipeline("cpu")
    with pytest.raises(RuntimeError, match="no model loaded"):
        pipe.adapt(_records())
    with pytest.raises(RuntimeError, match="no model loaded"):
        pipe.save_artifact("x")
    with pytest.raises(RuntimeError, match="no model loaded"):
        pipe.load_artifact("x")
    with pytest.raises(RuntimeError, match="no model loaded"):
        pipe.linear_probe(_records(), _records())


class _Param:
    def numel(self):
        return 1


class _Model:
    def named_parameters(self):
        names = ["esm.embeddings.word_embeddings.weight"]
        for layer in range(NUM_LAYERS):
            names += [f"esm.encoder.layer.{layer}.attention.self.query.weight", f"esm.encoder.layer.{layer}.output.dense.weight"]
        names += ["esm.encoder.emb_layer_norm_after.weight", "lm_head.dense.weight"]
        return [(n, _Param()) for n in names]


def test_trainable_names_selects_the_last_blocks_only():
    assert pl._trainable_names(_Model(), 0) == []
    two = pl._trainable_names(_Model(), 2)
    assert two == ["esm.encoder.layer.10.attention.self.query.weight", "esm.encoder.layer.10.output.dense.weight", "esm.encoder.layer.11.attention.self.query.weight", "esm.encoder.layer.11.output.dense.weight"]
    whole = pl._trainable_names(_Model(), NUM_LAYERS)
    assert len(whole) == 2 * NUM_LAYERS + 1 and not any(n.startswith(("esm.embeddings", "lm_head")) for n in whole)
    for bad in (-1, NUM_LAYERS + 1, True, 1.5):
        with pytest.raises(ValueError, match="layers"):
            pl._trainable_names(_Model(), bad)


def _manifest(tmp_path, **overrides):
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"tensor-bytes")
    manifest = {
        "format": ARTIFACT_FORMAT,
        "license": MODEL_LICENSE,
        "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": "base-digest"},
        "adapter": {"layers": 2, "seed": 0},
        "tensors": ["head.dense.weight", "esm.encoder.layer.11.output.dense.weight"],
        "files": [{"path": "adapter.safetensors", "bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"tensor-bytes").hexdigest()}],
    }
    manifest.update(overrides)
    return manifest


def test_check_artifact_manifest_accepts_a_consistent_manifest_and_refuses_each_deviation(tmp_path):
    pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="format"):
        pl._check_artifact_manifest(_manifest(tmp_path, format="other"), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trained on"):
        pl._check_artifact_manifest(_manifest(tmp_path, base={"model_id": "x", "revision": MODEL_REVISION, "weight_sha256": "base-digest"}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="base weight digest"):
        pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "another-digest")
    with pytest.raises(ValueError, match="licence"):
        pl._check_artifact_manifest(_manifest(tmp_path, license="apache-2.0"), tmp_path, "base-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["bytes"] += 1
    with pytest.raises(ValueError, match="size"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    with pytest.raises(ValueError, match="does not list"):
        pl._check_artifact_manifest(_manifest(tmp_path, files=[]), tmp_path, "base-digest")


def test_manifest_json_round_trip(tmp_path):
    payload = {"epoch": 1, "train_loss": 0.3, "val": {"accuracy": 0.8, "mcc": 0.6, "n": 200}}
    (tmp_path / "h.json").write_text(json.dumps([payload]), encoding="utf-8")
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))[0]["val"]["mcc"] == 0.6
