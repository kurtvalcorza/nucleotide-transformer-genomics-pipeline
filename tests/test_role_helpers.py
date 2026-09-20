"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from nucleotide_transformer_genomics_pipeline import (
    INPUT_SCHEMA,
    KMER,
    MAX_BASES,
    MAX_TOKENS,
    MIN_SCORED_RECORDS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)


def test_input_schema_names_the_contract():
    assert "sequence" in INPUT_SCHEMA["record"] and "label" in INPUT_SCHEMA["record"]
    assert str(MAX_BASES) in INPUT_SCHEMA["record"]["sequence"]
    assert MAX_BASES == KMER * MAX_TOKENS


def test_validate_inputs_records_identity_and_tokenization(forbid_model_imports):
    manifest = validate_inputs(["ACGTACGTACGTACGTACGT"])
    assert manifest["model_id"] == MODEL_ID and manifest["model_revision"] == MODEL_REVISION
    assert f"{KMER}-mers" in manifest["tokenization"] and str(MAX_TOKENS) in manifest["tokenization"]
    assert manifest["gc_fraction"] == {"min": 0.5, "max": 0.5}
    assert manifest["names"] is None


def test_evaluation_report_is_honest_about_small_samples_and_losses():
    n = MIN_SCORED_RECORDS - 1
    adapted = {"n": n, "accuracy": 0.6, "mcc": 0.2}
    frozen = {"n": n, "accuracy": 0.7, "mcc": 0.4}
    baselines = [{"baseline": "majority", "n": n, "accuracy": 0.5, "mcc": 0.0}]
    report = evaluation_report(adapted, frozen, baselines, sample_kind="drawn")
    assert report["small_sample"] is True and report["sample_kind"] == "drawn"
    assert report["adapted_beats_frozen"] is False and report["mcc_gain_over_frozen"] == -0.2
    assert report["headline_metric"] == "mcc"
    assert "not a benchmark" in report["note"]
    with pytest.raises(ValueError):
        evaluation_report(adapted, frozen, [])  # max() of an empty baseline list is a defect, not a verdict
