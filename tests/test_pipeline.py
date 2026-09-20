"""Offline tests for the snapshot perimeter and the sequence validators (no model, no network)."""
# ruff: noqa: E501

import hashlib
import json
import re
from pathlib import Path

import pytest

from nucleotide_transformer_genomics_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    MAX_BASES,
    MAX_SEQUENCES,
    MIN_BASES,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    REMOTE_CODE_FILES,
    WEIGHT_FILE,
    NucleotideTransformerPipeline,
    stage_missing_files,
    validate_inputs,
    validate_sequences,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    assert MODEL_LICENSE == "cc-by-nc-sa-4.0"
    assert REMOTE_CODE_FILES == ("modeling_esm.py", "esm_config.py")
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["modelId"] == MODEL_ID
    assert data["revision"] == MODEL_REVISION
    listed = {entry["path"] for entry in data["files"]}
    assert {WEIGHT_FILE, *REMOTE_CODE_FILES} <= listed, "the remote code must be inside the verified perimeter"


def _write_snapshot(root: Path, *, drop: str | None = None, corrupt: str | None = None, size_of: str | None = None) -> dict:
    files = {"config.json": b'{"auto_map": {}}', WEIGHT_FILE: b"weights", "modeling_esm.py": b"class EsmForMaskedLM: ...\n", "esm_config.py": b"class EsmConfig: ...\n"}
    entries = []
    for name, content in files.items():
        if name != drop:
            (root / name).write_bytes(content if name != corrupt else content + b"!")
        entries.append({"path": name, "bytes": len(content) + (1 if name == size_of else 0), "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": entries, "totalBytes": sum(len(c) for c in files.values())}
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_verify_snapshot_accepts_matching_manifest_and_names_the_remote_code(tmp_path):
    _write_snapshot(tmp_path)
    info = verify_snapshot(tmp_path)
    assert info["files"] == 4 and info["revision"] == MODEL_REVISION
    assert info["remote_code_files"] == list(REMOTE_CODE_FILES)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"drop": "modeling_esm.py"}, FileNotFoundError, "missing"),
        ({"corrupt": "esm_config.py"}, ValueError, "size"),
        ({"size_of": WEIGHT_FILE}, ValueError, "size"),
    ],
)
def test_verify_snapshot_refuses_missing_or_altered_files(tmp_path, kwargs, error, match):
    _write_snapshot(tmp_path, **kwargs)
    with pytest.raises(error, match=match):
        verify_snapshot(tmp_path)


def test_verify_snapshot_refuses_a_digest_mismatch(tmp_path):
    manifest = _write_snapshot(tmp_path)
    manifest["files"][2]["sha256"] = "0" * 64
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_manifest_must_list_the_remote_code(tmp_path):
    manifest = _write_snapshot(tmp_path)
    manifest["files"] = [e for e in manifest["files"] if e["path"] != "modeling_esm.py"]
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="verified perimeter"):
        verify_snapshot(tmp_path)
    with pytest.raises(ValueError, match="verified perimeter"):
        stage_missing_files(tmp_path, allow_download=True)


def test_manifest_identity_mismatch_is_refused(tmp_path):
    manifest = _write_snapshot(tmp_path)
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(ValueError, match="revision"):
        stage_missing_files(tmp_path, allow_download=True)


def test_stage_missing_files_fetches_only_absent_entries_through_the_injected_downloader(tmp_path):
    _write_snapshot(tmp_path, drop="esm_config.py")
    fetched = []

    def downloader(relative_path: str, root: Path) -> None:
        fetched.append(relative_path)
        (root / relative_path).write_bytes(b"class EsmConfig: ...\n")

    with pytest.raises(FileNotFoundError, match="allow_download"):
        stage_missing_files(tmp_path)
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == ["esm_config.py"]
    assert fetched == ["esm_config.py"]
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == []
    verify_snapshot(tmp_path)


def test_from_pretrained_refuses_without_a_manifest_before_importing_model_libraries(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="manifest"):
        NucleotideTransformerPipeline.from_pretrained(weights_dir=tmp_path, allow_download=True)


def test_validate_sequences_upper_cases_and_bounds():
    assert validate_sequences(["acgtn" * 3]) == ["ACGTN" * 3]
    with pytest.raises(ValueError, match="list of str"):
        validate_sequences("ACGTACGTACGT")
    with pytest.raises(ValueError, match="must be a str"):
        validate_sequences([12])
    with pytest.raises(ValueError, match="bases"):
        validate_sequences(["A" * (MIN_BASES - 1)])
    with pytest.raises(ValueError, match="bases"):
        validate_sequences(["A" * (MAX_BASES + 1)])
    with pytest.raises(ValueError, match="outside"):
        validate_sequences(["ACGTACGTACGU"])
    with pytest.raises(ValueError, match="per call"):
        validate_sequences(["ACGT" * 4] * (MAX_SEQUENCES + 1))


def test_validate_inputs_builds_the_input_manifest(forbid_model_imports):
    manifest = validate_inputs(["ACGT" * 10, "GGCCN" * 5], names=["a", "b"])
    assert manifest["n_sequences"] == 2
    assert manifest["bases"] == {"min": 25, "max": 40, "total": 65}
    assert manifest["n_with_N"] == 1
    assert len(manifest["sequence_sha256"][0]) == 64
    assert manifest["model_revision"] == MODEL_REVISION
    with pytest.raises(ValueError, match="names"):
        validate_inputs(["ACGT" * 10], names=["a", "b"])


def test_inference_methods_refuse_without_a_model(forbid_model_imports):
    pipe = NucleotideTransformerPipeline("cpu")
    with pytest.raises(RuntimeError, match="no model loaded"):
        pipe.embed(["ACGT" * 10])
    with pytest.raises(RuntimeError, match="no model loaded"):
        pipe.predict(["ACGT" * 10])
    with pytest.raises(RuntimeError, match="no model loaded"):
        pipe.token_counts(["ACGT" * 10])
    with pytest.raises(ValueError, match="outside"):
        pipe.embed(["XXXXXXXXXXXX"])  # input validation precedes the model check
