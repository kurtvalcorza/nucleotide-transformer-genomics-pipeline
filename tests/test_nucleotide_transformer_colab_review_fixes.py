"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on nucleotide_transformer_colab
(NTP-M2 in-place adaptation, NTP-m1 BYOD minimum, NTP-m3 spread claims, NTP-m5 CPU timing, NTP-m6 load summary,
spec 2.2 declaration). NTP-M1 and NTP-M3 are covered by tests/test_fleet_sweep_fixes.py (SWP-R, SWP-A).

The NTP-M2 package tests run a tiny stand-in encoder with the checkpoint's parameter naming
(`esm.encoder.layer.<i>.`), so they need torch (CI installs it) but no weights; the same guarantee on the real
weights is in tests/test_model_backed.py."""
# ruff: noqa: E501

from __future__ import annotations

import copy
import json
import random
import types
from pathlib import Path

import pytest

from nucleotide_transformer_genomics_pipeline import (
    HIDDEN_SIZE,
    MIN_RECORDS,
    NUM_LAYERS,
    NucleotideTransformerPipeline,
    split_dataset,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "nucleotide_transformer_colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code(nb: dict) -> list[str]:
    return [_src(c) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _records(n: int, seed: int = 0) -> list[dict]:
    # learnable from the first 12 bases (the stand-in tokenizer reads only those): label 1 = A/C-rich start
    rng = random.Random(seed)
    out = []
    for i in range(n):
        start = "".join(rng.choice("AC" if i % 2 else "GT") for _ in range(12))
        out.append({"id": f"r{i}", "sequence": start + "".join(rng.choice("ACGT") for _ in range(28)), "label": i % 2})
    return out


# --- NTP-M2: adapt() never continues from an earlier adaptation ---------------------------------------------------------


def _stand_in_pipeline():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class _Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.ModuleList(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(NUM_LAYERS))

    class _Esm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.Embedding(8, HIDDEN_SIZE)
            self.encoder = _Encoder()

        def forward(self, input_ids, attention_mask):
            hidden = self.embeddings(input_ids)
            for block in self.encoder.layer:
                hidden = torch.tanh(block(hidden)) * 0.5 + hidden * 0.5
            return types.SimpleNamespace(last_hidden_state=hidden)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.esm = _Esm()

    class _Batch(dict):
        def to(self, device):
            return self

    class _Tokenizer:
        def __call__(self, sequences, return_tensors=None, padding=None):
            ids = torch.tensor([["ACGTN".index(c) for c in s[:12]] for s in sequences])
            return _Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

    torch.manual_seed(0)
    model = _Model().eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return torch, model, _Tokenizer()


def _pipe(model, tokenizer):
    return NucleotideTransformerPipeline("cpu", model, tokenizer, None, "0" * 64)


def _blocks(model, layer: int) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(f"esm.encoder.layer.{layer}.")}


def test_ntp_m2_rerun_starts_from_the_pinned_base_and_probe_refuses_an_adapted_encoder() -> None:
    torch, model, tokenizer = _stand_in_pipeline()
    pristine = copy.deepcopy(model)
    pipe = _pipe(model, tokenizer)
    train, val = _records(16), _records(8, seed=1)
    sequences = [r["sequence"] for r in val]
    base11 = _blocks(model, 11)
    before = pipe.embed(sequences)
    assert pipe.encoder_is_base

    pipe.adapt(train, None, epochs=2, lr=1e-2, layers=2, seed=3)  # no validation split: the final epoch is kept
    assert not pipe.encoder_is_base
    assert any(not torch.equal(base11[k], model.state_dict()[k]) for k in base11), "two-block training changed block 11"
    with pytest.raises(RuntimeError, match="reset_to_base"):
        pipe.linear_probe(train, val)

    # the review's acceptance check: a head-only re-run equals the same run on a fresh pipeline
    rerun = pipe.adapt(train, val, epochs=2, lr=1e-2, layers=0, seed=3)
    assert all(torch.equal(base11[k], model.state_dict()[k]) for k in base11), "layers=0 re-run left block 11 trained"
    assert pipe.encoder_is_base
    fresh = _pipe(copy.deepcopy(pristine), tokenizer).adapt(train, val, epochs=2, lr=1e-2, layers=0, seed=3)
    assert rerun["history"] == fresh["history"] and rerun["n_trainable"] == fresh["n_trainable"]
    assert pipe.embed(sequences) == before

    # a two-block re-run equals a fresh two-block run too (it does not continue from the first adaptation)
    pipe.adapt(train, None, epochs=2, lr=1e-2, layers=2, seed=3)
    again = pipe.adapt(train, None, epochs=2, lr=1e-2, layers=2, seed=3)
    fresh2 = _pipe(copy.deepcopy(pristine), tokenizer).adapt(train, None, epochs=2, lr=1e-2, layers=2, seed=3)
    assert again["history"] == fresh2["history"]

    restored = pipe.reset_to_base()
    assert restored and pipe.adapter is None and pipe._head is None and pipe.encoder_is_base
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in pristine.state_dict().items())
    assert pipe.embed(sequences) == before
    assert pipe.reset_to_base() == []
    pipe.linear_probe(train, val)  # accepted again on the base encoder


def test_ntp_m2_failed_rerun_returns_to_its_entry_state() -> None:
    torch, model, tokenizer = _stand_in_pipeline()
    pipe = _pipe(model, tokenizer)
    train, val = _records(16), _records(8, seed=1)
    pipe.adapt(train, None, epochs=1, lr=1e-2, layers=2, seed=3)
    adapted = {k: v.detach().clone() for k, v in model.state_dict().items()}
    adapter, head, tracked = pipe.adapter, pipe._head, set(pipe._base_tensors)

    def explode(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(train, val, epochs=2, lr=1e-2, layers=1, seed=3, progress=explode)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in adapted.items())
    assert pipe.adapter is adapter and pipe._head is head and set(pipe._base_tensors) == tracked
    assert all(not p.requires_grad for p in model.parameters())


# --- notebook-side fixes -----------------------------------------------------------------------------------------------


def _section4_validation(nb: dict) -> str:
    source = next(s for s in _code(nb) if "def validate_split(name, part):" in s)
    start = source.index("BYOD_MIN_SEQUENCES = ")
    end = source.index("disjoint = check_split_disjoint(splits)")
    return source[start:end]


@pytest.mark.parametrize("n", [8, 10, 11])
def test_ntp_m1_too_small_byod_names_the_split_and_the_minimum(nb: dict, n: int) -> None:
    ns = {"validate_dataset": validate_dataset, "MIN_RECORDS": MIN_RECORDS, "splits": split_dataset(_records(n), seed=42)}
    with pytest.raises(ValueError, match=r"train split \(\d+ records\).*at least 12 distinct sequences"):
        exec(compile(_section4_validation(nb), "<section4>", "exec"), ns)


@pytest.mark.parametrize("n", [12, 13, 20, 49, 50])
def test_ntp_m1_stated_minimum_is_the_smallest_accepted_dataset(nb: dict, n: int) -> None:
    records = _records(n)
    ns = {"validate_dataset": validate_dataset, "MIN_RECORDS": MIN_RECORDS, "splits": split_dataset(records, seed=42)}
    exec(compile(_section4_validation(nb), "<section4>", "exec"), ns)
    assert ns["BYOD_MIN_SEQUENCES"] == 12
    assert sum(len(v) for v in ns["splits"].values()) == n
    markdown = _markdown(nb)
    assert "at least 12 distinct sequences" in markdown and "at least eight sequences" not in markdown


def test_ntp_m3_no_unrecorded_spread_claims(nb: dict) -> None:
    markdown = _markdown(nb)
    for phrase in ("run-to-run spread", "within a few hundredths", "from one GPU run to the next"):
        assert phrase not in markdown, phrase
    assert "spread across seeds has **not been measured**" in markdown
    assert markdown.count("**untested**") >= 2  # Section 8 worked answer and the interpretation


def test_ntp_m5_cpu_timing_is_labelled_an_untested_estimate(nb: dict) -> None:
    markdown = _markdown(nb)
    assert "a few tens of minutes on a 2-vCPU hosted runtime is an **untested estimate**" in markdown
    assert "the verified hosted runtime is a **T4 GPU**" in markdown


def test_ntp_m6_section3_prints_observed_values_only(nb: dict) -> None:
    load = next(s for s in _code(nb) if "pipe = NucleotideTransformerPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)" in s)
    assert "getattr(pipe, 'source'" not in load
    for observed in ("snapshot['revision']", "str(WEIGHTS_DIR)", "pipe.weight_sha256", "pipe.encoder_is_base"):
        assert observed in load, observed


def test_ntp_m4_declares_spec_2_2(nb: dict) -> None:
    assert nb["metadata"]["dimer"].get("notebook_spec", nb["metadata"]["dimer"].get("notebook_spec_version")) == "2.2"
