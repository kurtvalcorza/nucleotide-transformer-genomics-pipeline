"""Regression tests for the 2026-10-17 tutorial review of nucleotide_transformer_colab (row 58, T58-1..T58-8).

The notebook's own Section 4 / 7 / 8 cells are executed against stand-in pipelines (no model, no torch call), so the
pre-registration gate is tested as the learner runs it (SCORE_ONCE_RULE item 7)."""

# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from nucleotide_transformer_genomics_pipeline import protocol, samples
from nucleotide_transformer_genomics_pipeline.metrics import classification_metrics

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "nucleotide_transformer_colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell(nb: dict, marker: str) -> str:
    found = [c for c in _code(nb) if marker in c]
    assert len(found) == 1, marker
    return found[0]


# ---------------------------------------------------------------- data: T58-5, T58-6


def test_validation_is_locus_disjoint_from_train_and_train_and_test_are_unchanged():
    splits = samples.sample_dataset()
    assert samples.locus_overlaps(splits["validation"], splits["train"]) == []
    assert samples.dataset_digest(splits["train"]) == "3e4fe9685bab8554df451703ee65b288365f44d7a351ee9acc0a82f220c51810"
    assert samples.dataset_digest(splits["test"]) == "8384c09b5cba075000fe484f268adb6568c35acd602da8d47a31de36501fc747"
    assert samples.SAMPLE_SPLIT_DIGESTS["validation"] == "36da9e179a257c2c8a946f98ecf9de154d015007cf493a155e58f1320ef2a6c9"
    assert samples.SAMPLE_DIGEST == "96c002371dc813bc8b20edc7aa1d82103e99b69399946a6d963c2b8f4852c62b"


def test_exposed_test_overlap_is_what_the_notebook_states():
    splits = samples.sample_dataset()
    hits = {h[0] for h in samples.locus_overlaps(splits["test"], splits["train"] + splits["validation"])}
    assert len(hits) == 121 and sum(1 for r in splits["test"] if r["id"] in hits and r["label"] == 0) == 119
    assert len({h[0] for h in samples.locus_overlaps(splits["test"], splits["train"])}) == 104


def test_fresh_test_is_pinned_balanced_and_locus_disjoint_by_the_margin():
    fresh = samples.fresh_test_dataset()
    assert samples.FRESH_TEST_DIGEST == "338824e22a7d537a5a405e4f0b4757d9a5d92e2fc3f20b61fc14155c8a944c5d"
    assert len(fresh) == 400 and sum(r["label"] for r in fresh) == 200
    every = [r for part in samples.sample_dataset().values() for r in part]
    assert samples.locus_overlaps(fresh, every, margin=samples.LOCUS_MARGIN) == []
    assert samples.locus_overlaps(fresh, fresh, margin=samples.LOCUS_MARGIN) == []
    splits = {**samples.sample_dataset(), "fresh_test": fresh}
    samples.check_split_disjoint(splits, reverse_complement_too=True)


def test_fresh_test_refuses_an_edited_block(monkeypatch):
    monkeypatch.setattr(samples, "FRESH_TEST_DIGEST", "0" * 64)
    with pytest.raises(ValueError, match="edited"):
        samples.fresh_test_dataset()


def test_reverse_complement_duplicates_are_caught_when_asked():
    a = {"id": "a", "sequence": "AACCGGTTAC", "label": 1}
    b = {"id": "b", "sequence": samples.reverse_complement("AACCGGTTAC"), "label": 1}
    samples.check_split_disjoint({"x": [a], "y": [b]})  # exact-only by default (BYOD contract unchanged)
    with pytest.raises(ValueError, match="reverse complement"):
        samples.check_split_disjoint({"x": [a], "y": [b]}, reverse_complement_too=True)


def test_locus_overlaps_counts_overlap_and_margin():
    a = [{"id": "a", "region": "chr1", "start": 100, "end": 351}]
    b = [{"id": "b", "region": "chr1", "start": 300, "end": 551}, {"id": "c", "region": "chr1", "start": 500, "end": 751}, {"id": "d", "region": "chr2", "start": 100, "end": 351}]
    assert samples.locus_overlaps(a, b) == [("a", "b", 51)]
    assert {h[1] for h in samples.locus_overlaps(a, b, margin=251)} == {"b", "c"}
    with pytest.raises(ValueError, match="overlap"):
        samples.check_locus_disjoint({"x": a, "y": b}, [("x", "y")])


# ---------------------------------------------------------------- protocol: T58-2, T58-7


@pytest.mark.parametrize(
    ("mcc", "expected"),
    [
        ({0: 0.60, 2: 0.62, 12: 0.63}, 0),  # frozen within 0.03 of the best: kept
        ({0: 0.55, 2: 0.62, 12: 0.64}, 2),  # whole encoder ahead by < 0.03: keep 2
        ({0: 0.55, 2: 0.60, 12: 0.63}, 12),  # whole encoder ahead by exactly 0.03: beaten "by at least 0.03"
        ({0: 0.50, 2: 0.66, 12: 0.60}, 2),
        ({0: 0.50, 2: 0.55, 12: 0.70}, 12),
    ],
)
def test_select_trained_layers_follows_the_preregistered_rule(mcc, expected):
    results = {k: {"mcc": v, "accuracy": 0.8} for k, v in mcc.items()}
    assert protocol.select_trained_layers(results)["selected_layers"] == expected


def test_preregistration_names_every_differing_setting():
    base = dict(protocol.PREREGISTERED_SETTINGS)
    assert protocol.preregistration_check(base) == {"preregistered": True, "settings_differing": []}
    for key, value in (("epochs", 7), ("batch_size", 8), ("seed", 1), ("learning_rate", 1e-4), ("data_source", "BYOD"), ("selection_grid", False), ("trained_layers", 12), ("split_seed", 7)):
        check = protocol.preregistration_check({**base, key: value})
        assert check == {"preregistered": False, "settings_differing": [key]}, key
    assert protocol.preregistration_check({k: v for k, v in base.items() if k != "seed"})["settings_differing"] == ["seed"]


def test_test_look_labels_and_exploratory_suffix():
    assert protocol.test_look(False) == "not scored (exploratory)"
    assert protocol.test_look(True, 0, recorded_evaluation=None) == "pre-registered evaluation (this runtime)"
    assert protocol.test_look(True, 0) == "reproduction of the pre-registered evaluation"
    assert protocol.test_look(True, 1) == "reproduction (this runtime)"
    assert protocol.test_look(True, 0, state_matches=False) == "not scored (model-state check failed)"
    assert protocol.output_suffix("s", [], protocol.LOOK_PREREGISTERED) == ""
    assert protocol.output_suffix("s", ["s_evaluation_report_reproduction_1.json"], protocol.LOOK_REPRODUCTION) == "_reproduction_2"
    assert protocol.output_suffix("s", [], protocol.LOOK_STATE_FAILED) == "_exploratory_1"
    stem = "nucleotide_transformer"
    assert protocol.exploratory_suffix(stem, []) == "_exploratory_1"
    assert protocol.exploratory_suffix(stem, [f"{stem}_evaluation_report_exploratory_1.json", f"{stem}_result_exploratory_2.json"]) == "_exploratory_3"


def test_no_permanent_scored_once_wording(nb):
    text = (_markdown(nb) + "\n".join(_code(nb))).lower()
    assert "scored once" not in text and "score_once" not in text.replace("score_once_rule", "")


# ---------------------------------------------------------------- the notebook's own cells against stand-ins


def _modules(nb: dict) -> dict:
    ns: dict = {"__name__": "__main__", "os": os, "torch": type("TorchStandIn", (), {"__version__": "stand-in"})}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and cell.get("metadata", {}).get("dimer", {}).get("embedded_module"):
            exec(compile("".join(cell["source"]), "<module>", "exec"), ns)
    return ns


class _Pipe:
    """Stand-in pipeline: records which sets it scores; perfect predictions; never touches torch."""

    encoder_is_base = True
    adapter = None
    device = "cpu"

    def __init__(self):
        self.scored: list[str] = []

    def predict(self, sequences, batch_size=16):
        return {"labels": [0] * len(sequences), "probabilities": [[1.0, 0.0]] * len(sequences)}

    def evaluate(self, records, batch_size=16):
        self.scored.append("fresh" if records and records[0].get("split") == "fresh_test" else "other")
        scored = classification_metrics([r["label"] for r in records], [r["label"] for r in records])
        if self.adapter and len(records) == 200:  # validation: the kept epoch's numbers, as the real model reproduces them
            scored.update({k: self.adapter["history"][self.adapter["best_epoch"]]["val"][k] for k in ("accuracy", "mcc")})
        return scored

    def linear_probe(self, train, test):
        self.scored.append("fresh-probe" if test and test[0].get("split") == "fresh_test" else "probe")
        return {**classification_metrics([r["label"] for r in test], [r["label"] for r in test]), "train_accuracy": 1.0, "probe": "stand-in", "predictions": [r["label"] for r in test]}

    def save_artifact(self, path, metadata=None):
        Path(path).mkdir(parents=True, exist_ok=True)

    def reset_to_base(self):
        self._parked, self.adapter = self.adapter, None
        return []

    def load_artifact(self, path):
        self.adapter = self._parked
        return dict(self.adapter)

    def adapt(self, train, val, *, epochs, lr, layers, seed, progress=None):
        mcc = {0: 0.50, 2: 0.60, 12: 0.61}.get(layers, 0.4)
        self.adapter = {"history": [{"epoch": 0, "train_loss": None, "val": {"accuracy": 0.5, "mcc": 0.0, "n": len(val)}}, {"epoch": 1, "train_loss": 0.4, "val": {"accuracy": 0.8, "mcc": mcc, "n": len(val)}}], "best_epoch": 1, "n_trainable": 1, "n_head": 263682, "n_total": 2, "layers": layers, "selection": "highest validation MCC (ties: accuracy, earlier epoch)", "loss": "cross-entropy over the two labels", "train_accuracy": 1.0, "batch_size": 16, "epochs": epochs, "lr": lr, "seed": seed, "n_train": len(train)}
        return dict(self.adapter)


def _run_sections(nb, tmp_path, monkeypatch, overrides=None, ns=None):
    monkeypatch.chdir(tmp_path)
    ns = ns if ns is not None else _modules(nb)
    # Rule 8: the tests never score the real fresh-test labels, not even with the GC rule.
    for name in ("majority_baseline", "gc_threshold_baseline"):
        real = ns[name]
        ns[name] = (lambda fn: lambda train, test: {**classification_metrics([r["label"] for r in test], [r["label"] for r in test]), "baseline": "stand-in", "threshold": 0.5, "rule": "GC >= 0.5 -> promoter", "train_accuracy": 1.0, "label": 1} if test and test[0].get("split") == "fresh_test" else fn(train, test))(real)
    sec4 = _cell(nb, "USE_BYOD = False  # @param")
    exec(compile(sec4, "<section 4>", "exec"), ns)
    ns["pipe"] = ns.get("pipe") or _Pipe()
    sec6 = _cell(nb, "METRICS = ('accuracy', 'mcc')")
    exec(compile(sec6, "<section 6>", "exec"), ns)
    sec7 = _cell(nb, "RUN_SELECTION_GRID = True  # @param")
    for name, value in (overrides or {}).items():
        sec7 = re.sub(rf"^{name} = .*?  # @param", f"{name} = {value!r}  # @param", sec7, flags=re.M)
    exec(compile(sec7, "<section 7>", "exec"), ns)
    sec8 = _cell(nb, "fresh_adapted = pipe.evaluate(fresh_records)")
    exec(compile(sec8, "<section 8>", "exec"), ns)
    return ns


def test_default_settings_score_the_fresh_test_and_a_rerun_is_a_reproduction(nb, tmp_path, monkeypatch):
    ns = _run_sections(nb, tmp_path, monkeypatch)
    assert ns["protocol_check"] == {"preregistered": True, "settings_differing": []}
    assert ns["grid"]["selected_layers"] == 2  # the stand-in's 0.50 / 0.60 / 0.61: frozen beaten by 0.11, 12 ahead of 2 by only 0.01
    report = json.loads((tmp_path / "outputs" / "nucleotide_transformer_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["test_look"] == "reproduction of the pre-registered evaluation" and report["fresh_test"]["n"] == 400
    assert report["preregistered_evaluation_record"] == protocol.PREREGISTERED_EVALUATION_RECORD
    assert "fresh" in ns["pipe"].scored and "fresh-probe" in ns["pipe"].scored
    assert report["fresh_test"]["mcc_difference_adapted_minus_probe"]["resamples"] == 2000
    ns = _run_sections(nb, tmp_path, monkeypatch, ns=ns)
    first = json.loads((tmp_path / "outputs" / "nucleotide_transformer_evaluation_report.json").read_text(encoding="utf-8"))
    assert first["test_look"] == "reproduction of the pre-registered evaluation"  # m4: a reproduction never overwrites it
    again = json.loads((tmp_path / "outputs" / "nucleotide_transformer_evaluation_report_reproduction_1.json").read_text(encoding="utf-8"))
    assert again["test_look"] == "reproduction (this runtime)" and ns["PREREGISTERED_SCORINGS"] == 2


@pytest.mark.parametrize(
    ("overrides", "differing"),
    [({"EPOCHS": 5}, ["epochs"]), ({"LEARNING_RATE": 1e-4}, ["learning_rate"]), ({"SEED": 1}, ["seed"]), ({"RUN_SELECTION_GRID": False}, ["selection_grid", "grid_layers", "trained_layers"]), ({"TRAINED_LAYERS": 12}, ["trained_layers"])],
)
def test_a_changed_setting_never_scores_the_fresh_test(nb, tmp_path, monkeypatch, overrides, differing):
    ns = _run_sections(nb, tmp_path, monkeypatch, overrides)
    assert ns["protocol_check"] == {"preregistered": False, "settings_differing": differing}
    assert "fresh" not in ns["pipe"].scored and "fresh-probe" not in ns["pipe"].scored
    assert not (tmp_path / "outputs" / "nucleotide_transformer_evaluation_report.json").exists()
    report = json.loads((tmp_path / "outputs" / "nucleotide_transformer_evaluation_report_exploratory_1.json").read_text(encoding="utf-8"))
    assert report["test_look"] == "not scored (exploratory)" and report["fresh_test"] is None and report["settings_differing"] == differing


def test_a_changed_batch_size_never_scores_the_fresh_test(nb, tmp_path, monkeypatch):
    ns = _modules(nb)
    ns["pipe"] = _Pipe()
    real_adapt = ns["pipe"].adapt

    def adapt_batch_8(*args, **kwargs):  # the OOM tip: a smaller batch
        result = real_adapt(*args, **kwargs)
        ns["pipe"].adapter["batch_size"] = 8
        return {**result, "batch_size": 8}

    ns["pipe"].adapt = adapt_batch_8
    ns = _run_sections(nb, tmp_path, monkeypatch, ns=ns)
    assert ns["protocol_check"] == {"preregistered": False, "settings_differing": ["batch_size"]}
    assert "fresh" not in ns["pipe"].scored


def test_paired_bootstrap_and_distance_summary():
    gold = [1, 0] * 50
    same = protocol.paired_bootstrap_mcc_difference(gold, gold, gold, resamples=200)
    assert same["difference"] == 0.0 and same["ci95"] == [0.0, 0.0]
    better = protocol.paired_bootstrap_mcc_difference(gold, gold, [1] * 100, resamples=200)
    assert better["difference"] == 1.0 and better["ci95"][0] > 0.5
    fresh = samples.fresh_test_dataset()
    summary = samples.nearest_distance_summary(fresh, samples.sample_dataset()["train"])
    assert summary["0"]["median"] == 612 and summary["0"]["within_1kb"] == 134 and summary["1"]["within_1kb"] == 1


def test_a_changed_data_source_never_scores_the_fresh_test(nb, tmp_path, monkeypatch):
    ns = _modules(nb)
    ns["sample_dataset"] = lambda: {k: v for k, v in samples.sample_dataset().items()} | {"validation": samples.sample_dataset()["validation"][:150]}
    ns = _run_sections(nb, tmp_path, monkeypatch, ns=ns)
    assert ns["protocol_check"]["preregistered"] is False and "sample_digest" in ns["protocol_check"]["settings_differing"]
    assert "fresh" not in ns["pipe"].scored


def test_fresh_test_is_not_scored_when_the_model_in_memory_is_not_the_selected_one(nb, tmp_path, monkeypatch):
    ns = _modules(nb)
    ns["pipe"] = _Pipe()
    real_adapt = ns["pipe"].adapt

    def adapt_then_drift(*args, **kwargs):
        result = real_adapt(*args, **kwargs)
        ns["pipe"].adapter = {**ns["pipe"].adapter, "best_epoch": 0}  # e.g. a later cell changed the model in memory
        return result

    ns["pipe"].adapt = adapt_then_drift
    ns = _run_sections(nb, tmp_path, monkeypatch, ns=ns)
    assert ns["protocol_check"]["preregistered"] is True and ns["state"]["matches"] is False
    assert ns["look"] == "not scored (model-state check failed)"
    assert "fresh" not in ns["pipe"].scored and ns["fresh_result"] is None


def test_adapted_state_check_unit():
    result = {"layers": 2, "best_epoch": 1, "epochs": 6, "lr": 3e-5, "seed": 0, "batch_size": 16, "n_train": 10, "history": [{"val": None}, {"val": {"accuracy": 0.8, "mcc": 0.6}}]}
    assert protocol.adapted_state_check(dict(result), result, {"accuracy": 0.8, "mcc": 0.6})["matches"] is True
    assert protocol.adapted_state_check(None, result, {"accuracy": 0.8, "mcc": 0.6})["matches"] is False
    assert protocol.adapted_state_check({**result, "layers": 12}, result, {"accuracy": 0.8, "mcc": 0.6})["matches"] is False
    assert protocol.adapted_state_check(dict(result), result, {"accuracy": 0.8, "mcc": 0.59})["matches"] is False


def test_byod_refusal_is_one_message_and_later_cells_skip(nb, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ns = _modules(nb)
    sec4 = _cell(nb, "USE_BYOD = False  # @param").replace("USE_BYOD = False", "USE_BYOD = True").replace("BYOD_PATH = ''", "BYOD_PATH = 'missing.csv'")
    with pytest.raises(ValueError) as caught:
        exec(compile(sec4, "<section 4>", "exec"), ns)
    out = capsys.readouterr().out
    assert caught.value._render_traceback_()[0].startswith("Refused: missing.csv: BYOD path ")
    assert "Traceback" not in out and ns["RUN_STATE"]["refused"]
    status = json.loads((tmp_path / "outputs" / "nucleotide_transformer_run_status.json").read_text(encoding="utf-8"))
    assert status["status"].startswith("invalidated")
    exec(compile(_cell(nb, "METRICS = ('accuracy', 'mcc')"), "<section 6>", "exec"), ns)
    assert "skipped" in capsys.readouterr().out and "baseline_gc" not in ns


# ---------------------------------------------------------------- learner-facing text: T58-1..T58-5


def test_markdown_states_the_pretraining_overlap_and_the_exposed_test(nb):
    md = _markdown(nb)
    assert re.search(r"GRCh38\.p14.{0,400}pre-train|multi_species_genomes.{0,400}GRCh38", md, re.S)
    assert "**exposed development test**" in md and "**fresh test**" in md
    assert "4,449" in md  # the benchmark's own train/test locus overlap (T58-5)
    for stale in ("without leakage", "validation, sequence-disjoint split", "on a sequence-disjoint split", "Kaggle T4 run of 20 September", "0.7725", "640 windows", "@@"):
        assert stale not in md, stale
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "unknown, not ruled out" not in card and "GRCh38.p14" in card


def test_spec_2_5_kind_use_notice_and_isolated_env_documented(nb):
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    md = _markdown(nb)
    assert "> **⚠️ Notice.** For research, training and evaluation only." in md
    assert "`DIMER_ISOLATED_ENV`" in md
    assert "[1] H. Dalla-Torre" in md


# ---------------------------------------------------------------- provenance: T58-8


def test_stamped_revision_contains_the_carried_modules(nb):
    """`generated_from.revision` must name a commit whose package sources are the carried ones: the notebook is
    regenerated in a separate commit after the code commit. Skips only when that commit is not in the checkout."""
    meta = nb["metadata"]["dimer"]["generated_from"]
    revision = meta["revision"]
    try:
        present = subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=ROOT, capture_output=True).returncode == 0
    except OSError:
        present = False
    if not present:
        pytest.skip(f"stamped revision {revision} not in this checkout")
    text = "".join(subprocess.run(["git", "show", f"{revision}:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8") for rel in meta["modules"])
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == meta["module_sha256"], f"carried modules differ from {revision}"


def test_the_evaluation_of_record_is_named_and_a_later_identical_run_is_a_reproduction():
    """SCORE_ONCE_RULE items 3 and 7: the first pre-registered hosted run (Colab CLI T4, 2026-10-17, commit d606232,
    notebook blob b28c43a7) is THE evaluation; a later run with identical settings labels itself a reproduction."""
    record = protocol.PREREGISTERED_EVALUATION_RECORD
    assert record is not None
    for part in ("docs/release-verification.md", "Colab CLI T4", "2026-10-17", "d606232", "b28c43a7"):
        assert part in record, part
    assert protocol.test_look(True) == protocol.LOOK_RECORDED_REPRODUCTION == "reproduction of the pre-registered evaluation"
    assert protocol.test_look(True, 0, recorded_evaluation=None) == protocol.LOOK_PREREGISTERED
    assert protocol.test_look(True, 1) == protocol.LOOK_REPRODUCTION
    assert protocol.test_look(False) == protocol.LOOK_EXPLORATORY  # a changed setting never scores, record or not
    assert protocol.test_look(True, 0, state_matches=False) == protocol.LOOK_STATE_FAILED
    assert protocol.output_suffix("s", [], protocol.LOOK_RECORDED_REPRODUCTION) == ""
    assert protocol.output_suffix("s", ["s_evaluation_report.json"], protocol.LOOK_REPRODUCTION) == "_reproduction_1"
    doc = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "d606232" in doc and "b28c43a7" in doc


def test_notebook_quotes_the_fresh_test_result_not_the_exposed_gain(nb):
    """Row 58 requote: the conclusion states the fresh-test result and its unresolved interval; timings are measured."""
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "gained 0.06 MCC" not in text and "2–3 min" not in text and "0.6628 (epoch 4)" not in text
    assert "CPU build record of this" not in text
    for quote in ("0.664", "0.636", "−0.028 to +0.086", "not resolved", "12 blocks 0.6621 (epoch 1)", "138.6 s", "252.1 s"):
        assert quote in text, quote
