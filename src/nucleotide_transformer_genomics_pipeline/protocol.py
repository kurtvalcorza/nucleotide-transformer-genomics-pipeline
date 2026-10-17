"""The pre-registered evaluation protocol (docs/nucleotide-transformer-preregistration.md, 2026-10-17).

Three things live here, all pure Python so CI can test them without a model:

* ``PREREGISTERED_SETTINGS`` - every setting the pre-registration fixes. ``preregistration_check`` compares the
  settings a run actually used with ALL of them and returns ``preregistered`` plus the list of settings that
  differ. Only a pre-registered run may score the fresh test; every other run is exploratory and is compared on
  validation (and on the labelled exposed development test) only.
* ``select_trained_layers`` - the pre-registered validation grid's selection rule over ``GRID_LAYERS``.
* ``test_look`` - the label a fresh-test score carries: "pre-registered evaluation (this runtime)", "reproduction
  (this runtime)" or "not scored (exploratory)"; ``adapted_state_check`` - the model in memory is the selected one.
"""

from __future__ import annotations

# ruff: noqa: E501  -- contract lines are kept at the fleet width
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import DEFAULT_BATCH_SIZE, DEFAULT_TRAINED_LAYERS
from .samples import (
    FRESH_TEST_DIGEST,
    FRESH_TEST_SIZE,
    SAMPLE_DIGEST,
    SAMPLE_SEED,
    SAMPLE_SPLIT,
    SAMPLE_SPLIT_DIGESTS,
)

PREREGISTRATION_DOC = "docs/nucleotide-transformer-preregistration.md"
PREREGISTRATION_DATE = "2026-10-17"
GRID_LAYERS = (0, 2, 12)  # head only on the frozen encoder; the default two blocks; the whole encoder
GRID_INCUMBENT = DEFAULT_TRAINED_LAYERS  # 2: kept unless beaten by at least GRID_TOLERANCE validation MCC
GRID_TOLERANCE = 0.03  # a practical tolerance (the noise of a 200-window validation MCC), not a significance test
# A runtime cannot know about other runtimes (SCORE_ONCE_RULE item 8): which run is THE evaluation of record, and which
# runs are its reproductions, is recorded in docs/release-verification.md, never decided in the notebook.
RECORD_OF_EVALUATIONS = "docs/release-verification.md"
LOOK_PREREGISTERED = "pre-registered evaluation (this runtime)"
LOOK_REPRODUCTION = "reproduction (this runtime)"
LOOK_EXPLORATORY = "not scored (exploratory)"

PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data_source": "pinned promoter sample",
    "sample_digest": SAMPLE_DIGEST,
    "split_digests": dict(SAMPLE_SPLIT_DIGESTS),
    "split_sizes": dict(SAMPLE_SPLIT),
    "sample_seed": SAMPLE_SEED,
    "split_seed": 42,  # the notebook's SPLIT_SEED form field (used by a BYOD split; fixed here all the same)
    "fresh_test_digest": FRESH_TEST_DIGEST,
    "fresh_test_size": FRESH_TEST_SIZE,
    "epochs": 6,
    "learning_rate": 3e-5,
    "batch_size": DEFAULT_BATCH_SIZE,
    "seed": 0,
    "head": "mean-pooled (512 -> 512 tanh -> 2)",
    "optimizer": "AdamW, weight decay 0, gradient clipping 1.0, no scheduler",
    "epoch_selection": "highest validation MCC (ties: accuracy, earlier epoch)",
    "selection_grid": True,
    "grid_layers": list(GRID_LAYERS),
    "trained_layers": "grid selection",
}


def preregistration_check(settings: Mapping[str, Any]) -> dict[str, Any]:
    """``preregistered`` is True only when every pre-registered setting is present in ``settings`` with the
    pre-registered value; ``settings_differing`` names each one that is missing or different."""
    differing = []
    for key, expected in PREREGISTERED_SETTINGS.items():
        if key not in settings:
            differing.append(key)
            continue
        actual = settings[key]
        if isinstance(expected, float) or isinstance(actual, float):
            same = isinstance(actual, int | float) and not isinstance(actual, bool) and abs(float(actual) - float(expected)) <= 1e-12 * max(1.0, abs(float(expected)))
        elif isinstance(expected, list | tuple):
            same = isinstance(actual, list | tuple) and list(actual) == list(expected)
        else:
            same = type(actual) is type(expected) and actual == expected
        if not same:
            differing.append(key)
    return {"preregistered": not differing, "settings_differing": differing}


def select_trained_layers(results: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """The pre-registered grid rule over best-epoch validation results ``{layers: {"mcc", "accuracy", ...}}``.

    1. The head-only configuration on the frozen encoder (0 blocks) is kept unless another configuration beats its
       validation MCC by more than ``GRID_TOLERANCE``.
    2. Otherwise the incumbent (2 blocks) is kept unless a configuration with more blocks beats it by at least
       ``GRID_TOLERANCE``; when several do, the best of them is taken, and among configurations within the
       tolerance of that best one the smallest trainable scope wins.
    """
    if sorted(results) != sorted(GRID_LAYERS):
        raise ValueError(f"the grid needs exactly the configurations {GRID_LAYERS}, got {sorted(results)}")
    mcc = {layers: float(results[layers]["mcc"]) for layers in GRID_LAYERS}
    best = max(mcc.values())
    smallest = min(GRID_LAYERS)
    if best - mcc[smallest] <= GRID_TOLERANCE + 1e-12:
        selected, reason = smallest, f"the frozen encoder (head only) is not beaten by more than {GRID_TOLERANCE} validation MCC"
    else:
        challengers = [layers for layers in GRID_LAYERS if layers > GRID_INCUMBENT and mcc[layers] >= mcc[GRID_INCUMBENT] + GRID_TOLERANCE - 1e-12]
        if not challengers:
            selected, reason = GRID_INCUMBENT, f"no larger configuration beats {GRID_INCUMBENT} blocks by at least {GRID_TOLERANCE} validation MCC"
        else:
            top = max(mcc[layers] for layers in challengers)
            selected = min(layers for layers in challengers if top - mcc[layers] <= GRID_TOLERANCE + 1e-12)
            reason = f"{selected} blocks: beats {GRID_INCUMBENT} blocks by at least {GRID_TOLERANCE} validation MCC (smallest scope within the tolerance of the best)"
    return {"selected_layers": selected, "reason": reason, "validation_mcc": {str(k): round(v, 4) for k, v in mcc.items()}, "tolerance": GRID_TOLERANCE, "incumbent": GRID_INCUMBENT}


def test_look(preregistered: bool, earlier_preregistered_scorings: int = 0) -> str:
    """The label a fresh-test score carries (SCORE_ONCE_RULE items 3, 5 and 8): the first pre-registered scoring in this
    runtime, a later identical one in the same runtime, or none. Which hosted run is the evaluation of record is in
    ``RECORD_OF_EVALUATIONS``."""
    if not preregistered:
        return LOOK_EXPLORATORY
    return LOOK_REPRODUCTION if earlier_preregistered_scorings > 0 else LOOK_PREREGISTERED


def adapted_state_check(adapter: Mapping[str, Any] | None, adapt_result: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    """Before the fresh test is scored, the model in memory must be the selected adapted model, not merely a run with the
    right settings: the pipeline's adapter must be the one ``adapt_result`` describes (depth, kept epoch, epochs, learning
    rate, seed, batch size, training-set size) and re-scoring validation now must give the kept epoch's validation
    accuracy and MCC. Returns ``{"matches": bool, "mismatches": [...]}``."""
    mismatches = []
    if not adapter or adapter.get("provisional"):
        return {"matches": False, "mismatches": ["no adapted model in memory (re-run Section 7)"]}
    for key in ("layers", "best_epoch", "epochs", "lr", "seed", "batch_size", "n_train"):
        if adapter.get(key) != adapt_result.get(key):
            mismatches.append(f"adapter {key} {adapter.get(key)!r} != selected {adapt_result.get(key)!r}")
    kept = adapt_result["history"][adapt_result["best_epoch"]].get("val") or {}
    for key in ("accuracy", "mcc"):
        if kept.get(key) is None or abs(float(validation[key]) - float(kept[key])) > 1e-9:
            mismatches.append(f"validation {key} now {validation.get(key)!r} != kept epoch {kept.get(key)!r}")
    return {"matches": not mismatches, "mismatches": mismatches}


def exploratory_suffix(stem: str, existing: Sequence[str]) -> str:
    """``_exploratory_<n>`` with the smallest ``n`` for which neither ``<stem>_evaluation_report_exploratory_<n>.json``
    nor ``<stem>_result_exploratory_<n>.json`` exists, so an exploratory run never replaces a pre-registered file
    or an earlier exploratory one."""
    names = set(existing)
    n = 1
    while f"{stem}_evaluation_report_exploratory_{n}.json" in names or f"{stem}_result_exploratory_{n}.json" in names:
        n += 1
    return f"_exploratory_{n}"
