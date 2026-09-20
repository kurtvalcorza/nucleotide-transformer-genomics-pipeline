"""Binary sequence-classification measures and two non-neural baselines, in plain Python.

``classification_metrics`` scores predicted labels against gold labels: accuracy, Matthews correlation
(MCC, the headline metric — it is 0 for any constant or chance predictor and symmetric in the two
classes), per-class precision / recall / F1 and the confusion counts. ``majority_baseline`` predicts the
training majority class; ``gc_threshold_baseline`` predicts from GC content alone with the threshold and
direction chosen on the training split. Promoters are GC-rich, so the second baseline is the honest
composition-only reference any sequence model must beat.
"""

from __future__ import annotations

# ruff: noqa: E501  -- the inline sample block and contract lines are kept at the fleet width
import math
from collections.abc import Mapping, Sequence
from typing import Any

LABELS = (0, 1)


def gc_content(sequence: str) -> float:
    """Fraction of G and C bases (case-insensitive); ``N`` counts toward the length, not the GC total."""
    upper = sequence.upper()
    return (upper.count("G") + upper.count("C")) / len(upper) if upper else 0.0


def classification_metrics(predictions: Sequence[int], gold: Sequence[int]) -> dict[str, Any]:
    """Accuracy, MCC, per-class precision/recall/F1 and the confusion counts for binary labels."""
    if len(predictions) != len(gold):
        raise ValueError(f"{len(predictions)} predictions for {len(gold)} labels")
    if not gold:
        raise ValueError("no records to score")
    for value in (*predictions, *gold):
        if value not in LABELS:
            raise ValueError(f"labels must be 0 or 1, got {value!r}")
    tp = sum(1 for p, g in zip(predictions, gold, strict=True) if p == 1 and g == 1)
    tn = sum(1 for p, g in zip(predictions, gold, strict=True) if p == 0 and g == 0)
    fp = sum(1 for p, g in zip(predictions, gold, strict=True) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(predictions, gold, strict=True) if p == 0 and g == 1)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0

    def _prf(t: int, f_pos: int, f_neg: int) -> dict[str, float]:
        precision = t / (t + f_pos) if t + f_pos else 0.0
        recall = t / (t + f_neg) if t + f_neg else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    return {
        "n": len(gold),
        "accuracy": round((tp + tn) / len(gold), 4),
        "mcc": round(mcc, 4),
        "positive": _prf(tp, fp, fn),
        "negative": _prf(tn, fn, fp),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def majority_baseline(train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Predict the training majority label (ties go to 1) for every test record."""
    positives = sum(int(r["label"]) for r in train)
    label = 1 if positives * 2 >= len(train) else 0
    scored = classification_metrics([label] * len(test), [int(r["label"]) for r in test])
    return {"baseline": "majority", "label": label, **scored}


def gc_threshold_baseline(train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Predict from GC content alone: the midpoint threshold and direction that maximise training accuracy."""
    train_gc = [(gc_content(r["sequence"]), int(r["label"])) for r in train]
    observed = sorted({gc for gc, _ in train_gc})
    # thresholds halfway between neighbouring observed values, plus one below and one above the range
    candidates = [observed[0] - 1e-9, *((a + b) / 2 for a, b in zip(observed, observed[1:], strict=False)), observed[-1] + 1e-9]
    best = (-1.0, candidates[0], 1)
    for threshold in candidates:
        for direction in (1, -1):  # 1: GC >= threshold -> positive; -1: GC >= threshold -> negative
            correct = sum(1 for gc, label in train_gc if (int(gc >= threshold) if direction == 1 else int(gc < threshold)) == label)
            accuracy = correct / len(train_gc)
            if accuracy > best[0]:
                best = (accuracy, threshold, direction)
    _, threshold, direction = best
    predictions = [int(gc_content(r["sequence"]) >= threshold) if direction == 1 else int(gc_content(r["sequence"]) < threshold) for r in test]
    scored = classification_metrics(predictions, [int(r["label"]) for r in test])
    return {
        "baseline": "gc_threshold",
        "threshold": round(threshold, 4),
        "rule": "GC >= threshold -> positive" if direction == 1 else "GC < threshold -> positive",
        "train_accuracy": round(best[0], 4),
        **scored,
    }
