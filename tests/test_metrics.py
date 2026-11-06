"""Metrics and cascade simulation on small synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jev_benchmark import metrics as M
from jev_benchmark.tasks import get_task


def _frame(gold, pred, p, latency, cost, confidence=None):
    df = pd.DataFrame(
        {
            "example_id": [f"e{i}" for i in range(len(gold))],
            "gold": gold,
            "pred": pred,
            "p_pred": p,
            "latency_ms": latency,
            "cost_usd": cost,
            "server_ms": None,
            "expected_score": None,
        }
    )
    df["confidence"] = confidence if confidence is not None else p
    df["ok"] = df["pred"].notna()
    df["correct"] = df["ok"] & (df["pred"] == df["gold"])
    for col in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
        df[col] = 0
    return df


def test_quality_choice_counts_errors_as_wrong():
    task = get_task("workflow_next_step")
    labels = task.label_names
    df = _frame(
        [labels[0], labels[1], labels[2], labels[3]],
        [labels[0], labels[1], None, labels[0]],
        [0.9, 0.8, None, 0.4],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    )
    q = M.quality(df, task)
    assert q["accuracy"] == pytest.approx(0.5)
    assert q["errors"] == 1
    assert 0 <= q["accuracy_ci"][0] <= q["accuracy"] <= q["accuracy_ci"][1] <= 1


def test_quality_ordinal_metrics():
    task = get_task("output_quality")
    df = _frame(["0", "1", "2", "3", "4"], ["0", "2", "2", "3", "3"], [1.0] * 5, [1] * 5, [0] * 5)
    df["expected_score"] = [0.1, 1.8, 2.0, 3.1, 3.4]
    q = M.quality(df, task)
    assert q["accuracy"] == pytest.approx(0.6)
    assert q["within1"] == pytest.approx(1.0)
    assert q["mae"] == pytest.approx(0.4)
    assert q["spearman"] == pytest.approx(1.0)
    assert q["qwk"] > 0.8


def test_mcnemar():
    a = np.array([True, True, True, False, True, True])
    b = np.array([False, False, True, False, True, False])
    r = M.mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (3, 0)
    assert r["p_value"] == pytest.approx(0.25)


def test_cascade_extremes_and_monotonic_escalation():
    gold = ["a", "b", "a", "b"]
    first = _frame(gold, ["a", "a", "a", "a"], [0.95, 0.55, 0.9, 0.6], [100] * 4, [0.001] * 4)
    llm = _frame(gold, ["a", "b", "b", "b"], [1.0] * 4, [1000] * 4, [0.01] * 4)
    m = M.paired(first, llm)
    sweep = M.cascade_sweep(m, "p_pred")
    never, always = sweep.iloc[0], sweep.iloc[-1]
    assert never["escalation_rate"] == 0 and never["accuracy"] == pytest.approx(0.5)
    assert always["escalation_rate"] == 1 and always["accuracy"] == pytest.approx(0.75)
    assert always["usd_per_1k"] == pytest.approx(11.0)  # escalated requests pay both systems
    assert sweep["escalation_rate"].is_monotonic_increasing
    # escalating only the two uncertain answers fixes both mistakes: better than either system
    best = M.cascade_at(m, "p_pred", 0.7)
    assert best["accuracy"] == pytest.approx(1.0) and best["escalation_rate"] == pytest.approx(0.5)


def test_calibration_perfectly_calibrated_bins():
    rng = np.random.default_rng(0)
    p = np.repeat([0.25, 0.75], 400)
    correct = rng.random(800) < p
    df = _frame(["a"] * 800, np.where(correct, "a", "b").tolist(), p, [1] * 800, [0] * 800)
    cal = M.calibration(df)
    assert cal["ece"] < 0.05
