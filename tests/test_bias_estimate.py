"""Position-bias analysis on synthetic answers and the offline cost estimator."""

from __future__ import annotations

import pandas as pd
import pytest

from jev_benchmark import bias as B
from jev_benchmark import estimate as E
from jev_benchmark.config import GPT54_PRICING, JEV_PRICING
from jev_benchmark.tasks import get_task
from jev_benchmark.tasks.base import Task, TaskKind

TOY = Task(
    name="toy",
    title="toy",
    kind=TaskKind.CHOICE,
    instructions="Pick one",
    labels={name: f"option {name}" for name in "abcdefgh"},
    dataset="synthetic",
)


def _biased_answers(order: str) -> pd.DataFrame:
    """Every other example the system picks whatever is listed first (p 0.4), else the gold (p 0.9)."""
    rows = []
    for i in range(80):
        ex_id, gold = f"toy-{i}", "abcdefgh"[i % 8]
        first = next(iter(TOY.ordered_labels(order, ex_id)))
        pred, p = (first, 0.4) if i % 2 == 0 else (gold, 0.9)
        rows.append(
            {
                "example_id": ex_id,
                "gold": gold,
                "pred": pred,
                "ok": True,
                "correct": pred == gold,
                "p_pred": p,
                "probabilities": None,
            }
        )
    return pd.DataFrame(rows)


def test_position_bias_is_detected():
    res = B.analyse(TOY, {o: _biased_answers(o) for o in ("canonical", "reversed", "shuffled")})
    shuffled = res["orders"]["shuffled"]
    assert shuffled["gold_first"] < 0.25 < shuffled["pred_first"]  # first position over-selected
    assert shuffled["pred_deciles"][0] > shuffled["gold_deciles"][0]
    assert res["consistency"] < 1 and res["flips"]["n"] > 0
    assert res["flips"]["median_p_flip"] < res["flips"]["median_p_stable"]
    assert res["flips"]["auroc_low_p_flags_flip"] > 0.5  # low probability flags unstable answers


def test_unbiased_system_is_consistent():
    def perfect(order: str) -> pd.DataFrame:
        df = _biased_answers(order)
        df["pred"], df["correct"], df["p_pred"] = df["gold"], True, 0.95
        return df

    res = B.analyse(TOY, {o: perfect(o) for o in ("canonical", "reversed", "shuffled")})
    assert res["consistency"] == 1.0 and res["flips"]["n"] == 0
    assert all(o["accuracy"] == 1.0 for o in res["orders"].values())


def test_estimate_model_without_measurements(tmp_path):
    one = lambda text: 1  # noqa: E731 - every rendered string counts as one token
    rows = E.estimate(
        ["jev", "jev-perm", "gpt-5.4-none"], [get_task("intent_routing")], run_id="_none", count=one
    )
    by = {r.system: r for r in rows}
    assert by["jev"].pending == 500 and by["jev"].input_tokens == 3  # state + instructions + options
    assert by["jev-perm"].input_tokens == 7  # state + 3 x (instructions + options)
    assert by["jev"].usd == pytest.approx(500 * JEV_PRICING.cost(3, 0))
    llm = by["gpt-5.4-none"]
    assert llm.input_tokens == 2 + 1 + E.MESSAGE_OVERHEAD_TOKENS  # prompt + schema, state, overhead
    assert llm.output_tokens == 18 and llm.output_basis == "default"
    assert llm.usd == pytest.approx(500 * GPT54_PRICING.cost(15, 18))


def test_estimate_cache_model():
    big = lambda text: 2000  # noqa: E731
    task = get_task("intent_routing")
    canonical, shuffled = E.estimate(
        ["gpt-5.4-none", "gpt-5.4-none@shuffled"], [task], run_id="_none", count=big
    )
    assert canonical.cached_tokens == (4000 // 128) * 128  # prompt + schema prefix, 128-token blocks
    assert shuffled.cached_tokens == 0  # a different option order per example breaks the cache
    assert shuffled.usd > canonical.usd


def test_estimate_skips_unsupported_combinations():
    rows = E.estimate(["jev@reversed", "jev-perm"], [get_task("output_quality")], run_id="_none", count=len)
    assert rows == []
