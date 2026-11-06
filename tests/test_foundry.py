"""Prompt/schema construction and logprob parsing of the Foundry LLM baseline."""

from __future__ import annotations

import math
from types import SimpleNamespace as NS

import pytest

from jev_benchmark.systems.foundry import (
    _level_distribution,
    _tokens_in_span,
    _value_span,
    build_response_format,
    build_system_prompt,
)
from jev_benchmark.tasks import get_task


def test_choice_schema_is_strict_enum_of_all_labels():
    task = get_task("workflow_next_step")
    fmt = build_response_format(task)
    assert fmt["strict"] is True
    assert fmt["schema"]["properties"]["label"]["enum"] == task.label_names
    assert fmt["schema"]["additionalProperties"] is False


def test_ordinal_schema_uses_integer_levels():
    fmt = build_response_format(get_task("output_quality"))
    assert fmt["schema"]["properties"]["score"] == {"type": "integer", "enum": [0, 1, 2, 3, 4]}


def test_system_prompt_contains_same_instructions_and_descriptions():
    task = get_task("intent_routing")
    prompt = build_system_prompt(task)
    assert prompt.startswith(task.instructions)
    for name, description in task.labels.items():
        assert f"- {name}: {description}" in prompt


def _tok(token: str, logprob: float, alts=()):
    return NS(token=token, logprob=logprob, top_logprobs=[NS(token=t, logprob=lp) for t, lp in alts])


def test_choice_label_probability_is_product_of_label_tokens():
    tokens = [
        _tok('{"', 0.0),
        _tok("label", 0.0),
        _tok('":"', 0.0),
        _tok("card", -0.1),
        _tok("_arrival", -0.2),
        _tok('"}', 0.0),
    ]
    text = "".join(t.token for t in tokens)
    span = _value_span(text, "label")
    assert text[span[0] : span[1]] == "card_arrival"
    selected = _tokens_in_span(tokens, span)
    assert [t.token for t in selected] == ["card", "_arrival"]
    assert math.exp(sum(t.logprob for t in selected)) == pytest.approx(math.exp(-0.3))


def test_ordinal_distribution_from_top_logprobs():
    tokens = [
        _tok('{"', 0.0),
        _tok("score", 0.0),
        _tok('":', 0.0),
        _tok("3", math.log(0.6), alts=[("3", math.log(0.6)), ("4", math.log(0.3)), ("2", math.log(0.1))]),
        _tok("}", 0.0),
    ]
    text = "".join(t.token for t in tokens)
    span = _value_span(text, "score")
    assert text[span[0] : span[1]] == "3"
    dist = _level_distribution(tokens, span, ["0", "1", "2", "3", "4"])
    assert dist == pytest.approx({"0": 0.0, "1": 0.0, "2": 0.1, "3": 0.6, "4": 0.3})
