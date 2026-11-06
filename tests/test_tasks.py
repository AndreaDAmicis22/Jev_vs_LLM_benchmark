"""Invariants of the versioned benchmark sets and of the scenario definitions."""

from __future__ import annotations

from collections import Counter

import pytest

from jev_benchmark.tasks import TASKS, TaskKind
from jev_benchmark.tasks.base import render_state_text
from jev_benchmark.tasks.output_quality import render_prompt

EXPECTED = {"intent_routing": (77, 4), "workflow_next_step": (30, 10), "output_quality": (5, 60)}


@pytest.mark.parametrize("name", list(TASKS))
def test_benchmark_set_is_balanced_and_valid(name):
    task = TASKS[name]
    n_labels, per_class = EXPECTED[name]
    assert len(task.labels) == n_labels
    assert all(desc.strip() for desc in task.labels.values())
    examples = task.load()
    assert len({ex.id for ex in examples}) == len(examples)
    counts = Counter(ex.label for ex in examples)
    assert set(counts) == set(task.labels)
    assert set(counts.values()) == {per_class}
    assert all(ex.state and all(v.strip() for v in ex.state.values()) for ex in examples)


def test_ordinal_levels_are_consecutive_integers():
    task = TASKS["output_quality"]
    assert task.kind is TaskKind.ORDINAL
    assert task.label_names == [str(i) for i in range(len(task.labels))]


def test_workflow_examples_end_before_the_gold_action():
    for ex in TASKS["workflow_next_step"].load():
        assert ex.state["procedure"].startswith("Workflow: ")
        # one decision point per conversation
    convos = Counter(ex.meta["convo_id"] for ex in TASKS["workflow_next_step"].load())
    assert max(convos.values()) == 1


def test_render_state_text():
    assert render_state_text({"a": "x", "b": "y"}) == "<a>\nx\n</a>\n\n<b>\ny\n</b>"


def test_render_prompt_multiturn():
    raw = "hello\n<extra_id_1>Assistant\nhi!\n<extra_id_1>User\nthanks"
    assert render_prompt(raw) == "User: hello\n\nAssistant: hi!\n\nUser: thanks"
    assert render_prompt("single turn") == "single turn"
