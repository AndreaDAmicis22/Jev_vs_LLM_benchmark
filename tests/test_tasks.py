"""Invariants of the versioned benchmark sets and of the scenario definitions."""

from __future__ import annotations

from collections import Counter

import pytest

from jev_benchmark.runner import load_predictions, results_path
from jev_benchmark.tasks import BENCHMARK_SIZE, TASKS, TaskKind
from jev_benchmark.tasks.base import balanced_quotas, render_state_text, stratified_sample
from jev_benchmark.tasks.output_quality import render_prompt

N_OPTIONS = {"intent_routing": 77, "workflow_next_step": 30, "output_quality": 5, "intent_routing_it": 60}


@pytest.mark.parametrize("name", list(TASKS))
def test_benchmark_set_is_valid(name):
    task = TASKS[name]
    assert len(task.labels) == N_OPTIONS[name]
    assert all(desc.strip() for desc in task.labels.values())
    examples = task.load()
    assert len(examples) == BENCHMARK_SIZE
    assert len({ex.id for ex in examples}) == len(examples)
    assert {ex.label for ex in examples} <= set(task.labels)
    assert all(ex.state and all(v.strip() for v in ex.state.values()) for ex in examples)


@pytest.mark.parametrize("name", ["intent_routing", "workflow_next_step"])
def test_balanced_when_every_class_has_enough_examples(name):
    counts = Counter(ex.label for ex in TASKS[name].load())
    assert set(counts) == set(TASKS[name].labels)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_quasi_balanced_when_classes_are_rare():
    # HelpSteer2 validation has only 78 pairs at level 0 and 93 at level 1: all taken, others ~110
    counts = Counter(ex.label for ex in TASKS["output_quality"].load())
    assert counts["0"] == 78 and counts["1"] == 93
    assert {counts["2"], counts["3"], counts["4"]} <= {109, 110}
    it = Counter(ex.label for ex in TASKS["intent_routing_it"].load())
    assert max(it.values()) - min(c for c in it.values() if c >= 8) <= 1  # rare intents are capped


@pytest.mark.parametrize("name", list(TASKS))
def test_saved_results_match_the_current_benchmark_examples(name):
    """Results of earlier runs stay valid only if their examples are still in the set, unchanged."""
    examples = {ex.id: ex for ex in TASKS[name].load()}
    folder = results_path("main", name, "x").parent
    for path in folder.glob("*.jsonl") if folder.exists() else []:
        for pred in load_predictions(path):
            assert pred.example_id in examples, f"{path.name}: {pred.example_id} not in the benchmark set"
            assert pred.gold == examples[pred.example_id].label


def test_ordered_labels():
    task = TASKS["workflow_next_step"]
    canonical = list(task.ordered_labels("canonical", "x"))
    assert canonical == task.label_names
    assert list(task.ordered_labels("reversed", "x")) == canonical[::-1]
    shuffled_a = list(task.ordered_labels("shuffled", "example-a"))
    assert sorted(shuffled_a) == sorted(canonical) and shuffled_a != canonical
    assert shuffled_a == list(task.ordered_labels("shuffled", "example-a"))  # reproducible
    assert shuffled_a != list(task.ordered_labels("shuffled", "example-b"))  # per-example permutation
    assert task.ordered_labels("reversed", "x")[canonical[0]] == task.labels[canonical[0]]
    with pytest.raises(ValueError):
        task.ordered_labels("alphabetical", "x")


def test_balanced_quotas():
    assert balanced_quotas({"a": 10, "b": 10, "c": 10}, 9, seed=0) == {"a": 3, "b": 3, "c": 3}
    q = balanced_quotas({"a": 1, "b": 10, "c": 10}, 9, seed=0)
    assert q["a"] == 1 and q["b"] + q["c"] == 8 and abs(q["b"] - q["c"]) <= 1
    with pytest.raises(ValueError):
        balanced_quotas({"a": 1}, 2, seed=0)


def test_stratified_sample_grows_as_a_superset():
    items = [(i, "abc"[i % 3]) for i in range(60)]
    small = stratified_sample(items, key=lambda x: x[1], per_class=3, seed=42)
    large = stratified_sample(items, key=lambda x: x[1], per_class={"a": 7, "b": 6, "c": 6}, seed=42)
    assert set(small) <= set(large) and len(large) == 19


def test_ordinal_levels_are_consecutive_integers():
    task = TASKS["output_quality"]
    assert task.kind is TaskKind.ORDINAL
    assert task.label_names == [str(i) for i in range(len(task.labels))]


def test_workflow_examples():
    examples = TASKS["workflow_next_step"].load()
    assert all(ex.state["procedure"].startswith("Workflow: ") for ex in examples)
    assert max(Counter(ex.meta["convo_id"] for ex in examples).values()) == 1  # one point per conversation


def test_italian_scenario_is_in_italian():
    task = TASKS["intent_routing_it"]
    examples = task.load()
    assert all(set(ex.state) == {"richiesta_utente"} for ex in examples)
    assert "assistente virtuale" in task.instructions
    assert sum(ex.meta["split"] == "test" for ex in examples) > 450  # dev only fills rare intents


def test_render_state_text():
    assert render_state_text({"a": "x", "b": "y"}) == "<a>\nx\n</a>\n\n<b>\ny\n</b>"


def test_render_prompt_multiturn():
    raw = "hello\n<extra_id_1>Assistant\nhi!\n<extra_id_1>User\nthanks"
    assert render_prompt(raw) == "User: hello\n\nAssistant: hi!\n\nUser: thanks"
    assert render_prompt("single turn") == "single turn"
