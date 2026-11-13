"""Offline cost estimate of a planned run: no request is sent to any model endpoint.

For every (system, scenario) the estimate renders the exact prompts the system would send, counts
their tokens with `o200k_base` (the tokenizer of the GPT-5 family; for Jev it is an approximation,
its own tokenizer is not public) and applies the list prices of `config.py`:

* examples already answered in `results/<run_id>/` are skipped, like `jevbench run` does;
* output tokens use the mean measured on previous results of the same system and scenario when
  available, otherwise the defaults below (measured on run v1 of November 2026);
* Azure prompt caching is modelled as Azure applies it: a constant prefix of at least 1,024 tokens is
  billed at the cached price in blocks of 128 tokens. The system prompt is constant except with the
  per-example shuffled option order, which therefore gets no cache discount.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean

from jev_benchmark.config import JEV_PRICING, Pricing, llm_pricing
from jev_benchmark.runner import load_predictions, results_path
from jev_benchmark.systems.foundry import REASONING_EFFORTS, build_response_format, build_system_prompt
from jev_benchmark.systems.jev import PERM_ORDERS
from jev_benchmark.tasks.base import ORDERS, Example, Task, TaskKind, render_state_text

MESSAGE_OVERHEAD_TOKENS = 12  # role markers and formatting of the two chat messages
CACHE_MIN_TOKENS, CACHE_BLOCK = 1024, 128

# Mean output tokens per request measured in run v1 (gpt-5.4, 2026-11-06); reasoning tokens included.
DEFAULT_OUTPUT_TOKENS = {
    "none": dict.fromkeys(
        ("intent_routing", "workflow_next_step", "output_quality", "intent_routing_it"), 18
    ),
    "medium": {
        "intent_routing": 81,
        "workflow_next_step": 348,
        "output_quality": 193,
        "intent_routing_it": 81,
    },
}


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kind: str  # "jev" | "jev-perm" | "llm"
    order: str
    deployment: str | None = None
    effort: str | None = None

    @property
    def provider(self) -> str:
        return "TypeSafe (Jev)" if self.kind.startswith("jev") else "Microsoft Foundry"

    @property
    def pricing(self) -> Pricing:
        return JEV_PRICING if self.kind.startswith("jev") else llm_pricing(self.deployment or "")

    def supports(self, task: Task) -> bool:
        if self.kind == "jev-perm":
            return task.kind is TaskKind.CHOICE
        return self.order == "canonical" or task.kind is TaskKind.CHOICE


def parse_system(name: str) -> SystemSpec:
    """Same naming rules as `systems.build_system`, without creating clients or needing keys."""
    base, _, order = name.partition("@")
    order = order or "canonical"
    if order not in ORDERS:
        raise ValueError(f"Unknown option order in {name!r}")
    if base in ("jev", "jev-perm"):
        return SystemSpec(name, base, order)
    deployment, _, effort = base.rpartition("-")
    if not deployment or effort not in REASONING_EFFORTS:
        raise ValueError(f"Unknown system {name!r}")
    llm_pricing(deployment)  # unknown deployment -> clear error before estimating
    return SystemSpec(name, "llm", order, deployment, effort)


@dataclass
class EstimateRow:
    system: str
    task: str
    provider: str
    pending: int
    input_tokens: float  # mean per request
    cached_tokens: float  # mean per request
    output_tokens: float  # mean per request
    output_basis: str  # "measured" | "default" | "assumed"
    usd: float  # total for the pending requests


def token_counter() -> Callable[[str], int]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        return lambda text: len(encoding.encode(text, disallowed_special=()))
    except Exception:  # offline without the tokenizer file: ~4 characters per token
        return lambda text: max(1, math.ceil(len(text) / 4))


def _jev_input_tokens(spec: SystemSpec, task: Task, ex: Example, count: Callable[[str], int]) -> int:
    orders = PERM_ORDERS if spec.kind == "jev-perm" else (spec.order,)
    state = count(json.dumps(ex.state, ensure_ascii=False))
    questions = sum(
        count(task.instructions) + count(json.dumps(task.ordered_labels(o, ex.id), ensure_ascii=False))
        for o in orders
    )
    return state + questions


def _llm_tokens(spec: SystemSpec, task: Task, ex: Example, count: Callable[[str], int]) -> tuple[int, int]:
    """(input tokens, cached tokens) of one request from the token-count model."""
    labels = task.ordered_labels(spec.order, ex.id)
    # The Structured Outputs schema (with the enum of all labels) is billed as input and is part of
    # the constant prefix together with the system prompt.
    prefix = count(build_system_prompt(task, labels)) + count(json.dumps(build_response_format(task, labels)))
    total = prefix + count(render_state_text(ex.state)) + MESSAGE_OVERHEAD_TOKENS
    cacheable = spec.order != "shuffled" and prefix >= CACHE_MIN_TOKENS
    cached = (prefix // CACHE_BLOCK) * CACHE_BLOCK if cacheable else 0
    return total, cached


def _reference(run_id: str, task: Task, count: Callable[[str], int]) -> dict | None:
    """Measured vs modelled input tokens, and cache share, from canonical LLM results on disk.

    The prompts depend only on scenario and option order, not on deployment or reasoning effort, so
    measurements of one LLM system calibrate the estimate of the others on the same scenario.
    """
    examples = {ex.id: ex for ex in task.load()}
    for name in ("gpt-5.4-none", "gpt-5.4-medium", "gpt-5.4-mini-none"):
        measured = [
            p
            for p in load_predictions(results_path(run_id, task.name, name))
            if p.ok and p.example_id in examples
        ]
        if measured:
            spec = parse_system(name)
            modelled = fmean(_llm_tokens(spec, task, examples[p.example_id], count)[0] for p in measured)
            total_in = sum(p.input_tokens for p in measured)
            return {
                "factor": fmean(p.input_tokens for p in measured) / modelled,
                "cache_share": sum(p.cached_tokens for p in measured) / max(total_in, 1),
                "source": name,
            }
    return None


def _output_tokens(spec: SystemSpec, task: Task, measured: list) -> tuple[float, str]:
    if spec.kind != "llm":
        return 0.0, "free"
    if measured:
        return fmean(p.output_tokens for p in measured), "measured"
    defaults = DEFAULT_OUTPUT_TOKENS.get(spec.effort or "")
    if defaults and task.name in defaults:
        return float(defaults[task.name]), "default"
    return float(DEFAULT_OUTPUT_TOKENS["medium"].get(task.name, 200)), "assumed"


def estimate(
    systems: list[str],
    tasks: list[Task],
    run_id: str = "main",
    include_done: bool = False,
    count: Callable[[str], int] | None = None,
) -> list[EstimateRow]:
    count = count or token_counter()
    references = {task.name: _reference(run_id, task, count) for task in tasks}
    rows: list[EstimateRow] = []
    for name in systems:
        spec = parse_system(name)
        for task in tasks:
            if not spec.supports(task):
                continue
            previous = [p for p in load_predictions(results_path(run_id, task.name, name)) if p.ok]
            done = {p.example_id for p in previous}
            todo = [ex for ex in task.load() if include_done or ex.id not in done]
            out_tokens, basis = _output_tokens(spec, task, previous)
            ref = references[task.name]
            inputs, cached = [], []
            for ex in todo:
                if spec.kind == "llm":
                    total, cache = _llm_tokens(spec, task, ex, count)
                    if ref:  # calibrate on measured requests of the same scenario
                        total = round(total * ref["factor"])
                        cache = 0 if spec.order == "shuffled" else round(total * ref["cache_share"])
                else:
                    total, cache = _jev_input_tokens(spec, task, ex, count), 0
                inputs.append(total)
                cached.append(cache)
            usd = sum(spec.pricing.cost(i, round(out_tokens), c) for i, c in zip(inputs, cached, strict=True))
            rows.append(
                EstimateRow(
                    system=name,
                    task=task.name,
                    provider=spec.provider,
                    pending=len(todo),
                    input_tokens=fmean(inputs) if inputs else 0.0,
                    cached_tokens=fmean(cached) if cached else 0.0,
                    output_tokens=out_tokens,
                    output_basis=basis,
                    usd=usd,
                )
            )
    return rows


def calibration(run_id: str, tasks: list[Task], count: Callable[[str], int] | None = None) -> list[dict]:
    """Estimated vs measured mean input tokens for LLM results already on disk (sanity check)."""
    count = count or token_counter()
    checks = []
    for task in tasks:
        folder = results_path(run_id, task.name, "x").parent
        if not folder.exists():
            continue
        examples = {ex.id: ex for ex in task.load()}
        for path in sorted(folder.glob("*.jsonl")):
            try:
                spec = parse_system(path.stem)
            except (ValueError, KeyError):
                continue
            if spec.kind != "llm":
                continue
            measured = [p for p in load_predictions(path) if p.ok and p.example_id in examples]
            if not measured:
                continue
            estimated = fmean(_llm_tokens(spec, task, examples[p.example_id], count)[0] for p in measured)
            checks.append(
                {
                    "system": spec.name,
                    "task": task.name,
                    "n": len(measured),
                    "estimated_input": estimated,
                    "measured_input": fmean(p.input_tokens for p in measured),
                }
            )
    return checks
