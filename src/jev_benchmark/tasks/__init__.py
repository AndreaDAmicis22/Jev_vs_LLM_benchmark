"""Registry of the benchmark scenarios."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

from jev_benchmark.tasks import intent_routing, intent_routing_it, output_quality, workflow_next_step
from jev_benchmark.tasks.base import ORDERS, Example, Task, TaskKind

_MODULES: dict[str, ModuleType] = {
    m.TASK.name: m for m in (intent_routing, workflow_next_step, output_quality, intent_routing_it)
}

TASKS: dict[str, Task] = {name: m.TASK for name, m in _MODULES.items()}

# Examples per scenario of the current benchmark version (v2). v1 used 308 / 300 / 300.
BENCHMARK_SIZE = 500


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"Unknown task {name!r}. Available: {', '.join(TASKS)}") from None


def get_preparer(name: str) -> Callable[..., list[Example]]:
    get_task(name)
    return _MODULES[name].prepare


def resolve_tasks(spec: str) -> list[Task]:
    """`"all"`, `"choice"` (the scenarios of the position-bias study) or comma separated names."""
    key = spec.strip().lower()
    if key == "all":
        return list(TASKS.values())
    if key == "choice":
        return [t for t in TASKS.values() if t.kind is TaskKind.CHOICE]
    return [get_task(name.strip()) for name in spec.split(",") if name.strip()]


__all__ = [
    "BENCHMARK_SIZE",
    "ORDERS",
    "TASKS",
    "Example",
    "Task",
    "TaskKind",
    "get_preparer",
    "get_task",
    "resolve_tasks",
]
