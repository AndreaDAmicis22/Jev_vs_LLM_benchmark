"""Registry of the three benchmark scenarios."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

from jev_benchmark.tasks import intent_routing, output_quality, workflow_next_step
from jev_benchmark.tasks.base import Example, Task, TaskKind

_MODULES: dict[str, ModuleType] = {
    m.TASK.name: m for m in (intent_routing, workflow_next_step, output_quality)
}

TASKS: dict[str, Task] = {name: m.TASK for name, m in _MODULES.items()}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"Unknown task {name!r}. Available: {', '.join(TASKS)}") from None


def get_preparer(name: str) -> Callable[..., list[Example]]:
    get_task(name)
    return _MODULES[name].prepare


def resolve_tasks(spec: str) -> list[Task]:
    """`"all"` or a comma separated list of task names."""
    if spec.strip().lower() == "all":
        return list(TASKS.values())
    return [get_task(name.strip()) for name in spec.split(",") if name.strip()]


__all__ = ["TASKS", "Example", "Task", "TaskKind", "get_preparer", "get_task", "resolve_tasks"]
