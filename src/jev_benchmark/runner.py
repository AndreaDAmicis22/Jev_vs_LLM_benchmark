"""Run systems on tasks, with bounded concurrency, retries and resumable JSONL output.

Layout: `results/<run_id>/<task>/<system>.jsonl` (one `Prediction` per line). Re-running the same
command skips the examples already answered successfully, so an interrupted run can be resumed.
"""

from __future__ import annotations

import asyncio
import json
import platform
import random
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from jev_benchmark.config import PROJECT_ROOT, RESULTS_DIR
from jev_benchmark.systems.base import Classifier, Prediction
from jev_benchmark.tasks.base import Example, Task


def results_path(run_id: str, task: str, system: str) -> Path:
    return RESULTS_DIR / run_id / task / f"{system}.jsonl"


def load_predictions(path: Path) -> list[Prediction]:
    """Load a results file keeping the last record per example (later retries win)."""
    if not path.exists():
        return []
    latest: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                pred = Prediction.model_validate_json(line)
                latest[pred.example_id] = pred
    return list(latest.values())


async def _attempts(system: Classifier, task: Task, example: Example, max_attempts: int) -> Prediction:
    start = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        try:
            pred = await system.classify(task, example)
            pred.attempts = attempt
            pred.total_ms = (time.perf_counter() - start) * 1000
            return pred
        except Exception as exc:  # noqa: BLE001 - every failure must be recorded, not raised
            if attempt < max_attempts and system.is_retryable(exc):
                await asyncio.sleep(min(30.0, 2 ** (attempt - 1)) + random.random())
                continue
            return Prediction(
                task=task.name,
                system=system.name,
                example_id=example.id,
                gold=example.label,
                attempts=attempt,
                total_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {str(exc)[:500]}",
            )
    raise AssertionError("unreachable")


async def run_task(
    system: Classifier,
    task: Task,
    examples: list[Example],
    out_path: Path,
    concurrency: int = 8,
    max_attempts: int = 5,
    warmup: bool = True,
    on_progress: Callable[[Prediction], None] | None = None,
) -> list[Prediction]:
    """Classify every example not yet answered successfully; append results to `out_path`."""
    done = {p.example_id for p in load_predictions(out_path) if p.ok}
    todo = [ex for ex in examples if ex.id not in done]
    if not todo:
        return load_predictions(out_path)

    if warmup:  # open connections / TLS before measuring; result discarded
        await _attempts(system, task, todo[0], max_attempts=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    with out_path.open("a", encoding="utf-8", newline="\n") as fh:

        async def worker(example: Example) -> None:
            async with semaphore:
                pred = await _attempts(system, task, example, max_attempts)
            async with lock:
                fh.write(pred.model_dump_json() + "\n")
                fh.flush()
            if on_progress:
                on_progress(pred)

        await asyncio.gather(*(worker(ex) for ex in todo))
    return load_predictions(out_path)


def write_manifest(run_id: str, systems: dict[str, dict], tasks: list[Task], extra: dict) -> Path:
    """Record what was run (models, pricing, sample sizes, git commit) next to the results."""
    path = RESULTS_DIR / run_id / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        commit = None
    manifest.setdefault("runs", []).append(
        {
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": commit,
            "python": platform.python_version(),
            "host": platform.platform(),
            "tasks": {t.name: {"dataset": t.dataset, "n_labels": len(t.labels)} for t in tasks},
            "systems": systems,
            **extra,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
