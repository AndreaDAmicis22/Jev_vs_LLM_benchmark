"""Command line interface: `jevbench --help`."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from jev_benchmark.runner import load_predictions, results_path, run_task, write_manifest
from jev_benchmark.systems import DEFAULT_SYSTEMS, build_system, parse_systems
from jev_benchmark.tasks import TASKS, Example, get_preparer, get_task, resolve_tasks

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Jev (TypeSafe AI) vs GPT-5.4 on Foundry")
console = Console()

DEFAULT_PER_CLASS = {"intent_routing": 4, "workflow_next_step": 10, "output_quality": 60}


@app.command()
def prepare(
    tasks: Annotated[str, typer.Option(help="'all' or comma separated task names")] = "all",
    seed: Annotated[int, typer.Option(help="Sampling seed")] = 42,
) -> None:
    """Download the public datasets and write the sampled benchmark sets to data/benchmark/."""
    for task in resolve_tasks(tasks):
        examples = get_preparer(task.name)(per_class=DEFAULT_PER_CLASS[task.name], seed=seed)
        path = task.save(examples)
        dist = Counter(ex.label for ex in examples)
        console.print(
            f"[green]OK[/] {task.name}: {len(examples)} examples, {len(dist)} classes "
            f"(min {min(dist.values())}, max {max(dist.values())} per class) -> {path}"
        )


@app.command()
def check() -> None:
    """Send one request to every system to verify credentials and connectivity."""
    task = get_task("intent_routing")
    example = Example(
        id="check", label="card_arrival", state={"customer_message": "My new card still hasn't arrived"}
    )

    async def _check() -> None:
        for name in parse_systems(DEFAULT_SYSTEMS):
            try:
                system = build_system(name)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]FAIL[/] {name}: {exc}")
                continue
            try:
                pred = await system.classify(task, example)
                console.print(
                    f"[green]OK[/] {name}: pred={pred.pred} p={_fmt(pred.p_pred)} "
                    f"latency={pred.latency_ms:.0f} ms model={pred.model}"
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]FAIL[/] {name}: {type(exc).__name__}: {exc}")
            finally:
                await system.aclose()

    asyncio.run(_check())


@app.command()
def run(
    systems: Annotated[str, typer.Option(help="Comma separated systems")] = DEFAULT_SYSTEMS,
    tasks: Annotated[str, typer.Option(help="'all' or comma separated task names")] = "all",
    run_id: Annotated[str, typer.Option(help="Results folder name under results/")] = "main",
    limit: Annotated[int | None, typer.Option(help="Only the first N examples per task")] = None,
    concurrency: Annotated[int, typer.Option(help="Parallel requests per system")] = 8,
) -> None:
    """Run the benchmark (resumable: already answered examples are skipped)."""
    task_list = resolve_tasks(tasks)
    names = parse_systems(systems)

    async def _run() -> dict[str, dict]:
        described = {}
        for name in names:
            system = build_system(name)
            described[name] = system.describe()
            try:
                for task in task_list:
                    examples = task.load(limit)
                    out = results_path(run_id, task.name, name)
                    with _progress() as progress:
                        bar = progress.add_task(f"{name} | {task.name}", total=len(examples))
                        progress.update(bar, completed=sum(p.ok for p in load_predictions(out)))
                        preds = await run_task(
                            system,
                            task,
                            examples,
                            out,
                            concurrency=concurrency,
                            on_progress=lambda p, b=bar: progress.advance(b),
                        )
                    _print_quick_stats(name, task.name, preds)
            finally:
                await system.aclose()
        return described

    described = asyncio.run(_run())
    write_manifest(run_id, described, task_list, {"limit": limit, "concurrency": concurrency})


@app.command()
def classify(
    text: Annotated[str, typer.Argument(help="Text, or a JSON object with the task state fields")],
    task: Annotated[str, typer.Option(help="Task name")] = "intent_routing",
    systems: Annotated[str, typer.Option(help="Comma separated systems")] = "jev,gpt-5.4-none",
) -> None:
    """Classify one input with several systems side by side (quick demo)."""
    spec = get_task(task)
    try:
        state = json.loads(text)
        assert isinstance(state, dict)
    except (json.JSONDecodeError, AssertionError):
        first_field = {"intent_routing": "customer_message"}.get(task, "input")
        state = {first_field: text}
    example = Example(id="adhoc", label=spec.label_names[0], state=state)

    async def _classify() -> None:
        table = Table(title=f"{spec.title}")
        for col in ("system", "decision", "p(decision)", "top-3", "latency ms", "cost USD"):
            table.add_column(col)
        for name in parse_systems(systems):
            system = build_system(name)
            try:
                pred = await system.classify(spec, example)
            finally:
                await system.aclose()
            top = sorted((pred.probabilities or {}).items(), key=lambda kv: -kv[1])[:3]
            table.add_row(
                name,
                str(pred.pred),
                _fmt(pred.p_pred),
                ", ".join(f"{k} {v:.2f}" for k, v in top) or "-",
                f"{pred.latency_ms:.0f}",
                f"{pred.cost_usd:.6f}",
            )
        console.print(table)

    asyncio.run(_classify())


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _progress() -> Progress:
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def _print_quick_stats(system: str, task: str, preds: list) -> None:
    ok = [p for p in preds if p.ok]
    if not ok:
        console.print(f"[red]{system} | {task}: no successful predictions[/]")
        return
    acc = sum(p.correct for p in ok) / len(ok)
    lat = sorted(p.latency_ms for p in ok)
    errors = len(preds) - len(ok)
    console.print(
        f"  {system} | {task}: n={len(ok)} acc={acc:.3f} p50={lat[len(lat) // 2]:.0f} ms "
        f"cost=${sum(p.cost_usd for p in ok):.4f}" + (f" [red]errors={errors}[/]" if errors else "")
    )


@app.command(name="tasks")
def list_tasks() -> None:
    """List the benchmark scenarios."""
    for task in TASKS.values():
        console.print(
            f"[bold]{task.name}[/] - {task.title} ({task.kind}, {len(task.labels)} options) - {task.dataset}"
        )


if __name__ == "__main__":
    app()
