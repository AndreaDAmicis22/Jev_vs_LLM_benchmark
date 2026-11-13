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
from jev_benchmark.tasks import BENCHMARK_SIZE, TASKS, Example, get_preparer, get_task, resolve_tasks

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Jev (TypeSafe AI) vs GPT-5.4 on Foundry")
console = Console()


@app.command()
def prepare(
    tasks: Annotated[str, typer.Option(help="'all', 'choice' or comma separated task names")] = "all",
    size: Annotated[int, typer.Option(help="Examples per scenario")] = BENCHMARK_SIZE,
    seed: Annotated[int, typer.Option(help="Sampling seed")] = 42,
) -> None:
    """Download the public datasets and write the sampled benchmark sets to data/benchmark/.

    Offline with respect to the models: only the public datasets are downloaded.
    """
    for task in resolve_tasks(tasks):
        examples = get_preparer(task.name)(total=size, seed=seed)
        path = task.save(examples)
        dist = Counter(ex.label for ex in examples)
        splits = Counter(ex.meta.get("split") for ex in examples)
        console.print(
            f"[green]OK[/] {task.name}: {len(examples)} examples, {len(dist)} classes "
            f"(min {min(dist.values())}, max {max(dist.values())} per class), splits {dict(splits)} -> {path}"
        )


@app.command()
def check(
    systems: Annotated[str, typer.Option(help="Systems or presets (main, bias, all)")] = DEFAULT_SYSTEMS,
) -> None:
    """Send ONE request to every system to verify credentials, deployments and logprobs (~$0.01)."""
    task = get_task("intent_routing")
    example = Example(
        id="check", label="card_arrival", state={"customer_message": "My new card still hasn't arrived"}
    )

    async def _check() -> None:
        for name in parse_systems(systems):
            try:
                system = build_system(name)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]FAIL[/] {name}: {exc}")
                continue
            try:
                pred = await system.classify(task, example)
                note = system.describe().get("logprobs_note")
                console.print(
                    f"[green]OK[/] {name}: pred={pred.pred} p={_fmt(pred.p_pred)} "
                    f"latency={pred.latency_ms:.0f} ms model={pred.model}"
                    + (f" [yellow](logprobs {note})[/]" if note else "")
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]FAIL[/] {name}: {type(exc).__name__}: {exc}")
            finally:
                await system.aclose()

    asyncio.run(_check())


@app.command()
def run(
    systems: Annotated[str, typer.Option(help="Systems or presets (main, bias, all)")] = DEFAULT_SYSTEMS,
    tasks: Annotated[str, typer.Option(help="'all', 'choice' or comma separated task names")] = "all",
    run_id: Annotated[str, typer.Option(help="Results folder name under results/")] = "main",
    limit: Annotated[int | None, typer.Option(help="Only the first N examples per task")] = None,
    concurrency: Annotated[int, typer.Option(help="Parallel requests per system")] = 8,
) -> None:
    """Run the benchmark (resumable: already answered examples are skipped).

    Calls the paid endpoints: check the expected spend first with `jevbench estimate`.
    """
    task_list = resolve_tasks(tasks)
    names = parse_systems(systems)

    async def _run() -> dict[str, dict]:
        described = {}
        for name in names:
            system = build_system(name)
            try:
                for task in task_list:
                    if not system.supports(task):  # e.g. option-order variants on the ordinal S3
                        continue
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
                described[name] = system.describe()  # after the run: includes e.g. logprobs fallbacks
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
        # single-field scenarios: use the field name of the benchmark examples
        state = {next(iter(spec.load(limit=1)[0].state)): text}
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


@app.command()
def estimate(
    systems: Annotated[str, typer.Option(help="Systems or presets (main, bias, all)")] = "all",
    tasks: Annotated[str, typer.Option(help="'all', 'choice' or comma separated task names")] = "all",
    run_id: Annotated[str, typer.Option(help="Results already in this run are not counted")] = "main",
    include_done: Annotated[bool, typer.Option(help="Also count examples already answered")] = False,
) -> None:
    """Estimate tokens and cost of a run OFFLINE: no model endpoint is called."""
    from jev_benchmark import estimate as est

    count = est.token_counter()
    rows = est.estimate(parse_systems(systems), resolve_tasks(tasks), run_id, include_done, count)
    scope = "tutti gli esempi" if include_done else "solo esempi da eseguire"
    table = Table(title=f"Stima dei costi (run '{run_id}', {scope})")
    for col in (
        "sistema",
        "scenario",
        "richieste",
        "token input/req",
        "di cui cache",
        "token output/req",
        "USD",
    ):
        table.add_column(col, justify="left" if col in ("sistema", "scenario") else "right")
    for r in rows:
        table.add_row(
            r.system,
            r.task,
            f"{r.pending:,}",
            f"{r.input_tokens:,.0f}",
            f"{r.cached_tokens:,.0f}",
            f"{r.output_tokens:,.0f} ({r.output_basis})",
            f"${r.usd:,.4f}",
        )
    console.print(table)
    by_provider: dict[str, float] = {}
    for r in rows:
        by_provider[r.provider] = by_provider.get(r.provider, 0.0) + r.usd
    for provider, usd in by_provider.items():
        n = sum(r.pending for r in rows if r.provider == provider)
        console.print(f"  [bold]{provider}[/]: ${usd:,.2f}  ({n:,} richieste)")
    console.print(f"  [bold]Totale[/]: ${sum(by_provider.values()):,.2f}")
    checks = est.calibration(run_id, resolve_tasks(tasks), count)
    if checks:
        console.print(
            "\nVerifica del conteggio token sui risultati già misurati (input medio per richiesta):"
        )
        for c in checks:
            delta = c["estimated_input"] / c["measured_input"] - 1
            console.print(
                f"  {c['system']} | {c['task']}: stimati {c['estimated_input']:,.0f}, "
                f"misurati {c['measured_input']:,.0f} ({delta:+.1%}, n={c['n']})"
            )
    console.print(
        "[dim]Stima indicativa: token di Jev contati con il tokenizer GPT (o200k_base); output dei modelli "
        "con reasoning molto variabile. Nessuna chiamata agli endpoint è stata effettuata.[/]"
    )


@app.command()
def report(
    run_id: Annotated[str, typer.Option(help="Results folder name under results/")] = "main",
) -> None:
    """Compute metrics and write reports/<run_id>/report.md with figures."""
    from jev_benchmark.report import build_report  # matplotlib is imported only when needed

    path = build_report(run_id)
    console.print(f"[green]OK[/] report written to {path}")


@app.command(name="tasks")
def list_tasks() -> None:
    """List the benchmark scenarios."""
    for task in TASKS.values():
        console.print(
            f"[bold]{task.name}[/] - {task.title} ({task.kind}, {len(task.labels)} options) - {task.dataset}"
        )


if __name__ == "__main__":
    app()
