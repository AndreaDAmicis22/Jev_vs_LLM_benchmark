"""Markdown report (tables + figures in light and dark variants) for one benchmark run."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter  # noqa: E402

from jev_benchmark import metrics as M  # noqa: E402
from jev_benchmark.config import GPT54_PRICING, JEV_PRICING, REPORTS_DIR, RESULTS_DIR  # noqa: E402
from jev_benchmark.tasks import TASKS, TaskKind  # noqa: E402

SYSTEM_ORDER = ["jev", "gpt-5.4-none", "gpt-5.4-medium"]
SYSTEM_LABELS = {
    "jev": "Jev",
    "gpt-5.4-none": "GPT-5.4 · reasoning none",
    "gpt-5.4-medium": "GPT-5.4 · reasoning medium",
}
MIN_BIN = 5  # minimum answers in a reliability bin to be drawn

SHORT_TITLES = {
    "intent_routing": "S1 · Richieste in ingresso",
    "workflow_next_step": "S2 · Passo successivo workflow",
    "output_quality": "S3 · Qualità output generato",
}

# Reference categorical palette (first three slots validate all-pairs in both modes), fixed per
# entity: Jev = slot 1, GPT-5.4 none = slot 2, GPT-5.4 medium = slot 3.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "ramp": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
    },
}

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "savefig.dpi": 160,
    }
)


def label(system: str) -> str:
    return SYSTEM_LABELS.get(system, system)


def color(system: str, theme: dict) -> str:
    idx = SYSTEM_ORDER.index(system) if system in SYSTEM_ORDER else 3
    return theme["series"][idx]


# --- data -----------------------------------------------------------------------------------------


def discover_systems(run_id: str) -> list[str]:
    names: set[str] = set()
    for task in TASKS.values():
        folder = RESULTS_DIR / run_id / task.name
        if folder.exists():
            names |= {p.stem for p in folder.glob("*.jsonl")}
    return sorted(names, key=lambda s: (SYSTEM_ORDER.index(s) if s in SYSTEM_ORDER else 99, s))


def compute(run_id: str) -> tuple[dict, dict]:
    """Return (summary, frames). `summary` is JSON-serialisable."""
    systems = discover_systems(run_id)
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    summary: dict = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tasks": {},
    }
    for task in TASKS.values():
        entry: dict = {"title": task.title, "kind": str(task.kind), "dataset": task.dataset, "systems": {}}
        for system in systems:
            df = M.load_frame(run_id, task, system)
            if df is None:
                continue
            frames[(task.name, system)] = df
            models = sorted({m for m in df["model"].dropna()})
            entry["systems"][system] = {
                "models": models,
                "complete": len(df) == len(task.load()),
                "quality": M.quality(df, task),
                "latency": M.latency(df),
                "cost": M.cost(df),
                "calibration": M.calibration(df),
            }
        if "jev" in entry["systems"]:
            entry["mcnemar"], entry["cascades"] = {}, {}
            jev = frames[(task.name, "jev")]
            for system in entry["systems"]:
                if system == "jev":
                    continue
                other = frames[(task.name, system)]
                pair = jev[["example_id", "correct"]].merge(other[["example_id", "correct"]], on="example_id")
                entry["mcnemar"][system] = M.mcnemar_exact(
                    pair["correct_x"].to_numpy(bool), pair["correct_y"].to_numpy(bool)
                )
                m = M.paired(jev, other)
                signal = M.default_signal(task)
                llm_acc = float(other["correct"].mean())
                entry["cascades"][system] = {
                    "signal": signal,
                    "parity": M.cascade_operating_point(m, signal, target_accuracy=llm_acc),
                    "near_parity": M.cascade_operating_point(m, signal, target_accuracy=llm_acc - 0.02),
                    "sweep": M.cascade_sweep(m, signal).to_dict(orient="records"),
                }
        summary["tasks"][task.name] = entry
    return summary, frames


# --- figures --------------------------------------------------------------------------------------


def _figure(theme: dict, n: int, title: str, height: float = 3.5, width: float = 4.3):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), facecolor=theme["surface"], squeeze=False)
    fig.suptitle(
        title, x=0.01, y=0.99, ha="left", va="top", color=theme["ink"], fontsize=12, fontweight="bold"
    )
    return fig, list(axes[0])


def _style(ax, theme: dict, grid: str = "both") -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["ink2"], labelsize=8.5, length=0, pad=4)
    ax.grid(axis=grid, color=theme["grid"], linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.title.set_color(theme["ink"])
    ax.xaxis.label.set_color(theme["ink2"])
    ax.yaxis.label.set_color(theme["ink2"])


def _legend(ax, theme: dict, **kwargs) -> None:
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=theme["ink2"], handlelength=1.6, **kwargs)
    for text in leg.get_texts():
        text.set_color(theme["ink2"])


def _save(fig, path: Path, theme: dict) -> None:
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.2 / fig.get_figheight()))
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def _usd(x: float, _pos=None) -> str:
    return (
        f"${x:,.0f}" if x >= 10 else (f"${x:.1f}" if x >= 1 else (f"${x:.2f}" if x >= 0.1 else f"${x:.3f}"))
    )


def fig_quality(summary: dict, theme: dict, path: Path) -> None:
    tasks = list(summary["tasks"].values())
    n_sys = max(len(e["systems"]) for e in tasks)
    fig, axes = _figure(
        theme,
        len(tasks),
        "Accuratezza per scenario · punto = stima, barra = IC 95% bootstrap",
        height=1.3 + 0.55 * n_sys,
    )
    for ax, (name, entry) in zip(axes, summary["tasks"].items(), strict=True):
        _style(ax, theme, grid="x")
        systems = list(entry["systems"])
        for i, system in enumerate(systems):
            q = entry["systems"][system]["quality"]
            y = len(systems) - 1 - i
            lo, hi = q["accuracy_ci"]
            ax.plot([lo, hi], [y, y], color=color(system, theme), lw=2, solid_capstyle="round")
            ax.plot(q["accuracy"], y, "o", ms=8, color=color(system, theme), mec=theme["surface"], mew=1.2)
            ax.annotate(
                f"{q['accuracy']:.1%}",
                (hi, y),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color=theme["ink2"],
                fontsize=8.5,
            )
        ax.set_yticks(range(len(systems)), [label(s) for s in reversed(systems)])
        ax.set_ylim(-0.5, len(systems) - 0.5)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(SHORT_TITLES[name])
    _save(fig, path, theme)


def fig_latency(summary: dict, frames: dict, theme: dict, path: Path) -> None:
    fig, axes = _figure(
        theme, len(summary["tasks"]), "Distribuzione della latenza end-to-end (client → API → client)"
    )
    for k, (ax, (name, entry)) in enumerate(zip(axes, summary["tasks"].items(), strict=True)):
        _style(ax, theme)
        for system in entry["systems"]:
            df = frames[(name, system)]
            lat = np.sort(df.loc[df["ok"], "latency_ms"].to_numpy(float))
            ax.step(
                lat,
                np.arange(1, len(lat) + 1) / len(lat),
                where="post",
                lw=2,
                color=color(system, theme),
                label=label(system),
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:,.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("latenza (ms, scala log)")
        if k == 0:
            ax.set_ylabel("richieste completate entro x")
            _legend(ax, theme, loc="lower right")
        ax.set_title(SHORT_TITLES[name])
    _save(fig, path, theme)


def fig_tradeoff(summary: dict, theme: dict, path: Path) -> None:
    fig, axes = _figure(
        theme,
        len(summary["tasks"]),
        "Accuratezza vs costo · punti = sistemi, linee = cascata Jev → LLM al variare della soglia",
        height=3.8,
    )
    for k, (ax, (name, entry)) in enumerate(zip(axes, summary["tasks"].items(), strict=True)):
        _style(ax, theme)
        for fallback, cascade in entry.get("cascades", {}).items():
            sweep = pd.DataFrame(cascade["sweep"]).sort_values("tau")
            ax.plot(
                sweep["usd_per_1k"],
                sweep["accuracy"],
                lw=1.5,
                color=color(fallback, theme),
                alpha=0.85,
                label=f"Jev → {label(fallback)}",
            )
            op = cascade["parity"]
            ax.plot(
                op["usd_per_1k"],
                op["accuracy"],
                "D",
                ms=6,
                color=color(fallback, theme),
                mec=theme["surface"],
                mew=1.5,
            )
        for system, stats in entry["systems"].items():
            x, y = stats["cost"]["usd_per_1k"], stats["quality"]["accuracy"]
            ax.plot(
                x,
                y,
                "o",
                ms=9,
                color=color(system, theme),
                mec=theme["surface"],
                mew=2,
                label=label(system) if not entry.get("cascades") else None,
                zorder=5,
            )
            ax.annotate(
                label(system).split(" · ")[-1] if system != "jev" else "Jev",
                (x, y),
                xytext=(6, -12),
                textcoords="offset points",
                color=theme["ink2"],
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_usd))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("costo per 1.000 richieste (USD, scala log)")
        if k == 0:
            ax.set_ylabel("accuratezza")
            if ax.get_legend_handles_labels()[0]:
                _legend(ax, theme, loc="lower right")
        ax.set_title(SHORT_TITLES[name])
    _save(fig, path, theme)


def fig_calibration(summary: dict, theme: dict, path: Path) -> None:
    fig, axes = _figure(
        theme,
        len(summary["tasks"]),
        f"Calibrazione · accuratezza osservata per fascia di probabilità (fasce con ≥ {MIN_BIN} "
        "risposte, area del punto proporzionale al numero di risposte)",
    )
    for k, (ax, (name, entry)) in enumerate(zip(axes, summary["tasks"].items(), strict=True)):
        _style(ax, theme)
        ax.plot([0, 1], [0, 1], color=theme["muted"], lw=1, label="calibrazione perfetta")
        for system, stats in entry["systems"].items():
            cal = stats["calibration"]
            if not cal:
                continue
            rel = pd.DataFrame(cal["reliability"])
            rel = rel[rel["n"] >= MIN_BIN]  # tiny bins are noise
            ax.plot(
                rel["confidence"],
                rel["accuracy"],
                "-",
                lw=2,
                color=color(system, theme),
                label=f"{label(system)} (ECE {cal['ece']:.2f})",
            )
            ax.scatter(
                rel["confidence"],
                rel["accuracy"],
                s=18 + 220 * rel["n"] / rel["n"].sum(),
                color=color(system, theme),
                edgecolors=theme["surface"],
                linewidths=1.2,
                zorder=3,
            )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("probabilità assegnata alla risposta")
        if k == 0:
            ax.set_ylabel("risposte corrette")
        _legend(ax, theme, loc="upper left")
        ax.set_title(SHORT_TITLES[name])
    _save(fig, path, theme)


def fig_selective(summary: dict, frames: dict, theme: dict, path: Path) -> None:
    fig, axes = _figure(
        theme, len(summary["tasks"]), "Accuratezza vs copertura · si accettano solo le risposte più sicure"
    )
    for k, (ax, (name, entry)) in enumerate(zip(axes, summary["tasks"].items(), strict=True)):
        _style(ax, theme)
        task = TASKS[name]
        for system in entry["systems"]:
            signal = M.default_signal(task) if system == "jev" else "p_pred"
            curve = M.selective_curve(frames[(name, system)], signal)
            if curve is None:
                continue
            curve = curve[curve["coverage"] >= 0.05]  # the first few answers are noise
            ax.plot(
                curve["coverage"], curve["accuracy"], lw=2, color=color(system, theme), label=label(system)
            )
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("quota di richieste accettate (copertura)")
        if k == 0:
            ax.set_ylabel("accuratezza sulle richieste accettate")
            _legend(ax, theme, loc="lower left")
        ax.set_title(SHORT_TITLES[name])
    _save(fig, path, theme)


def fig_confusion(
    summary: dict, frames: dict, theme: dict, path: Path, task_name: str = "output_quality"
) -> None:
    entry = summary["tasks"][task_name]
    systems = list(entry["systems"])
    levels = TASKS[task_name].label_names
    fig, axes = _figure(
        theme,
        len(systems),
        "S3 · Matrici di confusione (righe = voto umano, colonne = voto del sistema)",
        height=3.6,
        width=3.6,
    )
    cmap = LinearSegmentedColormap.from_list("seq", theme["ramp"])
    for ax, system in zip(axes, systems, strict=True):
        df = frames[(task_name, system)]
        mat = pd.crosstab(df["gold"], df["pred"].fillna("err")).reindex(
            index=levels, columns=levels, fill_value=0
        )
        values = mat.to_numpy()
        ax.imshow(values, cmap=cmap, vmin=0, vmax=max(values.max(), 1))
        for i in range(len(levels)):
            for j in range(len(levels)):
                v = values[i, j]
                strong = v > values.max() * 0.55
                ink = ("#ffffff" if theme is THEMES["light"] else "#0b0b0b") if strong else theme["ink"]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5, color=ink)
        ax.set_xticks(range(len(levels)), levels)
        ax.set_yticks(range(len(levels)), levels)
        ax.tick_params(colors=theme["ink2"], length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("voto del sistema", color=theme["ink2"])
        ax.set_ylabel("voto umano", color=theme["ink2"])
        ax.set_title(label(system), color=theme["ink"])
    _save(fig, path, theme)


# --- markdown -------------------------------------------------------------------------------------


def _pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{x * 100:.{digits}f}%"


def _ci(ci) -> str:
    return f"[{ci[0] * 100:.1f}–{ci[1] * 100:.1f}]"


def _ms(x: float | None) -> str:
    return "–" if x is None else f"{x:,.0f}"


def _money(x: float) -> str:
    return f"${x:,.2f}" if x >= 1 else (f"${x:.3f}" if x >= 0.01 else f"${x:.5f}")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _picture(name: str, alt: str) -> str:
    return (
        f'<picture><source media="(prefers-color-scheme: dark)" srcset="figures/{name}-dark.png">'
        f'<img alt="{alt}" src="figures/{name}-light.png"></picture>'
    )


def render_markdown(summary: dict) -> str:
    tasks = summary["tasks"]
    systems_all = sorted(
        {s for e in tasks.values() for s in e["systems"]},
        key=lambda s: SYSTEM_ORDER.index(s) if s in SYSTEM_ORDER else 99,
    )
    models = sorted({m for e in tasks.values() for st in e["systems"].values() for m in st["models"]})
    has_jev = "jev" in systems_all
    lines = [
        f"# Risultati del benchmark · run `{summary['run_id']}`",
        "",
        f"> Generato da `jevbench report` il {summary['generated_at']} (UTC). "
        f"Modelli effettivamente serviti: {', '.join(f'`{m}`' for m in models)}.",
        "> Metodologia, limiti e interpretazione: [docs/03-metodologia.md](../../docs/03-metodologia.md) e "
        "[docs/04-risultati-e-raccomandazioni.md](../../docs/04-risultati-e-raccomandazioni.md).",
        "",
    ]
    if not has_jev:
        lines += [
            "> **Attenzione:** in questo run mancano le predizioni di Jev "
            "(serve `TYPESAFE_API_KEY` in `.env`, poi `jevbench run --systems jev`).",
            "",
        ]
    incomplete = [
        f"{t}/{s}" for t, e in tasks.items() for s, st in e["systems"].items() if not st["complete"]
    ]
    if incomplete:
        lines += [f"> **Run incompleto** per: {', '.join(incomplete)}.", ""]

    # Summary table
    lines += ["## Sintesi", ""]
    rows = []
    for name, entry in tasks.items():
        for system, st in entry["systems"].items():
            q = st["quality"]
            main = (
                f"QWK {q['qwk']:.3f}"
                if TASKS[name].kind is TaskKind.ORDINAL
                else f"macro-F1 {q['macro_f1']:.3f}"
            )
            rows.append(
                [
                    SHORT_TITLES[name],
                    label(system),
                    f"{_pct(q['accuracy'])} {_ci(q['accuracy_ci'])}",
                    main,
                    _ms(st["latency"]["p50_ms"]),
                    _ms(st["latency"]["p95_ms"]),
                    _money(st["cost"]["usd_per_1k"]),
                ]
            )
    lines += [
        _table(
            [
                "Scenario",
                "Sistema",
                "Accuratezza [IC 95%]",
                "Metrica secondaria",
                "Latenza p50 (ms)",
                "Latenza p95 (ms)",
                "Costo / 1.000 richieste",
            ],
            rows,
        ),
        "",
    ]

    lines += ["## Qualità", "", _picture("quality", "Accuratezza per scenario"), ""]
    for name, entry in tasks.items():
        task = TASKS[name]
        lines += [f"### {task.title}", "", f"Dataset: {task.dataset}. Opzioni: {len(task.labels)}.", ""]
        if task.kind is TaskKind.CHOICE:
            rows = [
                [
                    label(s),
                    str(st["quality"]["n"]),
                    str(st["quality"]["errors"]),
                    _pct(st["quality"]["accuracy"]),
                    _ci(st["quality"]["accuracy_ci"]),
                    f"{st['quality']['macro_f1']:.3f}",
                    f"[{st['quality']['macro_f1_ci'][0]:.3f}–{st['quality']['macro_f1_ci'][1]:.3f}]",
                ]
                for s, st in entry["systems"].items()
            ]
            lines += [
                _table(["Sistema", "n", "errori API", "Accuratezza", "IC 95%", "Macro-F1", "IC 95%"], rows),
                "",
            ]
        else:
            rows = [
                [
                    label(s),
                    f"{q['n_answered']}/{q['n']}",
                    _pct(q["accuracy"]),
                    _pct(q["within1"]),
                    f"{q['mae']:.2f}",
                    f"{q['qwk']:.3f} [{q['qwk_ci'][0]:.3f}–{q['qwk_ci'][1]:.3f}]",
                    f"{q['spearman']:.3f}",
                    _pct(q["pass_accuracy"]),
                    f"{q['pass_f1']:.3f}",
                    f"{q['mean_pred_level']:.2f} ({q['mean_gold_level']:.2f})",
                ]
                for s, st in entry["systems"].items()
                for q in [st["quality"]]
            ]
            lines += [
                _table(
                    [
                        "Sistema",
                        "Risposte",
                        "Esatta",
                        "Entro ±1",
                        "MAE",
                        "QWK [IC 95%]",
                        "Spearman",
                        "Gate ≥3: accuratezza",
                        "Gate ≥3: F1",
                        "Livello medio previsto (umano)",
                    ],
                    rows,
                ),
                "",
                "*Esatta* conta le richieste fallite come errori; le altre metriche sono calcolate sulle "
                "richieste con risposta. Spearman usa il punteggio atteso (media pesata dei livelli) quando "
                "disponibile. *Gate ≥3*: decisione binaria accetta/rifiuta (helpfulness ≥ 3).",
                "",
            ]
        if entry.get("mcnemar"):
            rows = [
                [f"Jev vs {label(s)}", str(r["a_only"]), str(r["b_only"]), f"{r['p_value']:.3g}"]
                for s, r in entry["mcnemar"].items()
            ]
            lines += [
                "Test di McNemar esatto (stessi esempi):",
                "",
                _table(["Confronto", "Solo Jev corretto", "Solo LLM corretto", "p-value"], rows),
                "",
            ]
    if "output_quality" in tasks and tasks["output_quality"]["systems"]:
        lines += [_picture("confusion", "Matrici di confusione S3"), ""]

    lines += ["## Latenza", "", _picture("latency", "Distribuzione della latenza"), ""]
    rows = [
        [
            SHORT_TITLES[n],
            label(s),
            _ms(st["latency"]["mean_ms"]),
            _ms(st["latency"]["p50_ms"]),
            _ms(st["latency"]["p90_ms"]),
            _ms(st["latency"]["p95_ms"]),
            _ms(st["latency"]["p99_ms"]),
            _ms(st["latency"]["server_p50_ms"]),
        ]
        for n, e in tasks.items()
        for s, st in e["systems"].items()
    ]
    lines += [
        _table(["Scenario", "Sistema", "Media", "p50", "p90", "p95", "p99", "Server p50"], rows),
        "",
        "Latenze in millisecondi, misurate lato client sul tentativo andato a buon fine (connessioni già aperte "
        "da una richiesta di warm-up). *Server p50* è il tempo di elaborazione dichiarato dal gateway "
        "TypeSafe (`x-envoy-upstream-service-time`); Azure non espone un equivalente.",
        "",
    ]

    lines += ["## Costo", ""]
    rows = [
        [
            SHORT_TITLES[n],
            label(s),
            _money(st["cost"]["usd_per_1k"]),
            f"{st['cost']['input_tokens_mean']:,.0f}",
            _pct(st["cost"]["cached_share"], 0),
            f"{st['cost']['output_tokens_mean']:,.0f}",
            f"{st['cost']['reasoning_tokens_mean']:,.0f}",
            _money(st["cost"]["usd_total"]),
        ]
        for n, e in tasks.items()
        for s, st in e["systems"].items()
    ]
    lines += [
        _table(
            [
                "Scenario",
                "Sistema",
                "USD / 1.000 richieste",
                "Token input medi",
                "Input in cache",
                "Token output medi",
                "di cui reasoning",
                "Costo del run",
            ],
            rows,
        ),
        "",
        f"Prezzi applicati: Jev {JEV_PRICING.input} $/M token input, output gratuito "
        f"([fonte]({JEV_PRICING.source.split(' ')[0]})); GPT-5.4 {GPT54_PRICING.input} $/M input, "
        f"{GPT54_PRICING.cached_input} $/M input in cache, {GPT54_PRICING.output} $/M output "
        f"([fonte]({GPT54_PRICING.source.split(' ')[0]})). I token di Jev e di GPT-5.4 sono contati da "
        "tokenizer diversi.",
        "",
    ]

    lines += [
        "## Calibrazione e confidenza",
        "",
        _picture("calibration", "Calibrazione"),
        "",
        _picture("selective", "Accuratezza vs copertura"),
        "",
    ]
    rows = [
        [
            SHORT_TITLES[n],
            label(s),
            f"{c['ece']:.3f}",
            f"{c['brier']:.3f}",
            "–" if c["auroc_correct"] is None else f"{c['auroc_correct']:.3f}",
            _pct(c["share_p_ge_099"], 0),
        ]
        for n, e in tasks.items()
        for s, st in e["systems"].items()
        if (c := st["calibration"])
    ]
    lines += [
        _table(
            ["Scenario", "Sistema", "ECE ↓", "Brier ↓", "AUROC giusto/sbagliato ↑", "Risposte con p ≥ 0,99"],
            rows,
        ),
        "",
        "La probabilità usata è quella assegnata alla risposta scelta: per Jev la `probabilities[choice]`, per "
        "GPT-5.4 senza reasoning la probabilità della sequenza di token della label (logprobs). Con reasoning "
        "attivo l'API non restituisce logprobs.",
        "",
    ]

    if has_jev:
        lines += [
            "## Combinazione: cascata Jev → LLM",
            "",
            _picture("tradeoff", "Accuratezza vs costo"),
            "",
            "Jev risponde a tutte le richieste; quelle con probabilità (S1, S2) o confidenza (S3) sotto la soglia "
            "τ vengono inoltrate all'LLM. Il punto operativo è la soglia più bassa che raggiunge l'accuratezza "
            "dell'LLM da solo (o -2 punti), scelta con *2-fold cross-fitting* (si sceglie τ su metà degli esempi "
            "e si misura sull'altra metà).",
            "",
        ]
        rows = []
        for name, entry in tasks.items():
            for fallback, c in entry.get("cascades", {}).items():
                llm = entry["systems"][fallback]
                for key, desc in (("parity", "parità con LLM"), ("near_parity", "LLM - 2 punti")):
                    op = c[key]
                    rows.append(
                        [
                            SHORT_TITLES[name],
                            f"Jev → {label(fallback)}",
                            desc,
                            ", ".join(f"{t:.3f}" for t in op["tau_folds"]),
                            _pct(op["escalation_rate"], 0),
                            f"{_pct(op['accuracy'])} (LLM {_pct(llm['quality']['accuracy'])})",
                            f"{_money(op['usd_per_1k'])} (LLM {_money(llm['cost']['usd_per_1k'])})",
                            f"{_ms(op['latency_mean_ms'])} (LLM {_ms(llm['latency']['mean_ms'])})",
                        ]
                    )
        lines += [
            _table(
                [
                    "Scenario",
                    "Cascata",
                    "Obiettivo",
                    "τ (fold 1, 2)",
                    "Inoltrate all'LLM",
                    "Accuratezza",
                    "USD / 1.000 richieste",
                    "Latenza media ms",
                ],
                rows,
            ),
            "",
        ]
    return "\n".join(lines) + "\n"


def build_report(run_id: str) -> Path:
    if not (RESULTS_DIR / run_id).exists():
        raise FileNotFoundError(f"results/{run_id} not found: run `jevbench run --run-id {run_id}` first")
    summary, frames = compute(run_id)
    out_dir = REPORTS_DIR / run_id
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        fig_quality(summary, theme, fig_dir / f"quality-{mode}.png")
        fig_latency(summary, frames, theme, fig_dir / f"latency-{mode}.png")
        fig_calibration(summary, theme, fig_dir / f"calibration-{mode}.png")
        fig_selective(summary, frames, theme, fig_dir / f"selective-{mode}.png")
        if summary["tasks"]["output_quality"]["systems"]:
            fig_confusion(summary, frames, theme, fig_dir / f"confusion-{mode}.png")
        if any(e.get("cascades") for e in summary["tasks"].values()):
            fig_tradeoff(summary, theme, fig_dir / f"tradeoff-{mode}.png")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8", newline="
")
    report = out_dir / "report.md"
    report.write_text(render_markdown(summary), encoding="utf-8", newline="
")
    return report
