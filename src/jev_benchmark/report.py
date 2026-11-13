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

from jev_benchmark import bias as B  # noqa: E402
from jev_benchmark import metrics as M  # noqa: E402
from jev_benchmark.config import (  # noqa: E402
    GPT54_MINI_PRICING,
    GPT54_PRICING,
    JEV_PRICING,
    REPORTS_DIR,
    RESULTS_DIR,
)
from jev_benchmark.tasks import TASKS, TaskKind  # noqa: E402

SYSTEM_ORDER = ["jev", "jev-perm", "gpt-5.4-none", "gpt-5.4-medium", "gpt-5.4-mini-none"]
SYSTEM_LABELS = {
    "jev": "Jev",
    "jev-perm": "Jev · media su 3 ordini",
    "gpt-5.4-none": "GPT-5.4 · reasoning none",
    "gpt-5.4-medium": "GPT-5.4 · reasoning medium",
    "gpt-5.4-mini-none": "GPT-5.4 mini · reasoning none",
}
FIRST_STAGES = ("jev", "jev-perm")  # systems that can open a cascade
MIN_BIN = 5  # minimum answers in a reliability bin to be drawn

SHORT_TITLES = {
    "intent_routing": "S1 · Richieste in ingresso",
    "workflow_next_step": "S2 · Passo successivo workflow",
    "output_quality": "S3 · Qualità output generato",
    "intent_routing_it": "S4 · Richieste in italiano",
}

# Reference categorical palette, colour fixed per entity (never by rank): Jev = slot 1 (blue),
# GPT-5.4 none = slot 2, GPT-5.4 medium = slot 3, GPT-5.4 mini = slot 4. jev-perm is the same model
# as Jev with another strategy: a darker step of the same blue ramp, plus its own marker and label.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "systems": {
            "jev": "#2a78d6",
            "jev-perm": "#184f95",
            "gpt-5.4-none": "#eb6834",
            "gpt-5.4-medium": "#1baf7a",
            "gpt-5.4-mini-none": "#eda100",
        },
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "systems": {
            "jev": "#3987e5",
            "jev-perm": "#9ec5f4",
            "gpt-5.4-none": "#d95926",
            "gpt-5.4-medium": "#199e70",
            "gpt-5.4-mini-none": "#c98500",
        },
        "ramp": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
    },
}
MARKERS = {"jev": "o", "jev-perm": "s", "gpt-5.4-none": "o", "gpt-5.4-medium": "^", "gpt-5.4-mini-none": "D"}

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
    return theme["systems"].get(system, theme["muted"])


def marker(system: str) -> str:
    return MARKERS.get(system, "o")


def linestyle(system: str) -> str:
    """Strategy variants of the same model are dashed (colour = model, dash = variant)."""
    return "--" if system in ("jev-perm", "gpt-5.4-medium") else "-"


# --- data -----------------------------------------------------------------------------------------


def discover_systems(run_id: str) -> list[str]:
    """Systems of the main comparison (option-order variants `x@order` go to the bias section)."""
    names: set[str] = set()
    for task in TASKS.values():
        folder = RESULTS_DIR / run_id / task.name
        if folder.exists():
            names |= {p.stem for p in folder.glob("*.jsonl") if "@" not in p.stem}
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
            entry["mcnemar"] = {}
            jev = frames[(task.name, "jev")]
            for system in entry["systems"]:
                if system == "jev":
                    continue
                other = frames[(task.name, system)]
                pair = jev[["example_id", "correct"]].merge(other[["example_id", "correct"]], on="example_id")
                entry["mcnemar"][system] = M.mcnemar_exact(
                    pair["correct_x"].to_numpy(bool), pair["correct_y"].to_numpy(bool)
                )
        llms = [s for s in entry["systems"] if not s.startswith("jev")]
        entry["cascades"] = {}
        for first in FIRST_STAGES:
            if first not in entry["systems"]:
                continue
            for fallback in llms:
                other = frames[(task.name, fallback)]
                m = M.paired(frames[(task.name, first)], other)
                signal = M.default_signal(task, first)
                llm_acc = float(other["correct"].mean())
                entry["cascades"][f"{first}->{fallback}"] = {
                    "first": first,
                    "fallback": fallback,
                    "signal": signal,
                    "parity": M.cascade_operating_point(m, signal, target_accuracy=llm_acc),
                    "near_parity": M.cascade_operating_point(m, signal, target_accuracy=llm_acc - 0.02),
                    "sweep": M.cascade_sweep(m, signal).to_dict(orient="records"),
                }
        summary["tasks"][task.name] = entry
    summary["bias"] = B.compute(run_id, list(TASKS.values()))
    return summary, frames


# --- figures --------------------------------------------------------------------------------------


def _active(summary: dict) -> dict:
    """Scenarios with at least one system (e.g. a new scenario not run yet is left out of the figures)."""
    return {name: entry for name, entry in summary["tasks"].items() if entry["systems"]}


def _figure(theme: dict, n: int, title: str, height: float = 3.5, width: float = 4.3):
    """One panel per scenario: a row for up to 3 panels, a 2x2 grid for 4 (legible when scaled down)."""
    # set_title() takes its colour from rcParams at call time (the left-aligned title is not ax.title)
    plt.rcParams["axes.titlecolor"] = theme["ink"]
    ncols = 2 if n == 4 else n
    nrows = -(-n // ncols)
    fig, grid = plt.subplots(
        nrows, ncols, figsize=(width * ncols, height * nrows), facecolor=theme["surface"], squeeze=False
    )
    axes = list(grid.flat)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(
        title, x=0.01, y=0.99, ha="left", va="top", color=theme["ink"], fontsize=12, fontweight="bold"
    )
    return fig, axes[:n]


def _first_col(ax) -> bool:
    return ax.get_subplotspec().is_first_col()


def _fig_legend(fig, axes, theme: dict, ncol: int = 3) -> None:
    """One legend for the whole figure, below the panels (series are the same in every panel)."""
    entries: dict[str, object] = {}
    for ax in axes:
        for handle, text in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(text, handle)
    if not entries:
        return
    ncol = min(ncol, len(entries))
    leg = fig.legend(
        list(entries.values()),
        list(entries),
        loc="lower center",
        ncol=ncol,
        frameon=False,
        fontsize=8.5,
        handlelength=1.8,
    )
    for text in leg.get_texts():
        text.set_color(theme["ink2"])
    fig._legend_margin_in = 0.2 + 0.24 * -(-len(entries) // ncol)


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
    bottom = getattr(fig, "_legend_margin_in", 0.0) / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.2 / fig.get_figheight()))
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def _usd(x: float, _pos=None) -> str:
    return (
        f"${x:,.0f}" if x >= 10 else (f"${x:.1f}" if x >= 1 else (f"${x:.2f}" if x >= 0.1 else f"${x:.3f}"))
    )


def fig_quality(summary: dict, theme: dict, path: Path) -> None:
    tasks = list(_active(summary).values())
    n_sys = max(len(e["systems"]) for e in tasks)
    fig, axes = _figure(
        theme,
        len(tasks),
        "Accuratezza per scenario · punto = stima, barra = IC 95% bootstrap",
        height=1.3 + 0.55 * n_sys,
    )
    for ax, (name, entry) in zip(axes, _active(summary).items(), strict=True):
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
        theme, len(_active(summary)), "Distribuzione della latenza end-to-end (client → API → client)"
    )
    for _k, (ax, (name, entry)) in enumerate(zip(axes, _active(summary).items(), strict=True)):
        _style(ax, theme)
        for system in entry["systems"]:
            df = frames[(name, system)]
            lat = np.sort(df.loc[df["ok"], "latency_ms"].to_numpy(float))
            ax.step(
                lat,
                np.arange(1, len(lat) + 1) / len(lat),
                where="post",
                lw=2,
                ls=linestyle(system),
                color=color(system, theme),
                label=label(system),
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:,.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("latenza (ms, scala log)")
        if _first_col(ax):
            ax.set_ylabel("richieste completate entro x")
        ax.set_title(SHORT_TITLES[name])
    _fig_legend(fig, axes, theme)
    _save(fig, path, theme)


def fig_tradeoff(summary: dict, theme: dict, path: Path) -> None:
    fig, axes = _figure(
        theme,
        len(_active(summary)),
        "Accuratezza vs costo · punti = sistemi, linee = cascata Jev → LLM al variare della soglia",
        height=3.8,
    )
    for _k, (ax, (name, entry)) in enumerate(zip(axes, _active(summary).items(), strict=True)):
        _style(ax, theme)
        for cascade in entry.get("cascades", {}).values():
            if cascade["first"] != "jev":  # jev-perm cascades are in the table only, to keep it legible
                continue
            fallback = cascade["fallback"]
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
                marker(system),
                ms=9,
                color=color(system, theme),
                mec=theme["surface"],
                mew=2,
                ls="none",
                label=label(system),
                zorder=5,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_usd))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("costo per 1.000 richieste (USD, scala log)")
        if _first_col(ax):
            ax.set_ylabel("accuratezza")
        ax.set_title(SHORT_TITLES[name])
    _fig_legend(fig, axes, theme)
    _save(fig, path, theme)


def fig_calibration(summary: dict, theme: dict, path: Path) -> None:
    fig, axes = _figure(
        theme,
        len(_active(summary)),
        f"Calibrazione · accuratezza osservata per fascia di probabilità (fasce con ≥ {MIN_BIN} "
        "risposte, area del punto proporzionale al numero di risposte)",
    )
    for _k, (ax, (name, entry)) in enumerate(zip(axes, _active(summary).items(), strict=True)):
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
        if _first_col(ax):
            ax.set_ylabel("risposte corrette")
        _legend(ax, theme, loc="upper left")
        ax.set_title(SHORT_TITLES[name])
    _save(fig, path, theme)


def fig_selective(summary: dict, frames: dict, theme: dict, path: Path) -> None:
    fig, axes = _figure(
        theme, len(_active(summary)), "Accuratezza vs copertura · si accettano solo le risposte più sicure"
    )
    for _k, (ax, (name, entry)) in enumerate(zip(axes, _active(summary).items(), strict=True)):
        _style(ax, theme)
        task = TASKS[name]
        for system in entry["systems"]:
            signal = M.default_signal(task, system) if system.startswith("jev") else "p_pred"
            curve = M.selective_curve(frames[(name, system)], signal)
            if curve is None:
                continue
            curve = curve[curve["coverage"] >= 0.05]  # the first few answers are noise
            ax.plot(
                curve["coverage"],
                curve["accuracy"],
                lw=2,
                ls=linestyle(system),
                color=color(system, theme),
                label=label(system),
            )
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("quota di richieste accettate (copertura)")
        if _first_col(ax):
            ax.set_ylabel("accuratezza sulle richieste accettate")
        ax.set_title(SHORT_TITLES[name])
    _fig_legend(fig, axes, theme)
    _save(fig, path, theme)


def fig_bias(summary: dict, theme: dict, path: Path) -> bool:
    """Selection rate by position decile vs gold rate, with per-example shuffled options."""
    names = [
        n
        for n, entry in summary.get("bias", {}).items()
        if any("shuffled" in entry.get(b, {}).get("orders", {}) for b in B.BIAS_BASE_SYSTEMS)
    ]
    if not names:
        return False
    fig, axes = _figure(
        theme,
        len(names),
        "Bias di posizione · quota di scelte per decile della lista diviso quota di risposte corrette "
        "(opzioni in ordine casuale; 1 = nessun bias)",
    )
    for _k, (ax, name) in enumerate(zip(axes, names, strict=True)):
        _style(ax, theme)
        ax.axhline(1.0, color=theme["muted"], lw=1)
        for base in B.BIAS_BASE_SYSTEMS:
            res = summary["bias"][name].get(base)
            if not res or "shuffled" not in res["orders"]:
                continue
            o = res["orders"]["shuffled"]
            ratio = [
                p / g if g else np.nan for p, g in zip(o["pred_deciles"], o["gold_deciles"], strict=True)
            ]
            ax.plot(range(1, 11), ratio, "-", lw=2, color=color(base, theme), label=label(base))
            ax.plot(
                range(1, 11),
                ratio,
                marker(base),
                ms=6,
                color=color(base, theme),
                mec=theme["surface"],
                mew=1.2,
            )
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("decile della posizione (1 = in cima alla lista)")
        if _first_col(ax):
            ax.set_ylabel("scelte / risposte corrette")
        ax.set_title(SHORT_TITLES[name])
    _fig_legend(fig, axes, theme)
    _save(fig, path, theme)
    return True


def _render_bias(summary: dict) -> list[str]:
    bias = summary.get("bias") or {}
    if not bias:
        return []
    order_names = {"canonical": "canonico", "reversed": "inverso", "shuffled": "casuale"}
    lines = [
        "## Bias di posizione (ordine delle opzioni)",
        "",
        "TypeSafe documenta che Jev tende a favorire la prima opzione di una Choice. Negli scenari Choice gli "
        "stessi esempi sono ripetuti con le opzioni in ordine **canonico**, **inverso** e **casuale** "
        "(permutazione diversa per ogni esempio, seed fisso). Con l'ordine casuale la posizione di un'opzione è "
        "indipendente dalla sua identità: se un sistema sceglie la prima posizione più spesso di quanto vi "
        "compaia la risposta corretta, l'eccesso è un puro effetto di posizione.",
        "",
    ]
    if (summary.get("_figures") or {}).get("bias"):
        lines += [_picture("bias", "Bias di posizione"), ""]
    for name, entry in bias.items():
        lines += [f"### {SHORT_TITLES.get(name, name)}", ""]
        rows, flips = [], []
        for base in B.BIAS_BASE_SYSTEMS:
            res = entry.get(base)
            if not res:
                continue
            for order, o in res["orders"].items():
                mc = res.get("mcnemar_vs_canonical", {}).get(order)
                delta = (
                    "–"
                    if order == "canonical" or mc is None
                    else f"{(o['accuracy'] - res['orders']['canonical']['accuracy']) * 100:+.1f} pt (p {mc['p_value']:.2g})"
                )
                rows.append(
                    [
                        label(base),
                        order_names[order],
                        f"{_pct(o['accuracy'])} {_ci(o['accuracy_ci'])}",
                        delta,
                        f"{_pct(o['pred_first'])} ({_pct(o['gold_first'])})",
                        f"{_pct(o['pred_top'])} ({_pct(o['gold_top'])})",
                    ]
                )
            if "consistency" in res:
                f = res.get("flips", {})
                p_flip, p_stable = f.get("median_p_flip"), f.get("median_p_stable")
                flips.append(
                    [
                        label(base),
                        str(res["n_common"]),
                        _pct(res["consistency"]),
                        str(f.get("n", 0)),
                        "–"
                        if p_flip is None
                        else f"{p_flip:.2f} / {p_stable:.2f}"
                        if p_stable is not None
                        else f"{p_flip:.2f}",
                        "–"
                        if f.get("share_flip_p_below_0_5") is None
                        else _pct(f["share_flip_p_below_0_5"], 0),
                        "–"
                        if f.get("auroc_low_p_flags_flip") is None
                        else f"{f['auroc_low_p_flags_flip']:.2f}",
                    ]
                )
        if rows:
            top_k = next(iter(entry[b] for b in B.BIAS_BASE_SYSTEMS if b in entry))["top_k"]
            lines += [
                _table(
                    [
                        "Sistema",
                        "Ordine",
                        "Accuratezza [IC 95%]",
                        "Δ vs canonico (McNemar)",
                        "Scelte in 1ª posizione (corrette lì)",
                        f"Scelte nelle prime {top_k} posizioni (corrette lì)",
                    ],
                    rows,
                ),
                "",
            ]
        if flips:
            lines += [
                _table(
                    [
                        "Sistema",
                        "Esempi nei 3 ordini",
                        "Stessa risposta in tutti gli ordini",
                        "Esempi che cambiano risposta",
                        "p mediana (cambiano / stabili)",
                        "Cambi con p < 0,5",
                        "AUROC: p bassa segnala il cambio",
                    ],
                    flips,
                ),
                "",
                "*p* è la probabilità della risposta nell'ordine canonico. Se gli esempi che cambiano risposta hanno "
                "p bassa, la soglia della cascata li inoltra già all'LLM.",
                "",
            ]
        if perm := entry.get("jev-perm"):
            mc = perm["mcnemar"]
            lines += [
                _table(
                    [
                        "Variante",
                        "Esempi",
                        "Accuratezza",
                        "Concordanza dei 3 ordini",
                        "Accuratezza se concordano / se no",
                    ],
                    [
                        ["Jev (ordine canonico)", str(perm["n"]), _pct(perm["accuracy_jev"]), "–", "–"],
                        [
                            "Jev · media su 3 ordini",
                            str(perm["n"]),
                            f"{_pct(perm['accuracy_perm'])} (McNemar p {mc['p_value']:.2g})",
                            _pct(perm["agreement"]),
                            f"{_pct(perm['accuracy_if_agree'])} / {_pct(perm['accuracy_if_disagree'])}",
                        ],
                    ],
                ),
                "",
            ]
    return lines


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

    missing = [SHORT_TITLES.get(n, n) for n, e in tasks.items() if not e["systems"]]
    if missing:
        lines += [f"Scenari ancora senza risultati: {', '.join(missing)}.", ""]

    lines += ["## Qualità", "", _picture("quality", "Accuratezza per scenario"), ""]
    for name, entry in tasks.items():
        if not entry["systems"]:
            continue
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
                _table(["Confronto", "Solo Jev corretto", "Solo l'altro sistema corretto", "p-value"], rows),
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
        f"([fonte]({GPT54_PRICING.source.split(' ')[0]})); GPT-5.4 mini {GPT54_MINI_PRICING.input} $/M input, "
        f"{GPT54_MINI_PRICING.cached_input} $/M in cache, {GPT54_MINI_PRICING.output} $/M output "
        f"([fonte]({GPT54_MINI_PRICING.source.split(' ')[0]})). I token di Jev e di GPT-5.4 sono contati da "
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
            "Jev risponde a tutte le richieste; quelle con probabilità (scenari Choice) o confidenza (S3) sotto la "
            "soglia τ vengono inoltrate all'LLM. Per *Jev · media su 3 ordini* viene inoltrata anche ogni richiesta "
            "su cui i tre ordini delle opzioni non concordano. Il punto operativo è la soglia più bassa che "
            "raggiunge l'accuratezza dell'LLM da solo (o -2 punti), scelta con *2-fold cross-fitting* (si sceglie "
            "τ su metà degli esempi e si misura sull'altra metà).",
            "",
        ]
        rows = []
        for name, entry in tasks.items():
            for c in entry.get("cascades", {}).values():
                fallback = c["fallback"]
                llm = entry["systems"][fallback]
                for key, desc in (("parity", "parità con LLM"), ("near_parity", "LLM - 2 punti")):
                    op = c[key]
                    rows.append(
                        [
                            SHORT_TITLES[name],
                            f"{label(c['first'])} → {label(fallback)}",
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
    lines += _render_bias(summary)
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
        summary.setdefault("_figures", {})["bias"] = fig_bias(summary, theme, fig_dir / f"bias-{mode}.png")
    summary_text = json.dumps(summary, indent=2, default=float)
    (out_dir / "summary.json").write_text(summary_text, encoding="utf-8", newline="\n")
    report = out_dir / "report.md"
    report.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    return report
