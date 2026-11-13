"""Quality, calibration, latency, cost and cascade metrics computed from saved predictions.

Every metric is computed on the full benchmark set of a task: failed requests count as wrong
answers (conservative) and are reported separately.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from jev_benchmark.runner import load_predictions, results_path
from jev_benchmark.tasks import Task, TaskKind
from jev_benchmark.tasks.output_quality import PASS_THRESHOLD

N_BOOT = 2000


# --- loading --------------------------------------------------------------------------------------


def load_frame(run_id: str, task: Task, system: str) -> pd.DataFrame | None:
    """Predictions of one system on one task, aligned to the benchmark examples."""
    preds = load_predictions(results_path(run_id, task.name, system))
    if not preds:
        return None
    by_id = {p.example_id: p for p in preds}
    rows = []
    for ex in task.load():
        p = by_id.get(ex.id)
        if p is None:
            continue
        agree = (p.extra or {}).get("agree")  # jev-perm: do the three option orders agree?
        rows.append(
            {
                "example_id": ex.id,
                "gold": ex.label,
                "pred": p.pred,
                "ok": p.ok,
                "correct": p.correct,
                "p_pred": p.p_pred,
                # cascade signal of jev-perm: disagreement between orders forces escalation
                "p_pred_agree": 0.0 if agree is False else p.p_pred,
                "agree": agree,
                "confidence": p.confidence,
                "probabilities": p.probabilities,
                "order": p.order,
                "expected_score": p.expected_score,
                "latency_ms": p.latency_ms,
                "server_ms": p.server_ms,
                "input_tokens": p.input_tokens,
                "cached_tokens": p.cached_tokens,
                "output_tokens": p.output_tokens,
                "reasoning_tokens": p.reasoning_tokens,
                "cost_usd": p.cost_usd,
                "model": p.model,
                "error": p.error,
            }
        )
    return pd.DataFrame(rows) if rows else None


# --- statistics helpers ---------------------------------------------------------------------------


def bootstrap_ci(
    n: int, statistic: Callable[[np.ndarray], float], n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap CI of `statistic(indices)` resampling the n examples."""
    rng = np.random.default_rng(seed)
    values = [statistic(rng.integers(0, n, n)) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired correctness (same examples, two systems)."""
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    if a_only + b_only == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    p = binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue
    return {"a_only": a_only, "b_only": b_only, "p_value": float(p)}


# --- quality --------------------------------------------------------------------------------------


def quality(df: pd.DataFrame, task: Task) -> dict:
    gold = df["gold"].to_numpy()
    pred = df["pred"].fillna("__error__").to_numpy()
    correct = df["correct"].to_numpy(dtype=bool)
    n = len(df)
    out = {
        "n": n,
        "errors": int((~df["ok"]).sum()),
        "accuracy": float(correct.mean()),
        "accuracy_ci": bootstrap_ci(n, lambda idx: correct[idx].mean()),
    }
    if task.kind is TaskKind.CHOICE:

        def macro_f1(idx: np.ndarray) -> float:
            # Average over the classes present in this (re)sample, gold or predicted: with few examples
            # per class a bootstrap resample misses classes, which must not count as F1 = 0.
            labels = sorted((set(gold[idx]) | set(pred[idx])) - {"__error__"})
            return f1_score(gold[idx], pred[idx], labels=labels, average="macro", zero_division=0)

        out["macro_f1"] = float(macro_f1(np.arange(n)))
        out["macro_f1_ci"] = bootstrap_ci(n, macro_f1)
        return out

    # ORDINAL: levels "0".."K-1". Exact accuracy above counts failed requests as wrong; distance and
    # correlation metrics are computed on the answered examples only (`n_answered`).
    valid = df["ok"].to_numpy(dtype=bool)
    g = gold[valid].astype(int)
    p = pred[valid].astype(int)
    expected = df.loc[valid, "expected_score"].astype(float).to_numpy()
    m = len(g)

    def qwk(idx: np.ndarray) -> float:
        return cohen_kappa_score(g[idx], p[idx], weights="quadratic", labels=list(range(len(task.labels))))

    passed_gold, passed_pred = g >= PASS_THRESHOLD, p >= PASS_THRESHOLD
    out |= {
        "n_answered": int(m),
        "within1": float((np.abs(g - p) <= 1).mean()),
        "mae": float(np.abs(g - p).mean()),
        "qwk": float(qwk(np.arange(m))),
        "qwk_ci": bootstrap_ci(m, qwk),
        "spearman": float(spearmanr(g, expected).statistic),
        "pass_accuracy": float((passed_gold == passed_pred).mean()),
        "pass_f1": float(f1_score(passed_gold, passed_pred, zero_division=0)),
        "mean_pred_level": float(p.mean()),
        "mean_gold_level": float(g.mean()),
    }
    return out


# --- calibration ----------------------------------------------------------------------------------


def calibration(df: pd.DataFrame, n_bins: int = 10) -> dict | None:
    """Reliability of `p_pred` (probability given to the predicted label)."""
    sub = df[df["ok"] & df["p_pred"].notna()]
    if len(sub) < 20:
        return None
    p = sub["p_pred"].to_numpy(dtype=float)
    y = sub["correct"].to_numpy(dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    table, ece = [], 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            conf, acc = p[mask].mean(), y[mask].mean()
            ece += mask.mean() * abs(acc - conf)
            table.append({"bin": b, "n": int(mask.sum()), "confidence": float(conf), "accuracy": float(acc)})
    auroc = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 and len(np.unique(p)) > 1 else None
    return {
        "ece": float(ece),
        "brier": float(np.mean((p - y) ** 2)),
        "auroc_correct": auroc,  # how well the probability separates right from wrong answers
        "share_p_ge_099": float((p >= 0.99).mean()),
        "reliability": table,
    }


def selective_curve(df: pd.DataFrame, signal: str = "p_pred") -> pd.DataFrame | None:
    """Accuracy on the answers kept when only the most confident fraction is accepted.

    Tie-aware: answers with the same signal value cannot be separated by any threshold, so inside a
    tie group the curve uses the expected accuracy under random tie-breaking (an LLM that gives
    p = 1.0 to 90% of its answers gets a flat curve over that 90%).
    """
    sub = df[df[signal].notna()]
    if sub.empty:
        return None
    groups = (
        sub.assign(_s=sub[signal].astype(float).round(6))
        .groupby("_s")["correct"]
        .agg(["sum", "count"])
        .sort_index(ascending=False)
    )
    counts, hits = groups["count"].to_numpy(), groups["sum"].to_numpy(dtype=float)
    group_of = np.repeat(np.arange(len(groups)), counts)  # position (0-based) -> tie group
    before_n = np.concatenate([[0], np.cumsum(counts)[:-1]])
    before_hits = np.concatenate([[0.0], np.cumsum(hits)[:-1]])
    kept = np.arange(1, len(sub) + 1)
    expected_hits = before_hits[group_of] + (kept - before_n[group_of]) * (hits / counts)[group_of]
    return pd.DataFrame({"coverage": kept / len(df), "accuracy": expected_hits / kept})


# --- latency & cost -------------------------------------------------------------------------------


def latency(df: pd.DataFrame) -> dict:
    lat = df.loc[df["ok"], "latency_ms"].astype(float)
    server = df.loc[df["ok"], "server_ms"].dropna().astype(float)
    pct = lambda s, q: float(np.percentile(s, q)) if len(s) else None  # noqa: E731
    return {
        "mean_ms": float(lat.mean()),
        "p50_ms": pct(lat, 50),
        "p90_ms": pct(lat, 90),
        "p95_ms": pct(lat, 95),
        "p99_ms": pct(lat, 99),
        "server_p50_ms": pct(server, 50),
    }


def cost(df: pd.DataFrame) -> dict:
    ok = df[df["ok"]]
    return {
        "usd_per_1k": float(ok["cost_usd"].mean() * 1000),
        "usd_total": float(df["cost_usd"].sum()),
        "input_tokens_mean": float(ok["input_tokens"].mean()),
        "cached_share": float(ok["cached_tokens"].sum() / max(ok["input_tokens"].sum(), 1)),
        "output_tokens_mean": float(ok["output_tokens"].mean()),
        "reasoning_tokens_mean": float(ok["reasoning_tokens"].mean()),
    }


# --- cascade: Jev first, escalate uncertain cases to the LLM ---------------------------------------


@dataclass(frozen=True)
class CascadeSpec:
    first: str
    fallback: str
    signal: str  # column of the first system compared with the threshold


def default_signal(task: Task, system: str = "jev") -> str:
    """TypeSafe guidance: threshold the probability of the chosen option for Choice questions,
    and the (ordinal-aware) confidence for Score questions. For jev-perm, disagreement between the
    three option orders also counts as uncertainty."""
    if task.kind is not TaskKind.CHOICE:
        return "confidence"
    return "p_pred_agree" if system == "jev-perm" else "p_pred"


def paired(first: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Join two systems on the same examples (suffix `_a` = first system, `_b` = fallback)."""
    cols = ["example_id", "gold", "correct", "p_pred", "p_pred_agree", "confidence", "latency_ms", "cost_usd"]
    return first[cols].merge(fallback[cols], on=["example_id", "gold"], suffixes=("_a", "_b"))


def cascade_at(m: pd.DataFrame, signal: str, tau: float) -> dict:
    """`answer = first if signal >= tau else fallback` on a paired frame.

    Uses the latencies and costs actually measured for both systems on each example; an escalated
    request pays both (sequential cascade). A failed first answer is always escalated.
    """
    sig = m[f"{signal}_a"].astype(float).fillna(-1.0).to_numpy()
    esc = sig < tau
    correct = np.where(esc, m["correct_b"], m["correct_a"]).astype(float)
    lat = m["latency_ms_a"].fillna(0).to_numpy() + np.where(esc, m["latency_ms_b"].fillna(0), 0.0)
    usd = m["cost_usd_a"].to_numpy() + np.where(esc, m["cost_usd_b"], 0.0)
    return {
        "escalation_rate": float(esc.mean()),
        "accuracy": float(correct.mean()),
        "usd_per_1k": float(usd.mean() * 1000),
        "latency_mean_ms": float(lat.mean()),
        "latency_p95_ms": float(np.percentile(lat, 95)),
    }


def cascade_sweep(m: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Cascade metrics for every threshold where the decision changes (0 = never escalate)."""
    sig = m[f"{signal}_a"].astype(float).dropna().to_numpy()
    thresholds = np.unique(np.concatenate([[0.0], sig, [1.0 + 1e-9]]))
    return pd.DataFrame([{"tau": float(t), **cascade_at(m, signal, t)} for t in thresholds])


def cascade_operating_point(m: pd.DataFrame, signal: str, target_accuracy: float, seed: int = 0) -> dict:
    """Least-escalating threshold that reaches `target_accuracy`, estimated with 2-fold cross-fitting.

    The threshold is chosen on one half of the examples and evaluated on the other half (and vice
    versa), so the reported operating point is not tuned on the data it is evaluated on. If the
    target is never reached, the most accurate threshold is used.
    """
    fold = np.random.default_rng(seed).permutation(len(m)) % 2
    folds = []
    for k in (0, 1):
        tune, test = m[fold != k], m[fold == k]
        sweep = cascade_sweep(tune, signal)
        reach = sweep[sweep["accuracy"] >= target_accuracy]
        best = (
            reach.sort_values("escalation_rate")
            if len(reach)
            else sweep.sort_values(["accuracy", "escalation_rate"], ascending=[False, True])
        )
        tau = float(best.iloc[0]["tau"])
        folds.append({"tau": tau, **cascade_at(test, signal, tau)})
    out = {key: float(np.mean([f[key] for f in folds])) for key in folds[0] if key != "tau"}
    out["tau_folds"] = [f["tau"] for f in folds]
    return out
