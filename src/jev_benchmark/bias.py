"""Position bias of Choice answers: does the order of the options change the decision?

TypeSafe documents that Jev "tends to favour the first option" of a Choice (model jaggedness, "Choice
option order") without quantifying it. For every CHOICE scenario the same system is run with the
options in canonical, reversed and per-example shuffled order (`<system>@reversed`,
`<system>@shuffled`) and we measure:

* accuracy in each order and its paired difference from the canonical order (McNemar);
* consistency: share of examples with the same answer in all orders;
* over-selection of the first positions: share of answers that pick the option listed first (or in
  the first 10% of the list) compared with the share of correct answers that are listed there. With
  the shuffled order the position of an option is independent of its identity, so any excess is a
  pure position effect;
* for Jev, the probability mass by position (decile of the list) relative to an even spread;
* the confidence of the examples that change answer: if it is low, the cascade threshold already
  sends them to the LLM.

`jev-perm` (average over three orders in one request) is summarised against plain `jev`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jev_benchmark import metrics as M
from jev_benchmark.tasks import ORDERS, Task, TaskKind

BIAS_BASE_SYSTEMS = ("jev", "gpt-5.4-none")
DECILES = 10


def variant(base: str, order: str) -> str:
    return base if order == "canonical" else f"{base}@{order}"


def load_variants(run_id: str, task: Task, base: str) -> dict[str, pd.DataFrame]:
    frames = {order: M.load_frame(run_id, task, variant(base, order)) for order in ORDERS}
    return {order: df for order, df in frames.items() if df is not None}


def _positions(task: Task, df: pd.DataFrame, order: str) -> tuple[np.ndarray, np.ndarray]:
    """Position (0-based) of predicted and gold option in the list shown to the system."""
    pred_pos, gold_pos = [], []
    for ex_id, gold, pred in zip(df["example_id"], df["gold"], df["pred"], strict=True):
        names = list(task.ordered_labels(order, ex_id))
        gold_pos.append(names.index(gold))
        pred_pos.append(names.index(pred) if pred in names else -1)  # -1: failed request
    return np.array(pred_pos), np.array(gold_pos)


def _decile_shares(pos: np.ndarray, n_options: int) -> list[float]:
    deciles = np.minimum(pos * DECILES // n_options, DECILES - 1)
    return [float(np.mean(deciles == d)) for d in range(DECILES)]


def _mass_by_decile(task: Task, df: pd.DataFrame, order: str) -> list[float] | None:
    """Mean probability mass on each decile of the list, divided by the decile's share of options."""
    rows = df[df["ok"] & df["probabilities"].notna()]
    if rows.empty:
        return None
    n = len(task.labels)
    width = np.bincount(np.minimum(np.arange(n) * DECILES // n, DECILES - 1), minlength=DECILES) / n
    mass = np.zeros(DECILES)
    for ex_id, probs in zip(rows["example_id"], rows["probabilities"], strict=True):
        for pos, name in enumerate(task.ordered_labels(order, ex_id)):
            mass[min(pos * DECILES // n, DECILES - 1)] += probs.get(name, 0.0)
    mass /= len(rows)
    return [float(m / w) if w else None for m, w in zip(mass, width, strict=True)]


def analyse(task: Task, variants: dict[str, pd.DataFrame]) -> dict:
    n_options = len(task.labels)
    top_k = max(1, round(0.1 * n_options))
    result: dict = {"options": n_options, "top_k": top_k, "orders": {}}
    for order, df in variants.items():
        pred_pos, gold_pos = _positions(task, df, order)
        valid = pred_pos >= 0
        correct = df["correct"].to_numpy(bool)
        result["orders"][order] = {
            "n": len(df),
            "accuracy": float(correct.mean()),
            "accuracy_ci": M.bootstrap_ci(len(df), lambda idx, c=correct: c[idx].mean()),
            "pred_first": float(np.mean(pred_pos[valid] == 0)),
            "gold_first": float(np.mean(gold_pos == 0)),
            "pred_top": float(np.mean(pred_pos[valid] < top_k)),
            "gold_top": float(np.mean(gold_pos < top_k)),
            "pred_deciles": _decile_shares(pred_pos[valid], n_options),
            "gold_deciles": _decile_shares(gold_pos, n_options),
            "mass_deciles": _mass_by_decile(task, df, order),
        }
    if "canonical" not in variants or len(variants) < 2:
        return result

    common = sorted(set.intersection(*(set(df["example_id"]) for df in variants.values())))
    aligned = {o: df.set_index("example_id").loc[common] for o, df in variants.items()}
    preds = pd.DataFrame({o: a["pred"].fillna("__error__") for o, a in aligned.items()})
    stable = (preds.nunique(axis=1) == 1).to_numpy()
    base = aligned["canonical"]
    result["n_common"] = len(common)
    result["consistency"] = float(stable.mean())
    result["mcnemar_vs_canonical"] = {
        o: M.mcnemar_exact(a["correct"].to_numpy(bool), base["correct"].to_numpy(bool))
        for o, a in aligned.items()
        if o != "canonical"
    }
    p = base["p_pred"].astype(float).to_numpy()
    flip = ~stable
    flips: dict = {"n": int(flip.sum())}
    if flip.any() and not np.isnan(p).all():
        has_p = ~np.isnan(p)
        flips |= {
            "median_p_flip": float(np.nanmedian(p[flip])),
            "median_p_stable": float(np.nanmedian(p[stable])) if stable.any() else None,
            "share_flip_p_below_0_5": float(np.mean(p[flip & has_p] < 0.5)) if (flip & has_p).any() else None,
            "auroc_low_p_flags_flip": float(roc_auc_score(flip[has_p], -p[has_p]))
            if 0 < flip[has_p].mean() < 1
            else None,
        }
    result["flips"] = flips
    return result


def perm_summary(run_id: str, task: Task) -> dict | None:
    """jev-perm (average over 3 orders) vs single-order Jev on the same examples."""
    perm, jev = M.load_frame(run_id, task, "jev-perm"), M.load_frame(run_id, task, "jev")
    if perm is None or jev is None:
        return None
    pair = jev[["example_id", "correct"]].merge(perm[["example_id", "correct", "agree"]], on="example_id")
    agree = pair["agree"].fillna(False).to_numpy(bool)
    correct_perm = pair["correct_y"].to_numpy(bool)
    return {
        "n": len(pair),
        "accuracy_jev": float(pair["correct_x"].mean()),
        "accuracy_perm": float(correct_perm.mean()),
        "mcnemar": M.mcnemar_exact(correct_perm, pair["correct_x"].to_numpy(bool)),
        "agreement": float(agree.mean()),
        "accuracy_if_agree": float(correct_perm[agree].mean()) if agree.any() else None,
        "accuracy_if_disagree": float(correct_perm[~agree].mean()) if (~agree).any() else None,
    }


def compute(run_id: str, tasks: list[Task]) -> dict:
    out: dict = {}
    for task in tasks:
        if task.kind is not TaskKind.CHOICE:
            continue
        entry: dict = {}
        for base in BIAS_BASE_SYSTEMS:
            variants = load_variants(run_id, task, base)
            if len(variants) >= 2:  # nothing to compare with a single order
                entry[base] = analyse(task, variants)
        if (perm := perm_summary(run_id, task)) is not None:
            entry["jev-perm"] = perm
        if entry:
            out[task.name] = entry
    return out
