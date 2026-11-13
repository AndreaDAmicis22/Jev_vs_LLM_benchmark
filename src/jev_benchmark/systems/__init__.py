"""Factory for the systems under test.

System names:

* `jev`                         -> Jev via TypeSafe API (model from JEV_MODEL)
* `jev-perm`                    -> Jev, same Choice in 3 option orders in one request, averaged
* `<deployment>-<effort>`       -> Foundry LLM, e.g. `gpt-5.4-none`, `gpt-5.4-medium`, `gpt-5.4-mini-none`
* `<system>@<order>`            -> same system with the options `reversed` or `shuffled` (bias study)

Presets: `main` (systems compared in the report), `bias` (order variants), `all`.
"""

from __future__ import annotations

from jev_benchmark.config import Settings, get_settings
from jev_benchmark.systems.base import Classifier, Prediction
from jev_benchmark.systems.foundry import REASONING_EFFORTS, FoundryLLMClassifier
from jev_benchmark.systems.jev import JevClassifier, JevPermutationClassifier
from jev_benchmark.tasks.base import ORDERS

MAIN_SYSTEMS = ["jev", "jev-perm", "gpt-5.4-none", "gpt-5.4-medium", "gpt-5.4-mini-none"]
BIAS_SYSTEMS = [f"{base}@{order}" for base in ("jev", "gpt-5.4-none") for order in ORDERS[1:]]
PRESETS = {"main": MAIN_SYSTEMS, "bias": BIAS_SYSTEMS, "all": MAIN_SYSTEMS + BIAS_SYSTEMS}
DEFAULT_SYSTEMS = "main"


def build_system(name: str, settings: Settings | None = None) -> Classifier:
    settings = settings or get_settings()
    base, _, order = name.partition("@")
    order = order or "canonical"
    if order not in ORDERS:
        raise ValueError(f"Unknown option order {order!r} in {name!r}: use one of {ORDERS}")
    jev_args = (settings.typesafe_api_key.get_secret_value(), settings.jev_model)
    if base == "jev":
        return JevClassifier(*jev_args, order=order, base_url=settings.typesafe_base_url)
    if base == "jev-perm":
        if order != "canonical":
            raise ValueError("jev-perm already averages over option orders: no @order suffix")
        return JevPermutationClassifier(*jev_args, base_url=settings.typesafe_base_url)
    deployment, _, effort = base.rpartition("-")
    if deployment and effort in REASONING_EFFORTS:
        return FoundryLLMClassifier(settings, reasoning_effort=effort, deployment=deployment, order=order)
    raise ValueError(
        f"Unknown system {name!r}: use 'jev', 'jev-perm' or '<deployment>-<effort>' "
        "(e.g. gpt-5.4-mini-none), optionally with '@reversed' or '@shuffled'"
    )


def parse_systems(spec: str) -> list[str]:
    """Comma separated system names and/or presets (`main`, `bias`, `all`), duplicates removed."""
    names: list[str] = []
    for item in (s.strip() for s in spec.split(",")):
        if item:
            names.extend(PRESETS.get(item, [item]))
    return list(dict.fromkeys(names))


__all__ = [
    "BIAS_SYSTEMS",
    "DEFAULT_SYSTEMS",
    "MAIN_SYSTEMS",
    "PRESETS",
    "Classifier",
    "Prediction",
    "build_system",
    "parse_systems",
]
