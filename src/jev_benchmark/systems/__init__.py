"""Factory for the systems under test.

System names:
* `jev`                -> Jev via TypeSafe API (model from JEV_MODEL)
* `<deployment>-<effort>`, e.g. `gpt-5.4-none`, `gpt-5.4-medium` -> Foundry LLM baseline
"""

from __future__ import annotations

from jev_benchmark.config import Settings, get_settings
from jev_benchmark.systems.base import Classifier, Prediction
from jev_benchmark.systems.foundry import REASONING_EFFORTS, FoundryLLMClassifier
from jev_benchmark.systems.jev import JevClassifier

DEFAULT_SYSTEMS = "jev,gpt-5.4-none,gpt-5.4-medium"


def build_system(name: str, settings: Settings | None = None) -> Classifier:
    settings = settings or get_settings()
    if name == "jev":
        return JevClassifier(settings.typesafe_api_key.get_secret_value(), settings.jev_model)
    deployment, _, effort = name.rpartition("-")
    if deployment and effort in REASONING_EFFORTS:
        return FoundryLLMClassifier(settings, reasoning_effort=effort, deployment=deployment)
    raise ValueError(f"Unknown system {name!r}: use 'jev' or '<deployment>-<effort>' (e.g. gpt-5.4-none)")


def parse_systems(spec: str) -> list[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]


__all__ = ["DEFAULT_SYSTEMS", "Classifier", "Prediction", "build_system", "parse_systems"]
