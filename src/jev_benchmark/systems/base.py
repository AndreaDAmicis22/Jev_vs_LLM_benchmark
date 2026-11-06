"""Common interface of the systems under test."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field

from jev_benchmark.tasks.base import Example, Task


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Prediction(BaseModel):
    """Outcome of one request of one system on one example."""

    task: str
    system: str
    example_id: str
    gold: str
    pred: str | None = None
    # Probability the system assigns to its own prediction (Jev: probabilities[pred];
    # LLM: exp(sum of the logprobs of the label tokens), only without reasoning).
    p_pred: float | None = None
    # Native confidence reported by Jev (concentration of the distribution), None for the LLM.
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    expected_score: float | None = None  # ordinal tasks only
    latency_ms: float | None = None  # client wall clock of the successful attempt
    total_ms: float | None = None  # including failed attempts and back-off
    server_ms: float | None = None  # server-side processing time, when the API exposes it
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    model: str | None = None
    attempts: int = 1
    error: str | None = None
    raw: str | None = None
    timestamp: str = Field(default_factory=_now)

    @property
    def ok(self) -> bool:
        return self.error is None and self.pred is not None

    @property
    def correct(self) -> bool:
        return self.ok and self.pred == self.gold


class Classifier(Protocol):
    """A system able to answer the decision of a `Task` for one `Example`."""

    name: str

    async def classify(self, task: Task, example: Example) -> Prediction:
        """Single attempt; raises on failure (retries are handled by the runner)."""
        ...

    def is_retryable(self, exc: BaseException) -> bool: ...

    async def aclose(self) -> None: ...

    def describe(self) -> dict: ...
