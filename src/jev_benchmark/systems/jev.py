"""Jev (TypeSafe AI System One model) through the official `typesafe-sdk`.

Mapping of the benchmark onto Jev primitives:

* CHOICE tasks  -> `Choice(instructions, criteria={label: description})`
* ORDINAL tasks -> `Score(instructions, criteria=[level_0, level_1, ...])`

The example `state` (a dict of named text fields) is sent as-is: Jev accepts JSON objects as state.
"""

from __future__ import annotations

import time

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    RetryPolicy,
    Score,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from jev_benchmark.config import JEV_PRICING
from jev_benchmark.systems.base import Prediction
from jev_benchmark.tasks.base import Example, Task, TaskKind

QUESTION_ID = "decision"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def build_question(task: Task) -> Choice | Score:
    if task.kind is TaskKind.CHOICE:
        return Choice(instructions=task.instructions, criteria=dict(task.labels))
    return Score(instructions=task.instructions, criteria=list(task.labels.values()))


class JevClassifier:
    name = "jev"

    def __init__(self, api_key: str, model: str, timeout: float = 60.0) -> None:
        if not api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set: add it to .env (keys: https://console.typesafe.ai/keys)"
            )
        self.model = model
        # Retries are disabled here: the runner retries, so latency is measured per attempt.
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            model=model,
            timeout=timeout,
            retry=RetryPolicy(max_retries=0, timeout=timeout),
        )

    def describe(self) -> dict:
        return {"provider": "TypeSafe AI", "model": self.model, "pricing": JEV_PRICING.__dict__}

    def is_retryable(self, exc: BaseException) -> bool:
        if isinstance(
            exc,
            TypeSafeRateLimitError
            | TypeSafeAPITimeoutError
            | TypeSafeAPIConnectionError
            | TypeSafeInternalServerError,
        ):
            return True
        return isinstance(exc, TypeSafeAPIError) and getattr(exc, "status", None) in RETRYABLE_STATUS

    async def classify(self, task: Task, example: Example) -> Prediction:
        question = build_question(task)
        start = time.perf_counter()
        response = await self._client.system_one(state=example.state, questions={QUESTION_ID: question})
        latency_ms = (time.perf_counter() - start) * 1000

        answer = response.answers[QUESTION_ID]
        expected = None
        if task.kind is TaskKind.CHOICE:
            probabilities = {str(k): float(v) for k, v in answer.probabilities.items()}
            pred = answer.choice
        else:
            probabilities = {str(k): float(v) for k, v in answer.probabilities.items()}
            pred = max(probabilities, key=probabilities.__getitem__)  # most likely level
            expected = float(answer.score)  # probability-weighted mean level

        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        return Prediction(
            task=task.name,
            system=self.name,
            example_id=example.id,
            gold=example.label,
            pred=pred,
            p_pred=probabilities.get(pred),
            confidence=float(answer.confidence),
            probabilities=probabilities,
            expected_score=expected,
            latency_ms=latency_ms,
            server_ms=_server_time_ms(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=JEV_PRICING.cost(input_tokens, output_tokens),
            model=response.model,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _server_time_ms(response) -> float | None:
    """Upstream processing time reported by TypeSafe's gateway (`x-envoy-upstream-service-time`)."""
    try:
        value = response.raw_http_response.headers.get("x-envoy-upstream-service-time")
        return float(value) if value is not None else None
    except Exception:  # header or raw response not available in this SDK version
        return None
