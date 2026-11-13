"""Jev (TypeSafe AI System One model) through the official `typesafe-sdk`.

Mapping of the benchmark onto Jev primitives:

* CHOICE tasks  -> `Choice(instructions, criteria={label: description})`
* ORDINAL tasks -> `Score(instructions, criteria=[level_0, level_1, ...])`

The example `state` (a dict of named text fields) is sent as-is: Jev accepts JSON objects as state.

Two variants address the documented bias towards the first option of a Choice
(https://docs.typesafe.ai/model-jaggedness.md, "Choice option order"):

* `jev@reversed` / `jev@shuffled`: same request with the options listed in another order, to *measure*
  the bias;
* `jev-perm`: the same Choice is asked three times in one request, with the options in canonical,
  reversed and shuffled order, and the probabilities are averaged in code, to *mitigate* it.
  Questions of one request are evaluated in parallel and in isolation, so latency barely changes; the
  repeated options cost extra input tokens. Disagreement between the three answers is recorded as an
  extra uncertainty signal.
"""

from __future__ import annotations

import time
from statistics import fmean

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
from jev_benchmark.tasks.base import ORDERS, Example, Task, TaskKind

QUESTION_ID = "decision"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
PERM_ORDERS = ("canonical", "reversed", "shuffled")


def build_question(task: Task, labels: dict[str, str] | None = None) -> Choice | Score:
    """`labels` are the options in presentation order (default: canonical)."""
    labels = task.labels if labels is None else labels
    if task.kind is TaskKind.CHOICE:
        return Choice(instructions=task.instructions, criteria=dict(labels))
    return Score(instructions=task.instructions, criteria=list(labels.values()))


def choice_confidence(probabilities: dict[str, float]) -> float:
    """TypeSafe's Choice confidence: (p_max - 1/n) / (1 - 1/n)."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (max(probabilities.values()) - 1 / n) / (1 - 1 / n)))


class JevClassifier:
    def __init__(
        self,
        api_key: str,
        model: str,
        order: str = "canonical",
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set: add it to .env (keys: https://console.typesafe.ai/keys)"
            )
        if order not in ORDERS:
            raise ValueError(f"Unknown option order {order!r}: use one of {ORDERS}")
        self.model = model
        self.order = order
        self.base_url = base_url
        # Retries are disabled here: the runner retries, so latency is measured per attempt.
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            retry=RetryPolicy(max_retries=0, timeout=timeout),
        )

    @property
    def name(self) -> str:
        return "jev" if self.order == "canonical" else f"jev@{self.order}"

    def supports(self, task: Task) -> bool:
        # The levels of a Score are an ordered scale: reordering them is not meaningful.
        return self.order == "canonical" or task.kind is TaskKind.CHOICE

    def describe(self) -> dict:
        return {
            "provider": "TypeSafe AI",
            "model": self.model,
            "option_order": self.order,
            "base_url": self.base_url or "https://api.typesafe.ai",
            "pricing": JEV_PRICING.__dict__,
        }

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

    async def _ask(self, example: Example, questions: dict) -> tuple[object, float]:
        start = time.perf_counter()
        response = await self._client.system_one(state=example.state, questions=questions)
        return response, (time.perf_counter() - start) * 1000

    def _prediction(self, task: Task, example: Example, response, latency_ms: float, **fields) -> Prediction:
        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        return Prediction(
            task=task.name,
            system=self.name,
            example_id=example.id,
            gold=example.label,
            latency_ms=latency_ms,
            server_ms=_server_time_ms(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=JEV_PRICING.cost(input_tokens, output_tokens),
            model=response.model,
            **fields,
        )

    async def classify(self, task: Task, example: Example) -> Prediction:
        labels = task.ordered_labels(self.order, example.id)
        response, latency_ms = await self._ask(example, {QUESTION_ID: build_question(task, labels)})
        answer = response.answers[QUESTION_ID]
        probabilities = {str(k): float(v) for k, v in answer.probabilities.items()}
        if task.kind is TaskKind.CHOICE:
            pred, expected = answer.choice, None
        else:
            pred = max(probabilities, key=probabilities.__getitem__)  # most likely level
            expected = float(answer.score)  # probability-weighted mean level
        return self._prediction(
            task,
            example,
            response,
            latency_ms,
            pred=pred,
            p_pred=probabilities.get(pred),
            confidence=float(answer.confidence),
            probabilities=probabilities,
            expected_score=expected,
            order=self.order,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class JevPermutationClassifier(JevClassifier):
    """`jev-perm`: one request, the same Choice in three option orders, probabilities averaged."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None, timeout: float = 60.0) -> None:
        super().__init__(api_key, model, order="canonical", base_url=base_url, timeout=timeout)

    @property
    def name(self) -> str:
        return "jev-perm"

    def supports(self, task: Task) -> bool:
        return task.kind is TaskKind.CHOICE

    def describe(self) -> dict:
        return {**super().describe(), "option_order": "average of " + ", ".join(PERM_ORDERS)}

    async def classify(self, task: Task, example: Example) -> Prediction:
        if task.kind is not TaskKind.CHOICE:
            raise ValueError("jev-perm only applies to Choice tasks")
        ids = [f"{QUESTION_ID}_{order}" for order in PERM_ORDERS]
        questions = {
            qid: build_question(task, task.ordered_labels(order, example.id))
            for qid, order in zip(ids, PERM_ORDERS, strict=True)
        }
        response, latency_ms = await self._ask(example, questions)
        answers = [response.answers[qid] for qid in ids]
        per_order = [{str(k): float(v) for k, v in a.probabilities.items()} for a in answers]
        averaged = {label: fmean(p.get(label, 0.0) for p in per_order) for label in task.label_names}
        pred = max(task.label_names, key=averaged.__getitem__)  # ties -> canonical order
        preds = [a.choice for a in answers]
        return self._prediction(
            task,
            example,
            response,
            latency_ms,
            pred=pred,
            p_pred=averaged[pred],
            confidence=choice_confidence(averaged),
            probabilities=averaged,
            order="perm",
            extra={
                "orders": list(PERM_ORDERS),
                "preds": preds,
                "p_choice_by_order": [p.get(c) for p, c in zip(per_order, preds, strict=True)],
                "agree": len(set(preds)) == 1,
            },
        )


def _server_time_ms(response) -> float | None:
    """Upstream processing time reported by TypeSafe's gateway (`x-envoy-upstream-service-time`)."""
    try:
        value = response.raw_http_response.headers.get("x-envoy-upstream-service-time")
        return float(value) if value is not None else None
    except Exception:  # header or raw response not available in this SDK version
        return None
