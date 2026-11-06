"""Generative LLM baseline: GPT-5.4 served by Microsoft Foundry (Azure OpenAI Responses API).

The answer is forced into the option set with Structured Outputs (strict JSON schema with an `enum`),
so the LLM can only return a valid label. Without reasoning (`effort="none"`) the API also returns
token logprobs, from which we derive the probability of the chosen label: this gives the LLM a
confidence signal comparable to Jev's probabilities.
"""

from __future__ import annotations

import json
import math
import time

import openai
from openai import AsyncAzureOpenAI

from jev_benchmark.config import GPT54_PRICING, Settings
from jev_benchmark.systems.base import Prediction
from jev_benchmark.tasks.base import Example, Task, TaskKind, render_state_text

TOP_LOGPROBS = 10
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")


def answer_field(task: Task) -> str:
    return "label" if task.kind is TaskKind.CHOICE else "score"


def build_system_prompt(task: Task) -> str:
    options = "\n".join(f"- {name}: {description}" for name, description in task.labels.items())
    header = "Choose exactly one label from this list:" if task.kind is TaskKind.CHOICE else "Use this scale:"
    return (
        f"{task.instructions}\n\n{header}\n{options}\n\n"
        "The input to evaluate is in the user message. Reply only with the JSON object required by "
        "the response schema."
    )


def build_response_format(task: Task) -> dict:
    field = answer_field(task)
    if task.kind is TaskKind.CHOICE:
        prop = {"type": "string", "enum": task.label_names}
    else:
        prop = {"type": "integer", "enum": [int(level) for level in task.label_names]}
    return {
        "type": "json_schema",
        "name": f"{task.name}_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {field: prop},
            "required": [field],
            "additionalProperties": False,
        },
    }


class FoundryLLMClassifier:
    def __init__(
        self,
        settings: Settings,
        reasoning_effort: str = "none",
        deployment: str | None = None,
        logprobs: bool | None = None,
        timeout: float = 180.0,
    ) -> None:
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
        if not settings.azure_openai_endpoint or not settings.azure_openai_api_key.get_secret_value():
            raise RuntimeError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY are not set in .env")
        self.deployment = deployment or settings.azure_openai_deployment
        self.effort = reasoning_effort
        self.name = f"{self.deployment}-{reasoning_effort}"
        # Logprobs are only available when the model does not reason.
        self.logprobs = (reasoning_effort == "none") if logprobs is None else logprobs
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key.get_secret_value(),
            api_version=settings.azure_openai_api_version,
            max_retries=0,  # retries are handled by the runner
            timeout=timeout,
        )
        self.api_version = settings.azure_openai_api_version

    def describe(self) -> dict:
        return {
            "provider": "Microsoft Foundry (Azure OpenAI Responses API)",
            "deployment": self.deployment,
            "api_version": self.api_version,
            "reasoning_effort": self.effort,
            "logprobs": self.logprobs,
            "pricing": GPT54_PRICING.__dict__,
        }

    def is_retryable(self, exc: BaseException) -> bool:
        return isinstance(
            exc,
            openai.RateLimitError
            | openai.APITimeoutError
            | openai.APIConnectionError
            | openai.InternalServerError,
        )

    def request_kwargs(self, task: Task, example: Example) -> dict:
        kwargs: dict = {
            "model": self.deployment,
            "input": [
                {"role": "system", "content": build_system_prompt(task)},
                {"role": "user", "content": render_state_text(example.state)},
            ],
            "text": {"format": build_response_format(task)},
            "reasoning": {"effort": self.effort},
            "max_output_tokens": 16_000,
            "store": False,
        }
        if self.logprobs:
            kwargs["include"] = ["message.output_text.logprobs"]
            kwargs["top_logprobs"] = TOP_LOGPROBS
        return kwargs

    async def classify(self, task: Task, example: Example) -> Prediction:
        kwargs = self.request_kwargs(task, example)
        start = time.perf_counter()
        raw = await self._client.responses.with_raw_response.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response = raw.parse()

        if response.status != "completed":
            reason = getattr(response.incomplete_details, "reason", None)
            raise RuntimeError(f"response status {response.status} ({reason})")

        field = answer_field(task)
        value = json.loads(response.output_text)[field]
        pred = str(value)
        tokens = _output_logprobs(response)
        p_pred, probabilities, expected = None, None, None
        if tokens:
            span = _value_span("".join(t.token for t in tokens), field)
            if span:
                if task.kind is TaskKind.CHOICE:
                    logprob = sum(t.logprob for t in _tokens_in_span(tokens, span))
                    p_pred = min(1.0, math.exp(logprob))  # API logprobs can round to tiny positives
                else:
                    probabilities = _level_distribution(tokens, span, task.label_names)
                    p_pred = probabilities.get(pred)
                    expected = sum(int(level) * p for level, p in probabilities.items())

        usage = response.usage
        input_tokens = usage.input_tokens
        cached = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
        output_tokens = usage.output_tokens
        reasoning = getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0
        return Prediction(
            task=task.name,
            system=self.name,
            example_id=example.id,
            gold=example.label,
            pred=pred,
            p_pred=p_pred,
            probabilities=probabilities,
            expected_score=expected
            if expected is not None
            else (float(value) if task.kind is TaskKind.ORDINAL else None),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            cached_tokens=cached,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning,
            cost_usd=GPT54_PRICING.cost(input_tokens, output_tokens, cached),
            model=raw.headers.get("x-ms-served-model") or response.model,
            raw=response.output_text,
        )

    async def aclose(self) -> None:
        await self._client.close()


# --- logprob helpers ---------------------------------------------------------------------------


def _output_logprobs(response) -> list:
    tokens = []
    for item in response.output:
        if item.type != "message":
            continue
        for part in item.content:
            tokens.extend(getattr(part, "logprobs", None) or [])
    return tokens


def _value_span(text: str, field: str) -> tuple[int, int] | None:
    """Character span of the JSON value of `field` (without quotes) inside the generated text."""
    key = text.find(f'"{field}"')
    if key < 0:
        return None
    i = text.find(":", key) + 1
    while i < len(text) and text[i] in " \t\n":
        i += 1
    if i >= len(text):
        return None
    if text[i] == '"':
        end = text.find('"', i + 1)
        return (i + 1, end) if end > i else None
    end = i
    while end < len(text) and (text[end].isdigit() or text[end] in "-."):
        end += 1
    return (i, end) if end > i else None


def _tokens_in_span(tokens: list, span: tuple[int, int]) -> list:
    """Tokens overlapping the character span (boundary tokens are forced by the schema)."""
    selected, pos = [], 0
    for tok in tokens:
        start, pos = pos, pos + len(tok.token)
        if start < span[1] and pos > span[0]:
            selected.append(tok)
    return selected


def _level_distribution(tokens: list, span: tuple[int, int], levels: list[str]) -> dict[str, float]:
    """Distribution over ordinal levels from the alternatives of the first value token."""
    first = _tokens_in_span(tokens, span)[0]
    candidates = [(first.token, first.logprob)] + [
        (alt.token, alt.logprob) for alt in (getattr(first, "top_logprobs", None) or [])
    ]
    mass: dict[str, float] = {}
    for token, logprob in candidates:
        level = token.strip()
        if level in levels and level not in mass:
            mass[level] = math.exp(logprob)
    total = sum(mass.values()) or 1.0
    return {level: mass.get(level, 0.0) / total for level in levels}
