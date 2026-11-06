"""Offline tests of the Jev integration through the real SDK with a mocked HTTP transport."""

from __future__ import annotations

import json

import httpx2
import pytest

from jev_benchmark.systems.jev import QUESTION_ID, JevClassifier
from jev_benchmark.tasks import Example, get_task


def _client_with(response_body: dict, captured: list) -> JevClassifier:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(
            {"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content)}
        )
        return httpx2.Response(200, json=response_body, headers={"x-envoy-upstream-service-time": "42"})

    jev = JevClassifier(api_key="test-key", model="jev-1.14.0")
    # Swap the transport of the SDK client: same request/response code path, no network.
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    jev._client = AsyncTypeSafeClient(
        api_key="test-key",
        model="jev-1.14.0",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
    )
    return jev


async def test_choice_request_and_parsing():
    task = get_task("intent_routing")
    example = Example(id="x", label="card_arrival", state={"customer_message": "Where is my card?"})
    captured: list = []
    probs = {name: 0.0 for name in task.label_names}
    probs |= {"card_arrival": 0.8, "card_delivery_estimate": 0.2}
    body = {
        "model": "jev-1.14.0",
        "answers": {
            QUESTION_ID: {
                "type": "choice",
                "choice": "card_arrival",
                "confidence": 0.79,
                "probabilities": probs,
            }
        },
        "usage": {"input_tokens": 1000, "output_tokens": 3},
    }
    jev = _client_with(body, captured)
    pred = await jev.classify(task, example)
    await jev.aclose()

    sent = captured[0]
    assert sent["url"].endswith("/v1/systemone")
    assert sent["headers"]["authorization"] == "Bearer test-key"
    question = sent["body"]["questions"][QUESTION_ID]
    assert question["type"] == "choice"
    assert question["instructions"] == task.instructions
    assert question["criteria"] == task.labels  # same options and descriptions as the LLM
    assert sent["body"]["state"] == example.state
    assert sent["body"]["model"] == "jev-1.14.0"

    assert pred.pred == "card_arrival" and pred.correct
    assert pred.p_pred == pytest.approx(0.8)
    assert pred.confidence == pytest.approx(0.79)
    assert pred.input_tokens == 1000
    assert pred.cost_usd == pytest.approx(1000 * 0.042 / 1e6)  # output tokens are free
    assert pred.server_ms == 42


async def test_score_request_and_parsing():
    task = get_task("output_quality")
    example = Example(id="y", label="3", state={"prompt": "hi", "response": "hello"})
    captured: list = []
    body = {
        "model": "jev-1.14.0",
        "answers": {
            QUESTION_ID: {
                "type": "score",
                "score": 2.6,
                "confidence": 0.41,
                "legend": {str(i): d for i, d in enumerate(task.labels.values())},
                "probabilities": {"0": 0.05, "1": 0.05, "2": 0.25, "3": 0.5, "4": 0.15},
            }
        },
        "usage": {"input_tokens": 200, "output_tokens": 0},
    }
    jev = _client_with(body, captured)
    pred = await jev.classify(task, example)
    await jev.aclose()

    question = captured[0]["body"]["questions"][QUESTION_ID]
    assert question["type"] == "score"
    assert question["criteria"] == list(task.labels.values())  # position = level

    assert pred.pred == "3"  # most likely level, not the rounded expectation
    assert pred.expected_score == pytest.approx(2.6)
    assert pred.p_pred == pytest.approx(0.5)
    assert set(pred.probabilities) == {"0", "1", "2", "3", "4"}


def test_missing_key_is_reported():
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        JevClassifier(api_key="", model="jev-1.14.0")
