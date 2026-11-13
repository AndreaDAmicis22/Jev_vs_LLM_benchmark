"""Offline tests of the Jev integration through the real SDK with a mocked HTTP transport."""

from __future__ import annotations

import json

import httpx2
import pytest

from jev_benchmark.systems.jev import (
    PERM_ORDERS,
    QUESTION_ID,
    JevClassifier,
    JevPermutationClassifier,
    choice_confidence,
)
from jev_benchmark.tasks import Example, get_task


def _client_with(response_body, captured: list, jev: JevClassifier | None = None) -> JevClassifier:
    """`response_body` is a dict, or a callable building the response from the request body."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        captured.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        payload = response_body(body) if callable(response_body) else response_body
        return httpx2.Response(200, json=payload, headers={"x-envoy-upstream-service-time": "42"})

    jev = jev or JevClassifier(api_key="test-key", model="jev-1.14.0")
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


def _choice_answer(choice: str, labels: list[str], p_top: float = 0.7) -> dict:
    rest = (1 - p_top) / (len(labels) - 1)
    probs = {name: (p_top if name == choice else rest) for name in labels}
    return {
        "type": "choice",
        "choice": choice,
        "confidence": choice_confidence(probs),
        "probabilities": probs,
    }


async def test_option_order_is_sent_as_listed():
    task = get_task("workflow_next_step")
    example = Example(
        id="abcd-x", label="pull-up-account", state={"procedure": "p", "conversation_so_far": "c"}
    )
    captured: list = []
    body = {
        "model": "jev-1.14.0",
        "answers": {QUESTION_ID: _choice_answer("pull-up-account", task.label_names)},
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }
    jev = _client_with(body, captured, JevClassifier(api_key="k", model="jev-1.14.0", order="reversed"))
    pred = await jev.classify(task, example)
    await jev.aclose()
    criteria = captured[0]["body"]["questions"][QUESTION_ID]["criteria"]
    assert list(criteria) == task.label_names[::-1]  # JSON object keys keep the presentation order
    assert jev.name == "jev@reversed" and pred.system == "jev@reversed" and pred.order == "reversed"


def test_order_variants_skip_ordinal_scales():
    jev = JevClassifier(api_key="k", model="m", order="shuffled")
    assert jev.supports(get_task("intent_routing")) and not jev.supports(get_task("output_quality"))
    assert JevClassifier(api_key="k", model="m").supports(get_task("output_quality"))
    perm = JevPermutationClassifier(api_key="k", model="m")
    assert not perm.supports(get_task("output_quality"))
    with pytest.raises(ValueError):
        JevClassifier(api_key="k", model="m", order="sideways")


async def test_permutation_variant_averages_three_orders():
    task = get_task("intent_routing")
    labels = task.label_names
    example = Example(id="banking77-x", label="card_arrival", state={"customer_message": "where is my card"})
    choices = {"canonical": "card_arrival", "reversed": "card_arrival", "shuffled": "card_delivery_estimate"}

    def respond(body: dict) -> dict:
        answers = {qid: _choice_answer(choices[qid.split("_", 1)[1]], labels) for qid in body["questions"]}
        return {
            "model": "jev-1.14.0",
            "answers": answers,
            "usage": {"input_tokens": 3000, "output_tokens": 0},
        }

    captured: list = []
    jev = _client_with(respond, captured, JevPermutationClassifier(api_key="k", model="jev-1.14.0"))
    pred = await jev.classify(task, example)
    await jev.aclose()

    questions = captured[0]["body"]["questions"]
    assert len(captured) == 1 and set(questions) == {f"{QUESTION_ID}_{o}" for o in PERM_ORDERS}  # one request
    orders = {qid: list(q["criteria"]) for qid, q in questions.items()}
    assert orders[f"{QUESTION_ID}_canonical"] == labels
    assert orders[f"{QUESTION_ID}_reversed"] == labels[::-1]
    assert orders[f"{QUESTION_ID}_shuffled"] == list(task.ordered_labels("shuffled", example.id))
    # two orders say card_arrival with 0.7: averaged probability (0.7 + 0.7 + rest) / 3
    rest = 0.3 / (len(labels) - 1)
    assert pred.pred == "card_arrival" and pred.p_pred == pytest.approx((0.7 + 0.7 + rest) / 3)
    assert sum(pred.probabilities.values()) == pytest.approx(1.0)
    assert pred.extra["agree"] is False and pred.extra["preds"].count("card_arrival") == 2
    assert pred.system == "jev-perm" and pred.order == "perm"
    assert pred.cost_usd == pytest.approx(3000 * 0.042 / 1e6)


def test_choice_confidence_formula():
    assert choice_confidence({"a": 1.0, "b": 0.0}) == pytest.approx(1.0)
    assert choice_confidence({"a": 0.5, "b": 0.5}) == pytest.approx(0.0)
    assert choice_confidence({"a": 0.6, "b": 0.2, "c": 0.2}) == pytest.approx((0.6 - 1 / 3) / (2 / 3))
