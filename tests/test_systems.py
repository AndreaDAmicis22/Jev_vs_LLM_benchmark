"""System factory, presets and the Foundry client (order, pricing, logprobs fallback), offline."""

from __future__ import annotations

from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from jev_benchmark.config import GPT54_MINI_PRICING, GPT54_PRICING, Settings, llm_pricing
from jev_benchmark.systems import BIAS_SYSTEMS, MAIN_SYSTEMS, build_system, parse_systems
from jev_benchmark.systems.foundry import FoundryLLMClassifier, build_response_format, build_system_prompt
from jev_benchmark.tasks import Example, get_task

FAKE = Settings(
    azure_openai_endpoint="https://example.cognitiveservices.azure.com/",
    azure_openai_api_key="fake",
    typesafe_api_key="fake",
)


def test_presets_expand_and_deduplicate():
    assert parse_systems("main") == MAIN_SYSTEMS
    assert parse_systems("bias") == BIAS_SYSTEMS
    assert parse_systems("jev,main,bias,jev") == MAIN_SYSTEMS + BIAS_SYSTEMS
    assert "gpt-5.4-mini-none" in MAIN_SYSTEMS and "jev-perm" in MAIN_SYSTEMS
    assert set(BIAS_SYSTEMS) == {
        "jev@reversed",
        "jev@shuffled",
        "gpt-5.4-none@reversed",
        "gpt-5.4-none@shuffled",
    }


@pytest.mark.parametrize(
    ("name", "cls_name", "order"),
    [
        ("jev", "JevClassifier", "canonical"),
        ("jev@shuffled", "JevClassifier", "shuffled"),
        ("jev-perm", "JevPermutationClassifier", "canonical"),
        ("gpt-5.4-none", "FoundryLLMClassifier", "canonical"),
        ("gpt-5.4-mini-none", "FoundryLLMClassifier", "canonical"),
        ("gpt-5.4-none@reversed", "FoundryLLMClassifier", "reversed"),
    ],
)
async def test_build_system(name, cls_name, order):
    system = build_system(name, FAKE)
    try:
        assert type(system).__name__ == cls_name and system.name == name and system.order == order
    finally:
        await system.aclose()


@pytest.mark.parametrize("name", ["jev-perm@reversed", "jev@sideways", "gpt-5.4-turbo", "gpt-9-none"])
def test_invalid_system_names(name):
    with pytest.raises((ValueError, KeyError)):
        build_system(name, FAKE)


def test_pricing_per_deployment():
    assert llm_pricing("gpt-5.4") is GPT54_PRICING and llm_pricing("gpt-5.4-mini") is GPT54_MINI_PRICING
    assert GPT54_MINI_PRICING.cost(1_000_000, 0) == pytest.approx(0.75)
    with pytest.raises(KeyError, match="LLM_PRICING"):
        llm_pricing("gpt-4o")


def test_prompt_and_enum_follow_the_option_order():
    task = get_task("intent_routing_it")
    labels = task.ordered_labels("reversed", "x")
    prompt = build_system_prompt(task, labels)
    first, last = task.label_names[0], task.label_names[-1]
    assert prompt.index(f"- {last}:") < prompt.index(f"- {first}:")
    assert build_response_format(task, labels)["schema"]["properties"]["label"]["enum"] == list(labels)
    assert prompt.startswith(task.instructions)  # Italian instructions for the Italian scenario


def _fake_raw(text: str, input_tokens: int = 100, output_tokens: int = 5):
    response = NS(
        status="completed",
        incomplete_details=None,
        output_text=text,
        output=[],
        model="gpt-5.4-mini",
        usage=NS(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=NS(cached_tokens=0),
            output_tokens_details=NS(reasoning_tokens=0),
        ),
    )
    return NS(headers={"x-ms-served-model": "gpt-5.4-mini-2026-03-17"}, parse=lambda: response)


def _bad_request(message: str) -> openai.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
    return openai.BadRequestError(message, response=response, body=None)


async def test_logprobs_refused_are_switched_off_and_the_request_repeated():
    clf = FoundryLLMClassifier(FAKE, reasoning_effort="none", deployment="gpt-5.4-mini")
    calls = []

    async def fake_create(kwargs):
        calls.append(kwargs)
        if "include" in kwargs:
            raise _bad_request("logprobs are not supported with this model")
        return _fake_raw('{"label":"card_arrival"}'), 12.0

    clf._create = fake_create
    task = get_task("intent_routing")
    pred = await clf.classify(
        task, Example(id="x", label="card_arrival", state={"customer_message": "card?"})
    )
    await clf.aclose()
    assert len(calls) == 2 and "include" not in calls[1]
    assert clf.logprobs is False and "logprobs" in clf.describe()["logprobs_note"]
    assert pred.correct and pred.p_pred is None and pred.latency_ms == 12.0
    assert pred.cost_usd == pytest.approx((100 * 0.75 + 5 * 4.50) / 1e6)  # GPT-5.4 mini price
    assert pred.model == "gpt-5.4-mini-2026-03-17"


async def test_other_bad_requests_are_not_retried_without_logprobs():
    clf = FoundryLLMClassifier(FAKE, reasoning_effort="none", deployment="gpt-5.4")

    async def fake_create(kwargs):
        raise _bad_request("The response was filtered due to the prompt triggering the content policy")

    clf._create = fake_create
    with pytest.raises(openai.BadRequestError):
        await clf.classify(
            get_task("intent_routing"), Example(id="x", label="card_arrival", state={"m": "?"})
        )
    await clf.aclose()
    assert clf.logprobs is True
