from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import BaseModel

from finqa_rag.agents.model import (
    ModelMessage,
    ModelOutputError,
    TransformersModelAdapter,
    VLLMOpenAIAdapter,
)
from finqa_rag.generation.qwen import GenerationResult


class RouteDecision(BaseModel):
    route: Literal["text", "numerical"]


class FakePromptGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = iter(outputs)

    def generate_prompt(self, system: str, user: str) -> GenerationResult:
        raw = next(self._outputs)
        return GenerationResult(raw_text=raw, parsed_text=raw, latency_ms=1.0)


def test_transformers_adapter_validates_structured_json() -> None:
    adapter = TransformersModelAdapter(
        FakePromptGenerator(['{"route":"numerical"}'])
    )

    result = adapter.generate(
        RouteDecision,
        [ModelMessage(role="user", content="Classify this question")],
    )

    assert result == RouteDecision(route="numerical")


def test_transformers_adapter_rejects_invalid_json_without_repair() -> None:
    adapter = TransformersModelAdapter(
        FakePromptGenerator(["route=numerical"])
    )

    with pytest.raises(ModelOutputError, match="valid structured output"):
        adapter.generate(
            RouteDecision,
            [ModelMessage(role="user", content="Classify this question")],
        )


def test_transformers_adapter_returns_raw_text_unchanged() -> None:
    adapter = TransformersModelAdapter(
        FakePromptGenerator(["  evidence-based answer [text_1]  "])
    )

    result = adapter.generate_text(
        [ModelMessage(role="user", content="Answer the question")]
    )

    assert result == "  evidence-based answer [text_1]  "


class FakeCompletions:
    def __init__(self, contents: list[str | None]) -> None:
        self._contents = iter(contents)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        content = next(self._contents)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAIClient:
    def __init__(self, contents: list[str | None]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(contents))


def test_vllm_adapter_sends_json_schema_request_contract() -> None:
    client = FakeOpenAIClient(['{"route":"text"}'])
    adapter = VLLMOpenAIAdapter(
        base_url="http://vllm:8000/v1",
        api_key="test-key",
        model="Qwen/Qwen3-4B-Instruct-2507",
        timeout=17.0,
        client=client,
    )

    result = adapter.generate(
        RouteDecision,
        [
            ModelMessage(role="system", content="Return a route."),
            ModelMessage(role="user", content="Why did revenue rise?"),
        ],
    )

    assert result == RouteDecision(route="text")
    assert adapter.base_url == "http://vllm:8000/v1"
    assert adapter.timeout == 17.0
    assert client.chat.completions.calls == [
        {
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "messages": [
                {"role": "system", "content": "Return a route."},
                {"role": "user", "content": "Why did revenue rise?"},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "RouteDecision",
                    "schema": RouteDecision.model_json_schema(),
                },
            },
        }
    ]


@pytest.mark.parametrize("content", ["not-json", None])
def test_vllm_adapter_rejects_malformed_or_missing_content(
    content: str | None,
) -> None:
    adapter = VLLMOpenAIAdapter(
        base_url="http://vllm:8000/v1",
        api_key="test-key",
        model="model",
        client=FakeOpenAIClient([content]),
    )

    with pytest.raises(ModelOutputError, match="valid structured output"):
        adapter.generate(
            RouteDecision,
            [ModelMessage(role="user", content="Classify")],
        )


def test_vllm_text_generation_does_not_request_schema() -> None:
    client = FakeOpenAIClient(["raw answer"])
    adapter = VLLMOpenAIAdapter(
        base_url="http://vllm:8000/v1",
        api_key="test-key",
        model="model",
        client=client,
    )

    result = adapter.generate_text(
        [ModelMessage(role="user", content="Answer")]
    )

    assert result == "raw answer"
    assert "response_format" not in client.chat.completions.calls[0]
