"""Structured model protocol and local/vLLM runtime adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from finqa_rag.generation.qwen import GenerationResult


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


def _not_blank(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be blank")
    return value


class ModelMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str

    _validate_content = field_validator("content")(_not_blank)


class ModelOutputError(ValueError):
    """Raised when a model response does not satisfy its declared contract."""


class StructuredModel(Protocol):
    def generate(
        self,
        schema: type[StructuredOutput],
        messages: Sequence[ModelMessage],
    ) -> StructuredOutput: ...

    def generate_text(self, messages: Sequence[ModelMessage]) -> str: ...


class PromptGenerator(Protocol):
    def generate_prompt(self, system: str, user: str) -> GenerationResult: ...


def _split_messages(messages: Sequence[ModelMessage]) -> tuple[str, str]:
    system_parts = [item.content for item in messages if item.role == "system"]
    conversation_parts = [
        f"{item.role}: {item.content}" for item in messages if item.role != "system"
    ]
    return "\n".join(system_parts), "\n".join(conversation_parts)


class TransformersModelAdapter:
    def __init__(self, generator: PromptGenerator) -> None:
        self._generator = generator

    def generate(
        self,
        schema: type[StructuredOutput],
        messages: Sequence[ModelMessage],
    ) -> StructuredOutput:
        system, user = _split_messages(messages)
        schema_instruction = (
            "Return only JSON matching this schema: "
            f"{schema.model_json_schema()}"
        )
        system = f"{system}\n{schema_instruction}" if system else schema_instruction
        raw = self._generator.generate_prompt(system, user).raw_text
        try:
            return schema.model_validate_json(raw)
        except (ValidationError, ValueError) as error:
            raise ModelOutputError(
                "model did not return valid structured output"
            ) from error

    def generate_text(self, messages: Sequence[ModelMessage]) -> str:
        system, user = _split_messages(messages)
        return self._generator.generate_prompt(system, user).raw_text


class VLLMOpenAIAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
        self._client = client

    @staticmethod
    def _payload(messages: Sequence[ModelMessage]) -> list[dict[str, str]]:
        return [
            {"role": item.role, "content": item.content}
            for item in messages
        ]

    @staticmethod
    def _content(response: object) -> str | None:
        try:
            return response.choices[0].message.content  # type: ignore[attr-defined]
        except (AttributeError, IndexError):
            return None

    def generate(
        self,
        schema: type[StructuredOutput],
        messages: Sequence[ModelMessage],
    ) -> StructuredOutput:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=self._payload(messages),
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            },
        )
        content = self._content(response)
        if content is None:
            raise ModelOutputError("model did not return valid structured output")
        try:
            return schema.model_validate_json(content)
        except (ValidationError, ValueError) as error:
            raise ModelOutputError(
                "model did not return valid structured output"
            ) from error

    def generate_text(self, messages: Sequence[ModelMessage]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=self._payload(messages),
            temperature=0,
        )
        content = self._content(response)
        if content is None:
            raise ModelOutputError("model did not return text output")
        return content
