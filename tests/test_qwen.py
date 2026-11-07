import pytest
import numpy as np

from finqa_rag.generation.qwen import QwenGenerator, parse_program_response


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("divide(39, 4).", "divide(39, 4)"),
        (
            "The FinQA program is:\n```text\nsubtract(120, 100), divide(#0, 100)\n```",
            "subtract(120, 100), divide(#0, 100)",
        ),
        (
            "Use table_average(fourth quarter, none) to calculate it.",
            "table_average(fourth quarter, none)",
        ),
    ],
)
def test_parse_program_response_extracts_only_the_safe_grammar(
    raw: str,
    expected: str,
) -> None:
    assert parse_program_response(raw) == expected


def test_parse_program_response_preserves_invalid_text_for_safe_rejection() -> None:
    assert parse_program_response("I cannot calculate this.") == "I cannot calculate this."


def test_generate_prompt_exposes_generic_chat_generation() -> None:
    class FakeTokenizer:
        def __init__(self) -> None:
            self.messages: list[dict[str, str]] = []

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            self.messages = messages
            return "prompt"

        def __call__(self, prompts: list[str], *, return_tensors: str) -> dict[str, object]:
            return {"input_ids": np.array([[1, 2]])}

        def decode(self, tokens: object, *, skip_special_tokens: bool) -> str:
            return "model answer"

    class FakeModel:
        def generate(self, **kwargs: object) -> np.ndarray:
            return np.array([[1, 2, 3]])

    tokenizer = FakeTokenizer()
    generator = QwenGenerator(tokenizer=tokenizer, model=FakeModel())

    result = generator.generate_prompt("system instruction", "user question")

    assert result.raw_text == "model answer"
    assert tokenizer.messages == [
        {"role": "system", "content": "system instruction"},
        {"role": "user", "content": "user question"},
    ]
