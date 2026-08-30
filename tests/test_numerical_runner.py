import json
from pathlib import Path

from finqa_rag.data.models import EvidenceUnit, FinQAExample
from finqa_rag.evaluation.numerical_runner import (
    run_numerical_evaluation,
    select_locked_examples,
)
from finqa_rag.generation.qwen import GenerationResult


def example(example_id: str, question: str, answer: str, program: str) -> FinQAExample:
    evidence = EvidenceUnit(
        "table_1",
        "report",
        "test",
        "table",
        "Metric: Revenue | 2023: 100 | 2024: 120",
    )
    return FinQAExample(
        example_id=example_id,
        report_id="report",
        split="test",
        question=question,
        answer=answer,
        exe_answer=answer,
        program=program,
        table=(("Metric", "2023", "2024"), ("Revenue", "100", "120")),
        evidence=(evidence,),
        gold_evidence_ids=("table_1",),
    )


EXAMPLES = (
    example("q1", "What is 10 plus 5?", "15", "add(10, 5)"),
    example("q2", "What is 20 divided by 4?", "5", "divide(20, 4)"),
    example("text-only", "Why did revenue grow?", "demand", ""),
)


class FakeGenerator:
    model_id = "fake-qwen"

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def calculate_direct(self, question, evidence, program) -> GenerationResult:
        answer = self.answers[question]
        return GenerationResult(
            raw_text=f"model output: {answer}",
            parsed_text=answer,
            latency_ms=12.5,
        )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_locked_selection_is_hash_sorted_and_excludes_text_examples() -> None:
    first = select_locked_examples(EXAMPLES, seed=42, sample_size=2)
    second = select_locked_examples(tuple(reversed(EXAMPLES)), seed=42, sample_size=2)

    assert [item.example_id for item in first] == [item.example_id for item in second]
    assert {item.example_id for item in first} == {"q1", "q2"}


def test_numerical_runner_compares_identical_examples(tmp_path: Path) -> None:
    generator = FakeGenerator(
        {"What is 10 plus 5?": "15", "What is 20 divided by 4?": "wrong"}
    )

    summary = run_numerical_evaluation(
        EXAMPLES,
        generator,
        tmp_path,
        sample_size=2,
        seed=42,
    )

    assert summary["query_count"] == 2
    assert summary["conditions"]["direct_llm"]["exact_match_count"] == 1
    assert summary["conditions"]["python_tool"]["exact_match_count"] == 2
    predictions = read_jsonl(tmp_path / "predictions.jsonl")
    assert len(predictions) == 2
    assert all(row["tool_correct"] for row in predictions)
    manifest = json.loads((tmp_path / "experiment_manifest.json").read_text())
    assert manifest["sample_ids"] == [row["example_id"] for row in predictions]
