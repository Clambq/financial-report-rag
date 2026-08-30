"""Controlled direct-Qwen versus safe-Python arithmetic evaluation."""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

from finqa_rag.common.atomic import atomic_write_json, atomic_write_jsonl
from finqa_rag.data.download import FINQA_REVISION
from finqa_rag.data.models import EvidenceUnit, FinQAExample
from finqa_rag.evaluation.numerical_metrics import (
    aggregate_numerical,
    answers_match,
    normalize_answer,
)
from finqa_rag.generation.qwen import GenerationResult
from finqa_rag.reasoning.program import ProgramError, execute_program


class DirectCalculator(Protocol):
    model_id: str

    def calculate_direct(
        self,
        question: str,
        evidence: Sequence[EvidenceUnit],
        program: str,
    ) -> GenerationResult: ...


def select_locked_examples(
    examples: Sequence[FinQAExample],
    *,
    seed: int,
    sample_size: int,
) -> list[FinQAExample]:
    eligible = [item for item in examples if item.program and item.exe_answer]
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(eligible):
        raise ValueError(
            f"sample_size {sample_size} exceeds {len(eligible)} eligible examples"
        )
    return sorted(
        eligible,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item.example_id}".encode("utf-8")
        ).hexdigest(),
    )[:sample_size]


def run_numerical_evaluation(
    examples: Sequence[FinQAExample],
    generator: DirectCalculator,
    output_dir: Path,
    *,
    sample_size: int = 100,
    seed: int = 42,
) -> dict[str, object]:
    selected = select_locked_examples(examples, seed=seed, sample_size=sample_size)
    comparison_records: list[dict[str, bool | float]] = []
    predictions: list[dict[str, object]] = []

    for example in selected:
        by_id = {item.evidence_id: item for item in example.evidence}
        gold_evidence = [by_id[item] for item in example.gold_evidence_ids]
        direct = generator.calculate_direct(
            example.question,
            gold_evidence,
            example.program,
        )
        direct_correct = answers_match(direct.parsed_text, example.exe_answer)
        try:
            normalize_answer(direct.parsed_text)
            direct_parse_error = False
        except ValueError:
            direct_parse_error = True

        tool_started = perf_counter_ns()
        tool_error: str | None = None
        try:
            tool_value = execute_program(example.program, table=example.table)
            tool_answer = str(tool_value)
            tool_correct = answers_match(tool_value, example.exe_answer)
        except ProgramError as error:
            tool_answer = ""
            tool_correct = False
            tool_error = str(error)
        tool_latency_ms = (perf_counter_ns() - tool_started) / 1_000_000

        if tool_error:
            error_category = "tool_execution_error"
        elif direct_parse_error:
            error_category = "direct_parse_error"
        elif not direct_correct:
            error_category = "direct_arithmetic_error"
        else:
            error_category = "none"
        comparison_records.append(
            {
                "direct_correct": direct_correct,
                "tool_correct": tool_correct,
                "direct_latency_ms": direct.latency_ms,
                "tool_latency_ms": tool_latency_ms,
            }
        )
        predictions.append(
            {
                "example_id": example.example_id,
                "report_id": example.report_id,
                "question": example.question,
                "evidence_ids": list(example.gold_evidence_ids),
                "program": example.program,
                "gold_answer": example.exe_answer,
                "direct_raw_answer": direct.raw_text,
                "direct_predicted_answer": direct.parsed_text,
                "direct_correct": direct_correct,
                "direct_latency_ms": direct.latency_ms,
                "tool_answer": tool_answer,
                "tool_correct": tool_correct,
                "tool_latency_ms": tool_latency_ms,
                "tool_error": tool_error,
                "error_category": error_category,
            }
        )

    summary = {
        "split": selected[0].split,
        "model_id": generator.model_id,
        **aggregate_numerical(comparison_records),
    }
    manifest: dict[str, object] = {
        "finqa_revision": FINQA_REVISION,
        "split": selected[0].split,
        "model_id": generator.model_id,
        "seed": seed,
        "sample_size": sample_size,
        "sample_ids": [item.example_id for item in selected],
        "max_new_tokens": getattr(generator, "max_new_tokens", None),
        "do_sample": getattr(generator, "do_sample", None),
        "python_version": platform.python_version(),
        "comparison_design": "same gold evidence and gold program for both conditions",
    }
    atomic_write_json(output_dir / "metrics_summary.json", summary)
    atomic_write_jsonl(output_dir / "predictions.jsonl", predictions)
    atomic_write_json(output_dir / "experiment_manifest.json", manifest)
    return summary
