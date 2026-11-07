from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from finqa_rag.agents.tools import (
    ComposeGroundedReportOutput,
    ExecuteFinancialProgramInput,
    ExecuteFinancialProgramOutput,
    FinancialProgramTool,
    QueryReportTableInput,
    ReportTableTool,
    RetrieveEvidenceInput,
    RetrievalTool,
    ToolFailure,
    ToolRegistry,
    VerifyGroundingInput,
)
from finqa_rag.data.models import EvidenceUnit
from finqa_rag.retrieval.types import ScoredEvidence


class EchoInput(BaseModel):
    value: int


class EchoOutput(BaseModel):
    value: int


def test_tool_inputs_reject_blank_ids_invalid_top_k_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RetrieveEvidenceInput(report_id=" ", query="revenue", top_k=5)
    with pytest.raises(ValidationError):
        RetrieveEvidenceInput(report_id="r1", query="revenue", top_k=0)
    with pytest.raises(ValidationError):
        QueryReportTableInput(
            report_id="r1",
            row_selector="Revenue",
            unexpected=True,
        )


def test_registry_rejects_duplicate_registration() -> None:
    registry = ToolRegistry()
    registry.register("echo", EchoInput, EchoOutput, lambda value: value)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("echo", EchoInput, EchoOutput, lambda value: value)


def test_registry_rejects_unknown_tool_before_execution() -> None:
    registry = ToolRegistry()

    result = asyncio.run(registry.ainvoke("missing", {"value": 1}))

    assert result == ToolFailure(
        tool_name="missing",
        code="unknown_tool",
        message="unknown tool: missing",
        retryable=False,
    )


def test_registry_rejects_invalid_payload_before_handler() -> None:
    calls = 0

    def handler(value: EchoInput) -> EchoOutput:
        nonlocal calls
        calls += 1
        return EchoOutput(value=value.value)

    registry = ToolRegistry()
    registry.register("echo", EchoInput, EchoOutput, handler)

    result = asyncio.run(registry.ainvoke("echo", {"value": "not-an-int"}))

    assert isinstance(result, ToolFailure)
    assert result.code == "validation_error"
    assert calls == 0


def test_registry_supports_sync_and_async_handlers() -> None:
    async def async_handler(value: EchoInput) -> EchoOutput:
        return EchoOutput(value=value.value + 1)

    registry = ToolRegistry()
    registry.register(
        "sync",
        EchoInput,
        EchoOutput,
        lambda value: EchoOutput(value=value.value),
    )
    registry.register("async", EchoInput, EchoOutput, async_handler)

    assert asyncio.run(registry.ainvoke("sync", {"value": 2})) == EchoOutput(value=2)
    assert asyncio.run(registry.ainvoke("async", {"value": 2})) == EchoOutput(value=3)


def test_report_contract_rejects_citations_outside_allowed_evidence() -> None:
    with pytest.raises(ValidationError, match="outside allowed evidence"):
        ComposeGroundedReportOutput(
            answer="Revenue increased [text_9].",
            citation_ids=["text_9"],
            allowed_evidence_ids=["text_1"],
        )


def test_verifier_contract_rejects_citations_outside_allowed_evidence() -> None:
    with pytest.raises(ValidationError, match="outside allowed evidence"):
        VerifyGroundingInput(
            draft_answer="Revenue increased [text_9].",
            citation_ids=["text_9"],
            allowed_evidence_ids=["text_1"],
            calculations=[],
        )


class FakeRetriever:
    def search(
        self,
        report_id: str,
        question: str,
    ) -> list[ScoredEvidence]:
        if report_id == "missing":
            raise KeyError(report_id)
        return [
            ScoredEvidence(
                EvidenceUnit(
                    evidence_id="table_1",
                    report_id=report_id,
                    split="test",
                    kind="table",
                    text="Metric: Revenue | 2024: 120",
                ),
                score=0.91,
                rank=1,
                method="hybrid_rerank",
            )
        ]


def test_retrieval_tool_preserves_ranked_evidence() -> None:
    result = RetrievalTool(FakeRetriever())(
        RetrieveEvidenceInput(report_id="report-a", query="revenue", top_k=5)
    )

    assert not isinstance(result, ToolFailure)
    assert result.evidence[0].evidence_id == "table_1"
    assert result.evidence[0].rank == 1
    assert result.evidence[0].score == pytest.approx(0.91)
    assert result.evidence[0].method == "hybrid_rerank"


def test_retrieval_tool_returns_execution_failure_for_missing_report() -> None:
    result = RetrievalTool(FakeRetriever())(
        RetrieveEvidenceInput(report_id="missing", query="revenue", top_k=5)
    )

    assert isinstance(result, ToolFailure)
    assert result.code == "execution_error"


def test_table_tool_matches_row_case_insensitively_with_stable_index() -> None:
    tool = ReportTableTool(
        {
            "report-a": (
                ("Metric", "2023", "2024"),
                ("Revenue", "100", "120"),
                ("Operating income", "20", "24"),
            )
        }
    )

    result = tool(
        QueryReportTableInput(report_id="report-a", row_selector="  REVENUE ")
    )

    assert not isinstance(result, ToolFailure)
    assert [(row.row_index, row.values) for row in result.rows] == [
        (1, ["Revenue", "100", "120"])
    ]


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("divide(39, 4)", Decimal("9.75")),
        ("greater(120, 100)", "yes"),
    ],
)
def test_financial_program_tool_executes_supported_programs(
    program: str,
    expected: Decimal | str,
) -> None:
    result = FinancialProgramTool({"report-a": (("Metric", "2024"),)})(
        ExecuteFinancialProgramInput(report_id="report-a", program=program)
    )

    assert result == ExecuteFinancialProgramOutput(
        value=str(expected),
        value_kind="decision" if isinstance(expected, str) else "decimal",
    )


@pytest.mark.parametrize(
    "program",
    [
        "divide(1, 0)",
        "add(#0, 1)",
        "__import__('os').system('whoami')",
    ],
)
def test_financial_program_tool_rejects_unsafe_or_invalid_programs(
    program: str,
) -> None:
    result = FinancialProgramTool({"report-a": (("Metric", "2024"),)})(
        ExecuteFinancialProgramInput(report_id="report-a", program=program)
    )

    assert isinstance(result, ToolFailure)
    assert result.code == "execution_error"
