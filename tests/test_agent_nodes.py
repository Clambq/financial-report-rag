from __future__ import annotations

import asyncio
from typing import Any

import pytest

from finqa_rag.agents.model import ModelMessage
from finqa_rag.agents.nodes import (
    AgentRuntime,
    quant_node,
    report_node,
    research_node,
    supervisor_node,
)
from finqa_rag.agents.state import CalculationRecord, FinSightState
from finqa_rag.agents.tools import (
    ComposeGroundedReportInput,
    ComposeGroundedReportOutput,
    EvidenceRecord,
    ExecuteFinancialProgramInput,
    ExecuteFinancialProgramOutput,
    FinancialProgramTool,
    QueryReportTableInput,
    QueryReportTableOutput,
    RetrieveEvidenceOutput,
    TableRowRecord,
    ToolFailure,
    ToolRegistry,
)


class SchemaQueueModel:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self._outputs = iter(outputs)

    def generate(self, schema: type[Any], messages: list[ModelMessage]) -> Any:
        return schema.model_validate(next(self._outputs))

    def generate_text(self, messages: list[ModelMessage]) -> str:
        raise AssertionError("nodes must use structured model output")


def make_state(question: str = "Why did revenue increase?") -> FinSightState:
    return FinSightState(
        request_id="req-1",
        report_id="report-a",
        question=question,
    )


@pytest.mark.parametrize(
    ("route", "capabilities"),
    [
        ("text", ["research", "report"]),
        ("numerical", ["research", "quant", "report"]),
        ("mixed", ["research", "quant", "report"]),
    ],
)
def test_supervisor_returns_structured_plan(
    route: str,
    capabilities: list[str],
) -> None:
    runtime = AgentRuntime(
        model=SchemaQueueModel(
            [
                {
                    "route": route,
                    "capabilities": capabilities,
                    "tool_names": [],
                }
            ]
        ),
        registry=ToolRegistry(),
    )

    update = asyncio.run(supervisor_node(make_state(), runtime))

    assert update["plan"].route == route
    assert update["plan"].capabilities == capabilities
    assert update["issues"] == []


def test_supervisor_records_unknown_tool_without_invoking_registry() -> None:
    calls = 0

    def should_not_run(value: QueryReportTableInput) -> QueryReportTableOutput:
        nonlocal calls
        calls += 1
        return QueryReportTableOutput(rows=[])

    registry = ToolRegistry()
    registry.register(
        "query_report_table",
        QueryReportTableInput,
        QueryReportTableOutput,
        should_not_run,
    )
    runtime = AgentRuntime(
        model=SchemaQueueModel(
            [
                {
                    "route": "text",
                    "capabilities": ["research", "report"],
                    "tool_names": ["query_report_table", "delete_database"],
                }
            ]
        ),
        registry=registry,
    )

    update = asyncio.run(supervisor_node(make_state(), runtime))

    assert calls == 0
    assert [(issue.code, issue.message) for issue in update["issues"]] == [
        ("unknown_tool", "unknown tool requested by supervisor: delete_database")
    ]


def evidence(
    evidence_id: str,
    *,
    rank: int,
    method: str,
    score: float,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        report_id="report-a",
        kind="text",
        text=f"source {evidence_id}",
        score=score,
        rank=rank,
        method=method,
    )


def test_research_starts_lexical_and_dense_retrieval_concurrently() -> None:
    async def scenario() -> dict[str, object]:
        lexical_started = asyncio.Event()
        dense_started = asyncio.Event()

        async def lexical(
            report_id: str,
            query: str,
            top_k: int,
        ) -> RetrieveEvidenceOutput:
            lexical_started.set()
            await asyncio.wait_for(dense_started.wait(), timeout=0.2)
            return RetrieveEvidenceOutput(
                evidence=[evidence("text_1", rank=1, method="bm25", score=2.0)]
            )

        async def dense(
            report_id: str,
            query: str,
            top_k: int,
        ) -> RetrieveEvidenceOutput:
            dense_started.set()
            await asyncio.wait_for(lexical_started.wait(), timeout=0.2)
            return RetrieveEvidenceOutput(
                evidence=[
                    evidence("text_2", rank=1, method="dense", score=0.8),
                    evidence("text_1", rank=2, method="dense", score=0.7),
                ]
            )

        runtime = AgentRuntime(
            model=SchemaQueueModel([]),
            registry=ToolRegistry(),
            lexical_search=lexical,
            dense_search=dense,
        )
        return await research_node(make_state(), runtime)

    update = asyncio.run(scenario())

    assert [item.evidence_id for item in update["evidence"]] == [
        "text_1",
        "text_2",
    ]
    assert all(item.method == "parallel_rrf" for item in update["evidence"])
    assert update["issues"] == []


def test_research_degrades_to_successful_retriever() -> None:
    async def failed(
        report_id: str,
        query: str,
        top_k: int,
    ) -> ToolFailure:
        return ToolFailure(
            tool_name="bm25",
            code="execution_error",
            message="index unavailable",
        )

    async def succeeded(
        report_id: str,
        query: str,
        top_k: int,
    ) -> RetrieveEvidenceOutput:
        return RetrieveEvidenceOutput(
            evidence=[evidence("text_2", rank=1, method="dense", score=0.8)]
        )

    runtime = AgentRuntime(
        model=SchemaQueueModel([]),
        registry=ToolRegistry(),
        lexical_search=failed,
        dense_search=succeeded,
    )

    update = asyncio.run(research_node(make_state(), runtime))

    assert [item.evidence_id for item in update["evidence"]] == ["text_2"]
    assert update["issues"][0].code == "retriever_degraded"
    assert update["issues"][0].recoverable is True
    assert update["needs_human_review"] is False


def test_research_sends_both_failures_to_human_review() -> None:
    async def failed(
        report_id: str,
        query: str,
        top_k: int,
    ) -> ToolFailure:
        return ToolFailure(
            tool_name="retriever",
            code="execution_error",
            message="unavailable",
        )

    runtime = AgentRuntime(
        model=SchemaQueueModel([]),
        registry=ToolRegistry(),
        lexical_search=failed,
        dense_search=failed,
    )

    update = asyncio.run(research_node(make_state(), runtime))

    assert update["evidence"] == []
    assert update["needs_human_review"] is True
    assert update["human_review_reason"] == "all retrievers failed"


def test_quant_queries_table_before_executing_program() -> None:
    order: list[str] = []
    registry = ToolRegistry()

    def query_table(value: QueryReportTableInput) -> QueryReportTableOutput:
        order.append("query_report_table")
        return QueryReportTableOutput(
            rows=[TableRowRecord(row_index=1, values=["Revenue", "100", "120"])]
        )

    def execute(
        value: ExecuteFinancialProgramInput,
    ) -> ExecuteFinancialProgramOutput:
        assert order == ["query_report_table"]
        order.append("execute_financial_program")
        return ExecuteFinancialProgramOutput(value="0.2", value_kind="decimal")

    registry.register(
        "query_report_table",
        QueryReportTableInput,
        QueryReportTableOutput,
        query_table,
    )
    registry.register(
        "execute_financial_program",
        ExecuteFinancialProgramInput,
        ExecuteFinancialProgramOutput,
        execute,
    )
    runtime = AgentRuntime(
        model=SchemaQueueModel(
            [{"row_selector": "Revenue", "program": "divide(20, 100)"}]
        ),
        registry=registry,
    )

    update = asyncio.run(quant_node(make_state("What was revenue growth?"), runtime))

    assert order == ["query_report_table", "execute_financial_program"]
    assert update["calculations"] == [
        CalculationRecord(
            program="divide(20, 100)",
            value="0.2",
            value_kind="decimal",
            verified=False,
        )
    ]


def test_quant_records_execution_issue_for_invalid_program() -> None:
    registry = ToolRegistry()
    registry.register(
        "query_report_table",
        QueryReportTableInput,
        QueryReportTableOutput,
        lambda value: QueryReportTableOutput(rows=[]),
    )
    registry.register(
        "execute_financial_program",
        ExecuteFinancialProgramInput,
        ExecuteFinancialProgramOutput,
        FinancialProgramTool({"report-a": (("Metric", "2024"),)}),
    )
    runtime = AgentRuntime(
        model=SchemaQueueModel(
            [{"row_selector": "Revenue", "program": "divide(1, 0)"}]
        ),
        registry=registry,
    )

    update = asyncio.run(quant_node(make_state(), runtime))

    assert update["calculations"] == []
    assert update["issues"][0].code == "execution_error"


def test_report_uses_only_typed_grounded_report_tool_payload() -> None:
    captured: list[ComposeGroundedReportInput] = []
    registry = ToolRegistry()

    def compose(value: ComposeGroundedReportInput) -> ComposeGroundedReportOutput:
        captured.append(value)
        return ComposeGroundedReportOutput(
            answer="Revenue increased by 20% [text_1].",
            citation_ids=["text_1"],
            allowed_evidence_ids=value.allowed_evidence_ids,
        )

    registry.register(
        "compose_grounded_report",
        ComposeGroundedReportInput,
        ComposeGroundedReportOutput,
        compose,
    )
    runtime = AgentRuntime(model=SchemaQueueModel([]), registry=registry)
    state = make_state()
    state.evidence = [evidence("text_1", rank=1, method="dense", score=0.8)]
    state.calculations = [
        CalculationRecord(
            program="divide(20, 100)",
            value="0.2",
            value_kind="decimal",
            verified=False,
        )
    ]

    update = asyncio.run(report_node(state, runtime))

    assert len(captured) == 1
    assert [item.evidence_id for item in captured[0].evidence] == ["text_1"]
    assert captured[0].calculations == [state.calculations[0].model_dump()]
    assert captured[0].allowed_evidence_ids == ["text_1"]
    assert update["draft_answer"] == "Revenue increased by 20% [text_1]."
    assert update["citation_ids"] == ["text_1"]
