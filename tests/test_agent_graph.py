from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from finqa_rag.agents.graph import FinSightAgent, build_finsight_graph
from finqa_rag.agents.model import ModelMessage
from finqa_rag.agents.nodes import AgentRuntime
from finqa_rag.agents.state import FinSightState
from finqa_rag.agents.tools import (
    ComposeGroundedReportInput,
    ComposeGroundedReportOutput,
    EvidenceRecord,
    ExecuteFinancialProgramInput,
    ExecuteFinancialProgramOutput,
    QueryReportTableInput,
    QueryReportTableOutput,
    RetrieveEvidenceInput,
    RetrieveEvidenceOutput,
    TableRowRecord,
    ToolRegistry,
    VerificationIssue,
    VerifyGroundingInput,
    VerifyGroundingOutput,
)


class QueueModel:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs: Iterator[dict[str, object]] = iter(outputs)

    def generate(
        self,
        schema: type[Any],
        messages: list[ModelMessage],
    ) -> Any:
        return schema.model_validate(next(self.outputs))

    def generate_text(self, messages: list[ModelMessage]) -> str:
        raise AssertionError("graph nodes must use structured output")


def registry(*, verifier_results: list[VerifyGroundingOutput] | None = None) -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(
        "retrieve_evidence",
        RetrieveEvidenceInput,
        RetrieveEvidenceOutput,
        lambda request: RetrieveEvidenceOutput(
            evidence=[
                EvidenceRecord(
                    evidence_id="ev-1",
                    report_id=request.report_id,
                    kind="text",
                    text="Revenue rose from 10 to 20.",
                    score=1.0,
                    rank=1,
                    method="test",
                )
            ]
        ),
    )
    tools.register(
        "query_report_table",
        QueryReportTableInput,
        QueryReportTableOutput,
        lambda request: QueryReportTableOutput(
            rows=[TableRowRecord(row_index=0, values=["Revenue", "10", "20"])]
        ),
    )
    tools.register(
        "execute_financial_program",
        ExecuteFinancialProgramInput,
        ExecuteFinancialProgramOutput,
        lambda request: ExecuteFinancialProgramOutput(
            value="2", value_kind="decimal"
        ),
    )
    tools.register(
        "compose_grounded_report",
        ComposeGroundedReportInput,
        ComposeGroundedReportOutput,
        lambda request: ComposeGroundedReportOutput(
            answer="Revenue rose from 10 to 20 [ev-1].",
            citation_ids=["ev-1"],
            allowed_evidence_ids=request.allowed_evidence_ids,
        ),
    )
    results = iter(verifier_results or [VerifyGroundingOutput(passed=True)])
    tools.register(
        "verify_grounding",
        VerifyGroundingInput,
        VerifyGroundingOutput,
        lambda request: next(results),
    )
    return tools


def request(request_id: str = "req-1") -> FinSightState:
    return FinSightState(
        request_id=request_id,
        report_id="report-a",
        question="How did revenue change?",
    )


@pytest.mark.parametrize(
    ("route", "capabilities", "model_outputs", "expected_nodes"),
    [
        (
            "text",
            ["research", "report"],
            [],
            ["supervisor", "research", "report", "verifier"],
        ),
        (
            "numerical",
            ["research", "quant", "report"],
            [{"row_selector": "Revenue", "program": "divide(20, 10)"}],
            ["supervisor", "research", "quant", "report", "verifier"],
        ),
    ],
)
def test_graph_happy_paths_are_ordered_and_return_final_answer(
    route: str,
    capabilities: list[str],
    model_outputs: list[dict[str, object]],
    expected_nodes: list[str],
) -> None:
    model = QueueModel(
        [
            {"route": route, "capabilities": capabilities, "tool_names": []},
            *model_outputs,
        ]
    )
    agent = FinSightAgent(AgentRuntime(model=model, registry=registry()))

    result = asyncio.run(agent.ainvoke(request()))

    assert result.final_answer == "Revenue rose from 10 to 20 [ev-1]."
    assert [trace.node for trace in result.traces] == expected_nodes
    assert result.needs_human_review is False


def test_verification_repairs_once_then_terminates_for_human_review() -> None:
    failure = VerifyGroundingOutput(
        passed=False,
        issues=[
            VerificationIssue(
                code="citation_not_grounded",
                message="citation does not support the claim",
                recoverable=True,
            )
        ],
    )
    model = QueueModel(
        [
            {
                "route": "text",
                "capabilities": ["research", "report"],
                "tool_names": [],
            }
        ]
    )
    agent = FinSightAgent(
        AgentRuntime(
            model=model,
            registry=registry(verifier_results=[failure, failure]),
        ),
        max_retries=1,
    )

    result = asyncio.run(agent.ainvoke(request()))

    assert result.final_answer is None
    assert result.needs_human_review is True
    assert result.human_review_reason == "verification failed after 1 repair"
    assert [trace.node for trace in result.traces].count("verifier") == 2
    assert len(result.traces) == 7


def test_build_graph_returns_a_compiled_langgraph() -> None:
    model = QueueModel([])

    graph = build_finsight_graph(AgentRuntime(model=model, registry=registry()))

    assert graph.get_graph().nodes


def test_unexpected_programmer_error_surfaces() -> None:
    class BrokenModel:
        def generate(self, schema, messages):  # type: ignore[no-untyped-def]
            raise RuntimeError("programmer bug")

        def generate_text(self, messages):  # type: ignore[no-untyped-def]
            raise AssertionError

    agent = FinSightAgent(
        AgentRuntime(model=BrokenModel(), registry=registry()),
    )

    with pytest.raises(RuntimeError, match="programmer bug"):
        asyncio.run(agent.ainvoke(request()))

