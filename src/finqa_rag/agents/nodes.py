"""FinSight workflow nodes with explicit state updates."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter_ns

from pydantic import BaseModel, ConfigDict, field_validator

from finqa_rag.agents.events import AgentEvent
from finqa_rag.agents.model import ModelMessage, ModelOutputError, StructuredModel
from finqa_rag.agents.state import (
    AgentIssue,
    AgentPlan,
    CalculationRecord,
    FinSightState,
)
from finqa_rag.agents.tools import (
    ComposeGroundedReportOutput,
    EvidenceRecord,
    ExecuteFinancialProgramOutput,
    QueryReportTableOutput,
    RetrieveEvidenceOutput,
    ToolFailure,
    ToolRegistry,
)


SearchCallable = Callable[
    [str, str, int],
    Awaitable[RetrieveEvidenceOutput | ToolFailure],
]
EventSink = Callable[[AgentEvent], Awaitable[None] | None]


@dataclass(slots=True)
class AgentRuntime:
    model: StructuredModel
    registry: ToolRegistry
    lexical_search: SearchCallable | None = None
    dense_search: SearchCallable | None = None
    clock: Callable[[], int] = perf_counter_ns
    event_sink: EventSink | None = None


class QuantDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_selector: str
    program: str

    @field_validator("row_selector", "program")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


async def supervisor_node(
    state: FinSightState,
    runtime: AgentRuntime,
) -> dict[str, object]:
    plan = runtime.model.generate(
        AgentPlan,
        [
            ModelMessage(
                role="system",
                content=(
                    "Classify the request as text, numerical, mixed, or unsupported. "
                    "Choose only research, quant, and report capabilities and list "
                    "the tools the workflow needs."
                ),
            ),
            ModelMessage(role="user", content=state.question),
        ],
    )
    issues = [
        AgentIssue(
            code="unknown_tool",
            message=f"unknown tool requested by supervisor: {tool_name}",
            node="supervisor",
            recoverable=False,
        )
        for tool_name in plan.tool_names
        if tool_name not in runtime.registry.names
    ]
    return {"plan": plan, "issues": issues}


def _fuse_parallel_rankings(
    rankings: list[RetrieveEvidenceOutput],
) -> list[EvidenceRecord]:
    scores: dict[str, float] = {}
    records: dict[str, EvidenceRecord] = {}
    for ranking in rankings:
        for item in ranking.evidence:
            records.setdefault(item.evidence_id, item)
            scores[item.evidence_id] = scores.get(item.evidence_id, 0.0) + (
                1.0 / (60 + item.rank)
            )
    ordered = sorted(scores, key=lambda item: (-scores[item], item))
    return [
        EvidenceRecord(
            evidence_id=records[evidence_id].evidence_id,
            report_id=records[evidence_id].report_id,
            kind=records[evidence_id].kind,
            text=records[evidence_id].text,
            score=scores[evidence_id],
            rank=rank,
            method="parallel_rrf",
        )
        for rank, evidence_id in enumerate(ordered, start=1)
    ]


def _expected_search_failure(error: BaseException) -> ToolFailure:
    if not isinstance(error, (TimeoutError, ConnectionError, KeyError, ValueError)):
        raise error
    return ToolFailure(
        tool_name="retrieve_evidence",
        code="execution_error",
        message=str(error),
        retryable=isinstance(error, (TimeoutError, ConnectionError)),
    )


async def research_node(
    state: FinSightState,
    runtime: AgentRuntime,
) -> dict[str, object]:
    if runtime.lexical_search is None or runtime.dense_search is None:
        result = await runtime.registry.ainvoke(
            "retrieve_evidence",
            {"report_id": state.report_id, "query": state.question, "top_k": 5},
        )
        if isinstance(result, ToolFailure):
            return {
                "evidence": [],
                "issues": [
                    AgentIssue(
                        code=result.code,
                        message=result.message,
                        node="research",
                        recoverable=result.retryable,
                    )
                ],
                "needs_human_review": True,
                "human_review_reason": "retrieval failed",
            }
        assert isinstance(result, RetrieveEvidenceOutput)
        return {
            "evidence": result.evidence,
            "issues": [],
            "needs_human_review": False,
        }

    raw_results = await asyncio.gather(
        runtime.lexical_search(state.report_id, state.question, 10),
        runtime.dense_search(state.report_id, state.question, 10),
        return_exceptions=True,
    )
    results = [
        _expected_search_failure(result) if isinstance(result, BaseException) else result
        for result in raw_results
    ]
    successes = [
        result for result in results if isinstance(result, RetrieveEvidenceOutput)
    ]
    failures = [result for result in results if isinstance(result, ToolFailure)]
    if not successes:
        return {
            "evidence": [],
            "issues": [
                AgentIssue(
                    code=failure.code,
                    message=failure.message,
                    node="research",
                    recoverable=failure.retryable,
                )
                for failure in failures
            ],
            "needs_human_review": True,
            "human_review_reason": "all retrievers failed",
        }
    if failures:
        return {
            "evidence": successes[0].evidence,
            "issues": [
                AgentIssue(
                    code="retriever_degraded",
                    message=failure.message,
                    node="research",
                    recoverable=True,
                )
                for failure in failures
            ],
            "needs_human_review": False,
        }
    return {
        "evidence": _fuse_parallel_rankings(successes),
        "issues": [],
        "needs_human_review": False,
    }


async def quant_node(
    state: FinSightState,
    runtime: AgentRuntime,
) -> dict[str, object]:
    try:
        decision = runtime.model.generate(
            QuantDecision,
            [
                ModelMessage(
                    role="system",
                    content=(
                        "Select one relevant table row and return one fixed-grammar "
                        "FinQA program. Never calculate the answer yourself."
                    ),
                ),
                ModelMessage(role="user", content=state.question),
            ],
        )
    except ModelOutputError as error:
        return {
            "calculations": [],
            "issues": [
                AgentIssue(
                    code="model_output_error",
                    message=str(error),
                    node="quant",
                    recoverable=True,
                )
            ],
        }

    table_result = await runtime.registry.ainvoke(
        "query_report_table",
        {"report_id": state.report_id, "row_selector": decision.row_selector},
    )
    if isinstance(table_result, ToolFailure):
        return {
            "calculations": [],
            "issues": [
                AgentIssue(
                    code=table_result.code,
                    message=table_result.message,
                    node="quant",
                    recoverable=table_result.retryable,
                )
            ],
        }
    assert isinstance(table_result, QueryReportTableOutput)

    execution = await runtime.registry.ainvoke(
        "execute_financial_program",
        {"report_id": state.report_id, "program": decision.program},
    )
    if isinstance(execution, ToolFailure):
        return {
            "calculations": [],
            "issues": [
                AgentIssue(
                    code=execution.code,
                    message=execution.message,
                    node="quant",
                    recoverable=execution.retryable,
                )
            ],
        }
    assert isinstance(execution, ExecuteFinancialProgramOutput)
    return {
        "calculations": [
            CalculationRecord(
                program=decision.program,
                value=execution.value,
                value_kind=execution.value_kind,
                verified=False,
            )
        ],
        "issues": [],
    }


async def report_node(
    state: FinSightState,
    runtime: AgentRuntime,
) -> dict[str, object]:
    allowed_ids = [item.evidence_id for item in state.evidence]
    result = await runtime.registry.ainvoke(
        "compose_grounded_report",
        {
            "question": state.question,
            "evidence": [item.model_dump() for item in state.evidence],
            "calculations": [item.model_dump() for item in state.calculations],
            "allowed_evidence_ids": allowed_ids,
        },
    )
    if isinstance(result, ToolFailure):
        return {
            "draft_answer": None,
            "citation_ids": [],
            "issues": [
                AgentIssue(
                    code=result.code,
                    message=result.message,
                    node="report",
                    recoverable=result.retryable,
                )
            ],
        }
    assert isinstance(result, ComposeGroundedReportOutput)
    return {
        "draft_answer": result.answer,
        "citation_ids": result.citation_ids,
        "issues": [],
    }
