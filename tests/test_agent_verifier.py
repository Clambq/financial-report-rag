from __future__ import annotations

import asyncio

from finqa_rag.agents.nodes import AgentRuntime, verifier_node
from finqa_rag.agents.state import CalculationRecord, FinSightState
from finqa_rag.agents.tools import (
    EvidenceRecord,
    GroundingVerifierTool,
    ToolRegistry,
    VerifyGroundingInput,
    VerifyGroundingOutput,
)


class UnusedModel:
    def generate(self, schema, messages):  # type: ignore[no-untyped-def]
        raise AssertionError("verifier must not call the model")

    def generate_text(self, messages):  # type: ignore[no-untyped-def]
        raise AssertionError("verifier must not call the model")


def evidence(evidence_id: str, report_id: str, text: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        report_id=report_id,
        kind="text",
        text=text,
        score=1.0,
        rank=1,
        method="test",
    )


def state(**updates: object) -> FinSightState:
    values: dict[str, object] = {
        "request_id": "req-1",
        "report_id": "report-a",
        "question": "How did revenue change?",
        "evidence": [evidence("ev-1", "report-a", "Revenue rose from 10 to 20.")],
        "draft_answer": "Revenue rose from 10 to 20 [ev-1].",
        "citation_ids": ["ev-1"],
    }
    values.update(updates)
    return FinSightState.model_validate(values)


def runtime_with_verifier(handler) -> AgentRuntime:  # type: ignore[no-untyped-def]
    registry = ToolRegistry()
    registry.register(
        "verify_grounding",
        VerifyGroundingInput,
        VerifyGroundingOutput,
        handler,
    )
    return AgentRuntime(model=UnusedModel(), registry=registry)


def test_verifier_uses_typed_contract_and_accepts_current_request_citations() -> None:
    calls: list[VerifyGroundingInput] = []

    def verify(request: VerifyGroundingInput) -> VerifyGroundingOutput:
        calls.append(request)
        return VerifyGroundingOutput(passed=True)

    update = asyncio.run(verifier_node(state(), runtime_with_verifier(verify)))

    assert len(calls) == 1
    assert calls[0].allowed_evidence_ids == ["ev-1"]
    assert [item.evidence_id for item in calls[0].evidence] == ["ev-1"]
    assert update["verification_passed"] is True
    assert update["final_answer"] == "Revenue rose from 10 to 20 [ev-1]."
    assert update["issues"] == []


def test_verifier_rejects_invented_and_cross_report_citations_before_handler() -> None:
    calls = 0

    def verify(request: VerifyGroundingInput) -> VerifyGroundingOutput:
        nonlocal calls
        calls += 1
        return VerifyGroundingOutput(passed=True)

    runtime = runtime_with_verifier(verify)
    invented = asyncio.run(
        verifier_node(state(citation_ids=["invented"]), runtime)
    )
    cross_report = asyncio.run(
        verifier_node(
            state(
                evidence=[evidence("ev-other", "report-b", "Revenue was 20.")],
                citation_ids=["ev-other"],
                draft_answer="Revenue was 20 [ev-other].",
            ),
            runtime,
        )
    )

    assert calls == 0
    assert invented["verification_passed"] is False
    assert cross_report["verification_passed"] is False
    assert invented["issues"][0].code == "validation_error"
    assert cross_report["issues"][0].code == "validation_error"
    assert invented["repair_target"] == "research"


def test_grounding_verifier_matches_numbers_against_evidence_and_quant_result() -> None:
    tool = GroundingVerifierTool()
    valid = VerifyGroundingInput(
        draft_answer="Revenue rose from 10 to 20 and the ratio is 2 [ev-1].",
        citation_ids=["ev-1"],
        allowed_evidence_ids=["ev-1"],
        evidence=[evidence("ev-1", "report-a", "Revenue rose from 10 to 20.")],
        calculations=[
            CalculationRecord(
                program="divide(20, 10)",
                value="2",
                value_kind="decimal",
            ).model_dump()
        ],
    )
    invalid = valid.model_copy(
        update={"draft_answer": "Revenue rose from 10 to 25 [ev-1]."}
    )

    assert tool(valid).passed is True
    result = tool(invalid)
    assert result.passed is False
    assert result.issues[0].code == "numerical_mismatch"


def test_grounding_verifier_marks_unsupported_comparison_unverifiable() -> None:
    tool = GroundingVerifierTool()
    request = VerifyGroundingInput(
        draft_answer="Revenue was more than operating income [ev-1].",
        citation_ids=["ev-1"],
        allowed_evidence_ids=["ev-1"],
        evidence=[evidence("ev-1", "report-a", "Revenue was 20; operating income was 10.")],
        calculations=[],
    )

    result = tool(request)

    assert result.passed is False
    assert result.issues[0].code == "unverifiable_comparison"

