"""Deterministic FinSight reliability runner and artifact verifier."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

from finqa_rag.agents.tools import ToolCallRecord
from finqa_rag.common.atomic import atomic_write_json, atomic_write_jsonl
from finqa_rag.evaluation.agent_metrics import AgentPrediction, compute_agent_metrics


Scenario = Literal[
    "text",
    "numerical",
    "mixed",
    "malformed_arguments",
    "unknown_tool",
    "missing_evidence",
    "unsupported_operation",
    "retriever_degradation",
    "verification_retry",
    "retry_exhaustion",
]


class AgentCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    scenario: Scenario
    report_id: str
    question: str


def load_agent_cases(path: Path) -> list[AgentCase]:
    cases = TypeAdapter(list[AgentCase]).validate_json(path.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("agent case file must not be empty")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("agent case IDs must be unique")
    return cases


def _call(
    name: str,
    success: bool,
    *,
    attempt: int = 1,
    error_code: str | None = None,
) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=name,
        success=success,
        latency_ms=1.0,
        attempt=attempt,
        error_code=error_code,
    )


def _fake_prediction(case: AgentCase, index: int) -> AgentPrediction:
    base = {
        "case_id": case.case_id,
        "latency_ms": float((index + 1) * 10),
        "model_call_count": 2,
    }
    if case.scenario == "text":
        return AgentPrediction(
            **base,
            completed=True,
            tool_calls=[_call("retrieve", True), _call("verify", True)],
            citation_valid=True,
        )
    if case.scenario in {"numerical", "mixed"}:
        return AgentPrediction(
            **base,
            completed=True,
            tool_calls=[
                _call("retrieve", True),
                _call("query_table", True),
                _call("execute", True),
                _call("verify", True),
            ],
            citation_valid=True,
            numerical_verified=True,
        )
    if case.scenario == "malformed_arguments":
        return AgentPrediction(
            **base,
            completed=False,
            tool_calls=[_call("retrieve", False, error_code="validation_error")],
            error_category="malformed_arguments",
        )
    if case.scenario == "unknown_tool":
        return AgentPrediction(
            **base,
            completed=False,
            tool_calls=[_call("unknown", False, error_code="unknown_tool")],
            error_category="unknown_tool",
        )
    if case.scenario == "missing_evidence":
        return AgentPrediction(
            **base,
            completed=False,
            tool_calls=[_call("retrieve", True), _call("verify", False, error_code="validation_error")],
            citation_valid=False,
            error_category="missing_evidence",
        )
    if case.scenario == "unsupported_operation":
        return AgentPrediction(
            **base,
            completed=False,
            tool_calls=[_call("execute", False, error_code="execution_error")],
            numerical_verified=False,
            error_category="unsupported_operation",
        )
    if case.scenario == "retriever_degradation":
        return AgentPrediction(
            **base,
            completed=True,
            tool_calls=[
                _call("bm25", True),
                _call("dense", False, error_code="timeout"),
                _call("verify", True),
            ],
            citation_valid=True,
            error_category="retriever_degradation",
        )
    if case.scenario == "verification_retry":
        return AgentPrediction(
            **base,
            completed=True,
            tool_calls=[
                _call("verify", False, error_code="validation_error"),
                _call("verify", True, attempt=2),
            ],
            citation_valid=True,
        )
    return AgentPrediction(
        **base,
        completed=False,
        tool_calls=[
            _call("verify", False, error_code="validation_error"),
            _call("verify", False, attempt=2, error_code="validation_error"),
        ],
        citation_valid=False,
        error_category="retry_exhaustion",
    )


def _canonical_payload(predictions: list[AgentPrediction]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        row = prediction.model_dump(mode="json")
        row.pop("latency_ms", None)
        for call in row["tool_calls"]:
            call.pop("latency_ms", None)
        rows.append(row)
    return rows


def _canonical_hash(predictions: list[AgentPrediction]) -> str:
    payload = json.dumps(
        _canonical_payload(predictions),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_agent_evaluation(
    cases: list[AgentCase],
    output_dir: Path,
    *,
    model: str = "fake",
) -> dict[str, object]:
    if model != "fake":
        raise ValueError("only the deterministic fake evaluator is available in this command")
    predictions = [_fake_prediction(case, index) for index, case in enumerate(cases)]
    summary = compute_agent_metrics(predictions)
    manifest: dict[str, object] = {
        "label": "deterministic_application_test",
        "model": model,
        "query_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "python_version": platform.python_version(),
    }
    reproducibility: dict[str, object] = {
        "canonical_predictions_sha256": _canonical_hash(predictions),
        "excluded_nondeterministic_fields": [
            "prediction.latency_ms",
            "tool_calls[].latency_ms",
        ],
    }
    atomic_write_json(output_dir / "metrics_summary.json", summary)
    atomic_write_jsonl(
        output_dir / "predictions.jsonl",
        [item.model_dump(mode="json") for item in predictions],
    )
    atomic_write_json(output_dir / "experiment_manifest.json", manifest)
    atomic_write_json(output_dir / "reproducibility.json", reproducibility)
    return summary


def verify_agent_artifacts(output_dir: Path) -> dict[str, object]:
    summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    reproducibility = json.loads((output_dir / "reproducibility.json").read_text(encoding="utf-8"))
    predictions = [
        AgentPrediction.model_validate_json(line)
        for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [item.case_id for item in predictions] != manifest["case_ids"]:
        raise ValueError("agent prediction case IDs do not match manifest")
    recomputed = compute_agent_metrics(predictions)
    if recomputed != summary:
        raise ValueError("agent metrics summary does not match prediction rows")
    canonical = _canonical_hash(predictions)
    if canonical != reproducibility["canonical_predictions_sha256"]:
        raise ValueError("agent prediction reproducibility hash mismatch")
    return {
        "query_count": len(predictions),
        "canonical_predictions_sha256": canonical,
        "metrics_recomputed": True,
    }

