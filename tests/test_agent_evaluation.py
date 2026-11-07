from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from finqa_rag.agents.tools import ToolCallRecord
from finqa_rag.cli import app
from finqa_rag.evaluation.agent_metrics import (
    AgentPrediction,
    compute_agent_metrics,
)
from finqa_rag.evaluation.agent_runner import (
    load_agent_cases,
    run_agent_evaluation,
    verify_agent_artifacts,
)


FIXTURE = Path(__file__).parent / "fixtures" / "agent_eval_cases.json"


def prediction_one() -> AgentPrediction:
    return AgentPrediction(
        case_id="case-1",
        completed=True,
        tool_calls=[
            ToolCallRecord(tool_name="retrieve", success=True, latency_ms=1, attempt=1),
            ToolCallRecord(
                tool_name="execute",
                success=False,
                latency_ms=2,
                attempt=1,
                error_code="execution_error",
            ),
            ToolCallRecord(tool_name="execute", success=True, latency_ms=1, attempt=2),
        ],
        citation_valid=True,
        numerical_verified=True,
        latency_ms=10,
        model_call_count=2,
    )


def prediction_two() -> AgentPrediction:
    return AgentPrediction(
        case_id="case-2",
        completed=False,
        tool_calls=[
            ToolCallRecord(
                tool_name="unknown",
                success=False,
                latency_ms=1,
                attempt=1,
                error_code="unknown_tool",
            )
        ],
        citation_valid=False,
        numerical_verified=None,
        error_category="unknown_tool",
        latency_ms=30,
        model_call_count=1,
    )


def test_metrics_keep_explicit_denominators_and_interpolated_latency() -> None:
    metrics = compute_agent_metrics([prediction_one(), prediction_two()])

    assert metrics["query_count"] == 2
    assert metrics["task_completion"] == {"count": 1, "denominator": 2, "rate": 0.5}
    assert metrics["tool_success"] == {"count": 2, "denominator": 4, "rate": 0.5}
    assert metrics["first_attempt_success"] == {
        "count": 1,
        "denominator": 3,
        "rate": 1 / 3,
    }
    assert metrics["after_retry_success"] == {
        "count": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert metrics["citation_validity"] == {
        "count": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert metrics["numerical_verification"] == {
        "count": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert metrics["error_category_counts"] == {"unknown_tool": 1}
    assert metrics["model_call_count"] == 3
    assert metrics["tool_call_count"] == 4
    assert metrics["latency"] == {"p50_ms": 20.0, "p95_ms": 29.0}


def test_fixture_covers_required_agent_scenarios() -> None:
    cases = load_agent_cases(FIXTURE)

    assert {case.scenario for case in cases} == {
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
    }
    assert len({case.case_id for case in cases}) == len(cases)


def test_runner_writes_atomic_recomputable_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "agent"
    summary = run_agent_evaluation(
        load_agent_cases(FIXTURE),
        output,
        model="fake",
    )

    assert summary["query_count"] == 10
    assert (output / "metrics_summary.json").exists()
    assert (output / "predictions.jsonl").exists()
    assert (output / "experiment_manifest.json").exists()
    assert (output / "reproducibility.json").exists()
    assert not list(output.glob("*.partial"))
    rows = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["case_id"] for row in rows] == [
        case.case_id for case in load_agent_cases(FIXTURE)
    ]
    verified = verify_agent_artifacts(output)
    assert verified["query_count"] == 10
    assert verified["metrics_recomputed"] is True


def test_cli_prints_output_directory_and_query_count(tmp_path: Path) -> None:
    output = tmp_path / "smoke"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-agent",
            "--cases",
            str(FIXTURE),
            "--output",
            str(output),
            "--model",
            "fake",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["query_count"] == 10
    assert payload["output_dir"] == str(output)

