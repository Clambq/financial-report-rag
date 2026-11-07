from pathlib import Path


ROOT = Path(__file__).parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_readme_documents_five_nodes_routes_and_reproduction() -> None:
    text = read("README.md")
    lowered = text.lower()

    for node in ("supervisor", "research", "quant", "verifier", "report"):
        assert node in lowered
    assert "/agent/ask" in text
    assert "/agent/stream" in text
    assert "evaluate-agent" in text
    assert "docker compose --profile vllm" in lowered
    assert "deterministic application test" in lowered
    assert "real-model" in lowered
    assert "97%" not in text


def test_architecture_documents_boundaries_retry_sse_and_vllm() -> None:
    text = read("docs/finsight-architecture.md").lower()

    assert "langgraph" in text
    assert "retry_count" in text
    assert "max_retries=1" in text
    assert "json_schema" in text
    assert "--enable-prefix-caching" in text
    assert "run_started" in text
    assert "run_finished" in text
    assert "human review" in text


def test_evaluation_documents_denominators_and_result_boundary() -> None:
    text = read("docs/finsight-evaluation.md").lower()

    for term in (
        "all tool attempts",
        "first attempts",
        "retry attempts",
        "citation denominator",
        "numerical denominator",
    ):
        assert term in text
    assert "63.64%" in text
    assert "fault-injection" in text
    assert "not a real-model tool-use accuracy" in text


def test_interview_guide_separates_implemented_claims_from_unmeasured_claims() -> None:
    text = read("docs/interview-guide.md")

    assert "FinSight Multi-Agent" in text
    assert "一次修复" in text
    assert "97%" in text
    assert "尚未复现" in text

