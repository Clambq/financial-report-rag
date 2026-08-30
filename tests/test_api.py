from fastapi.testclient import TestClient

from finqa_rag.api.app import create_app
from finqa_rag.api.pipeline import RAGPipeline, route_live_question
from finqa_rag.api.schemas import AskRequest, AskResponse, Citation
from finqa_rag.data.models import EvidenceUnit
from finqa_rag.generation.qwen import GenerationResult
from finqa_rag.retrieval.types import ScoredEvidence


class FakePipeline:
    def answer(self, request: AskRequest) -> AskResponse:
        return AskResponse(
            question=request.question,
            answer="Revenue increased by 20% [table_1].",
            route="text",
            generated_program=None,
            citations=[
                Citation(
                    evidence_id="table_1",
                    evidence_type="table",
                    source_text="Metric: Revenue | 2023: 100 | 2024: 120",
                    retrieval_score=0.91,
                    rank=1,
                )
            ],
            needs_human_review=False,
            error=None,
            latency_ms=4.2,
        )

    def verified_metrics(self) -> dict[str, object]:
        return {"retrieval": {"recall_at_5": 0.8145597210113339}}


client = TestClient(create_app(FakePipeline()))


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_structured_citations() -> None:
    response = client.post(
        "/ask",
        json={"report_id": "report-a", "question": "What changed?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] in {"text", "numerical"}
    assert body["citations"][0]["evidence_id"] == "table_1"
    assert body["needs_human_review"] is False
    assert body["latency_ms"] >= 0


def test_ask_rejects_blank_question_and_report_id() -> None:
    blank_question = client.post(
        "/ask",
        json={"report_id": "report-a", "question": "   "},
    )
    blank_report = client.post(
        "/ask",
        json={"report_id": "   ", "question": "What changed?"},
    )

    assert blank_question.status_code == 422
    assert blank_report.status_code == 422


def test_metrics_returns_only_verified_artifacts() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["retrieval"]["recall_at_5"] == 0.8145597210113339


class FakeRetriever:
    def search(self, report_id: str, question: str) -> list[ScoredEvidence]:
        assert report_id == "report-a"
        evidence = EvidenceUnit(
            "table_1",
            report_id,
            "test",
            "table",
            "Metric: Revenue | 2023: 100 | 2024: 120",
        )
        return [ScoredEvidence(evidence, 0.91, 1, "hybrid_rerank")]


class FakeServiceGenerator:
    def __init__(self, program: str = "subtract(120, 100)") -> None:
        self.program = program

    def answer_text(self, question, evidence) -> GenerationResult:
        return GenerationResult("Demand increased [table_1].", "", 2.0)

    def generate_program(self, question, evidence) -> GenerationResult:
        return GenerationResult(self.program, self.program, 3.0)


def make_real_pipeline(program: str = "subtract(120, 100)") -> RAGPipeline:
    return RAGPipeline(
        report_tables={
            "report-a": (
                ("Metric", "2023", "2024"),
                ("Revenue", "100", "120"),
            )
        },
        retriever=FakeRetriever(),
        generator=FakeServiceGenerator(program),
        metrics={"retrieval": {"recall_at_5": 0.81}},
    )


def test_live_router_distinguishes_explanations_from_calculations() -> None:
    assert route_live_question("Why did revenue increase?") == "text"
    assert route_live_question("What was the percentage increase in revenue?") == "numerical"


def test_pipeline_executes_generated_program_and_returns_sources() -> None:
    response = make_real_pipeline().answer(
        AskRequest(
            report_id="report-a",
            question="What was the difference in revenue?",
        )
    )

    assert response.route == "numerical"
    assert response.answer == "20"
    assert response.generated_program == "subtract(120, 100)"
    assert response.citations[0].evidence_id == "table_1"
    assert response.needs_human_review is False


def test_pipeline_marks_unsafe_program_for_human_review() -> None:
    response = make_real_pipeline("__import__('os').system('whoami')").answer(
        AskRequest(
            report_id="report-a",
            question="Calculate the revenue difference.",
        )
    )

    assert response.needs_human_review is True
    assert response.generated_program == "__import__('os').system('whoami')"
    assert "unsupported operation" in response.error
    assert response.citations
