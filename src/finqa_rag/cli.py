"""Command-line entry points for data preparation and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from finqa_rag.api.bootstrap import build_finsight_app, build_service_app
from finqa_rag.data.download import download_finqa
from finqa_rag.data.finqa import load_split
from finqa_rag.evaluation.numerical_runner import run_numerical_evaluation
from finqa_rag.evaluation.agent_runner import load_agent_cases, run_agent_evaluation
from finqa_rag.evaluation.report import (
    analyze_error_artifacts,
    verify_artifacts,
    write_reports,
)
from finqa_rag.evaluation.retrieval_runner import run_retrieval_evaluation
from finqa_rag.generation.qwen import QwenGenerator


app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Run reproducible FinQA data, evaluation, and serving workflows."""


@app.command("prepare-data")
def prepare_data(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for the pinned public FinQA splits."),
    ] = Path("data/raw/finqa"),
) -> None:
    """Download and validate train, development, and public test data."""

    manifest = download_finqa(output_dir)
    for split in ("train", "dev", "test"):
        load_split(output_dir / f"{split}.json", split)
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))


@app.command("evaluate-retrieval")
def evaluate_retrieval(
    data_dir: Annotated[
        Path,
        typer.Option(help="Directory containing the pinned FinQA JSON files."),
    ] = Path("data/raw/finqa"),
    split: Annotated[
        str,
        typer.Option(help="Public FinQA split to evaluate."),
    ] = "dev",
    methods: Annotated[
        str,
        typer.Option(help="Comma-separated retrieval method names."),
    ] = "bm25",
    top_k: Annotated[
        int,
        typer.Option(min=1, help="Number of evidence units returned per query."),
    ] = 5,
    candidate_k: Annotated[
        int,
        typer.Option(min=1, help="Candidate pool size before optional reranking."),
    ] = 20,
    bm25_weight: Annotated[
        float,
        typer.Option(min=0.0, help="BM25 weight used by reciprocal-rank fusion."),
    ] = 1.0,
    dense_weight: Annotated[
        float,
        typer.Option(min=0.0, help="Dense weight used by reciprocal-rank fusion."),
    ] = 1.0,
    rrf_k: Annotated[
        int,
        typer.Option(min=1, help="Reciprocal-rank-fusion smoothing constant."),
    ] = 60,
    dense_model: Annotated[
        str,
        typer.Option(help="Sentence-transformer model used for dense retrieval."),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    reranker_model: Annotated[
        str,
        typer.Option(help="Cross-encoder model used to rerank fused candidates."),
    ] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: Annotated[
        str,
        typer.Option(help="Model device, for example cpu or cuda."),
    ] = "cpu",
    output: Annotated[
        Path,
        typer.Option(help="Directory for summary, predictions, and manifest."),
    ] = Path("artifacts/retrieval/dev-bm25"),
) -> None:
    """Evaluate report-local evidence retrieval on one FinQA split."""

    if split not in {"train", "dev", "test"}:
        raise typer.BadParameter("split must be train, dev, or test", param_hint="split")
    method_names = tuple(item.strip() for item in methods.split(",") if item.strip())
    examples = load_split(data_dir / f"{split}.json", split)
    summary = run_retrieval_evaluation(
        examples,
        methods=method_names,
        output_dir=output,
        top_k=top_k,
        candidate_k=candidate_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        rrf_k=rrf_k,
        dense_model_name=dense_model,
        reranker_model_name=reranker_model,
        device=device,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("evaluate-numerical")
def evaluate_numerical(
    data_dir: Annotated[
        Path,
        typer.Option(help="Directory containing the pinned FinQA JSON files."),
    ] = Path("data/raw/finqa"),
    split: Annotated[
        str,
        typer.Option(help="Public FinQA split to evaluate."),
    ] = "test",
    limit: Annotated[
        int,
        typer.Option(min=1, help="Number of locked numerical examples."),
    ] = 100,
    seed: Annotated[
        int,
        typer.Option(help="Seed embedded in deterministic ID hashing."),
    ] = 42,
    model_id: Annotated[
        str,
        typer.Option(help="Hugging Face causal language model identifier."),
    ] = "Qwen/Qwen3-4B-Instruct-2507",
    max_new_tokens: Annotated[
        int,
        typer.Option(min=1, help="Maximum generated tokens per direct answer."),
    ] = 64,
    output: Annotated[
        Path,
        typer.Option(help="Directory for summary, predictions, and manifest."),
    ] = Path("artifacts/numerical/test-run-1"),
) -> None:
    """Compare deterministic direct-model arithmetic with safe Python execution."""

    if split not in {"train", "dev", "test"}:
        raise typer.BadParameter("split must be train, dev, or test", param_hint="split")
    examples = load_split(data_dir / f"{split}.json", split)
    generator = QwenGenerator(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    summary = run_numerical_evaluation(
        examples,
        generator,
        output,
        sample_size=limit,
        seed=seed,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("evaluate-agent")
def evaluate_agent(
    cases: Annotated[
        Path,
        typer.Option(help="JSON file containing deterministic agent evaluation cases."),
    ] = Path("tests/fixtures/agent_eval_cases.json"),
    output: Annotated[
        Path,
        typer.Option(help="Directory for agent metrics, predictions, and manifests."),
    ] = Path("artifacts/agent/smoke"),
    model: Annotated[
        str,
        typer.Option(help="Evaluation runtime; use fake for deterministic smoke tests."),
    ] = "fake",
) -> None:
    """Run the reproducible FinSight application reliability suite."""

    summary = run_agent_evaluation(load_agent_cases(cases), output, model=model)
    typer.echo(
        json.dumps(
            {**summary, "output_dir": str(output)},
            indent=2,
            sort_keys=True,
        )
    )


@app.command("serve")
def serve(
    data_dir: Annotated[
        Path,
        typer.Option(help="Directory containing the pinned FinQA JSON files."),
    ] = Path("data/raw/finqa"),
    split: Annotated[
        str,
        typer.Option(help="Public FinQA split used as the report corpus."),
    ] = "test",
    device: Annotated[
        str,
        typer.Option(help="Retrieval model device, for example cuda or cpu."),
    ] = "cuda",
    host: Annotated[
        str,
        typer.Option(help="Address for the local FastAPI server."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(min=1, max=65535, help="Port for the local FastAPI server."),
    ] = 8000,
) -> None:
    """Start the evaluated RAG pipeline and interactive OpenAPI documentation."""

    if split not in {"train", "dev", "test"}:
        raise typer.BadParameter("split must be train, dev, or test", param_hint="split")
    service_app = build_service_app(data_dir, split, device)
    uvicorn.run(service_app, host=host, port=port)


@app.command("serve-agent")
def serve_agent(
    host: Annotated[
        str,
        typer.Option(help="Address for the FinSight API server."),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(min=1, max=65535, help="Port for the FinSight API server."),
    ] = 8000,
) -> None:
    """Start the environment-configured vLLM-backed FinSight service."""

    uvicorn.run(build_finsight_app(), host=host, port=port)


@app.command("generate-reports")
def generate_reports(
    artifact_root: Annotated[
        Path,
        typer.Option(help="Root containing verified retrieval and numerical artifacts."),
    ] = Path("artifacts"),
) -> None:
    """Generate Markdown results and resume bullets from verified JSON only."""

    verified = verify_artifacts(artifact_root)
    retrieval = json.loads(
        (artifact_root / "retrieval" / "metrics_summary.json").read_text(
            encoding="utf-8"
        )
    )
    numerical = json.loads(
        (artifact_root / "numerical" / "metrics_summary.json").read_text(
            encoding="utf-8"
        )
    )
    paths = write_reports(
        retrieval,
        numerical,
        artifact_root,
        error_analysis=analyze_error_artifacts(artifact_root),
    )
    typer.echo(
        json.dumps(
            {
                **verified,
                "results": str(paths.results),
                "resume_bullets": str(paths.resume_bullets),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("verify-artifacts")
def verify_artifacts_command(
    artifact_root: Annotated[
        Path,
        typer.Option(help="Root containing verified retrieval and numerical artifacts."),
    ] = Path("artifacts"),
) -> None:
    """Recompute published metrics and deterministic-run hashes."""

    typer.echo(json.dumps(verify_artifacts(artifact_root), indent=2, sort_keys=True))
