"""Construct the production FastAPI app from verified project components."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI

from finqa_rag.api.app import create_app
from finqa_rag.api.pipeline import RAGPipeline, ReportRetrievalIndex
from finqa_rag.data.finqa import load_split
from finqa_rag.generation.qwen import QwenGenerator


def _read_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_service_app(
    data_dir: Path = Path("data/raw/finqa"),
    split: str = "test",
    device: str = "cuda",
) -> FastAPI:
    examples = load_split(data_dir / f"{split}.json", split)
    retriever = ReportRetrievalIndex(examples, device=device)
    generator = QwenGenerator()
    metrics = {
        "retrieval": _read_metrics(Path("artifacts/retrieval/metrics_summary.json")),
        "numerical": _read_metrics(Path("artifacts/numerical/metrics_summary.json")),
    }
    pipeline = RAGPipeline.from_examples(
        examples,
        retriever=retriever,
        generator=generator,
        metrics=metrics,
    )
    return create_app(pipeline)
