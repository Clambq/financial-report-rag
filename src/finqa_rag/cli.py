"""Command-line entry points for data preparation and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from finqa_rag.data.download import download_finqa
from finqa_rag.data.finqa import load_split
from finqa_rag.evaluation.retrieval_runner import run_retrieval_evaluation


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
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
