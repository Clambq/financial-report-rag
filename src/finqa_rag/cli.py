"""Command-line entry points for data preparation and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from finqa_rag.data.download import download_finqa
from finqa_rag.data.finqa import load_split


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
