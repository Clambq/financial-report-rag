import json
from pathlib import Path

from typer.testing import CliRunner

from finqa_rag.cli import app


FIXTURE = Path(__file__).parent / "fixtures" / "finqa_mini.json"


def test_prepare_data_downloads_and_validates_all_public_splits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = FIXTURE.read_bytes()

    def fake_download(output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True)
        for split in ("train", "dev", "test"):
            (output_dir / f"{split}.json").write_bytes(payload)
        manifest = {"revision": "fixture-revision", "files": {}}
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    monkeypatch.setattr("finqa_rag.cli.download_finqa", fake_download)
    output_dir = tmp_path / "finqa"

    result = CliRunner().invoke(app, ["prepare-data", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert "fixture-revision" in result.stdout
    assert all((output_dir / f"{split}.json").exists() for split in ("train", "dev", "test"))


def test_evaluate_retrieval_writes_bm25_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "finqa"
    data_dir.mkdir()
    (data_dir / "test.json").write_bytes(FIXTURE.read_bytes())
    output_dir = tmp_path / "artifacts"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-retrieval",
            "--data-dir",
            str(data_dir),
            "--split",
            "test",
            "--methods",
            "bm25",
            "--top-k",
            "5",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary["query_count"] == 2
    assert summary["methods"]["bm25"]["recall_at_5"] == 1.0


def test_evaluate_retrieval_forwards_hybrid_tuning_options(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "finqa"
    data_dir.mkdir()
    (data_dir / "dev.json").write_bytes(FIXTURE.read_bytes())
    captured: dict[str, object] = {}

    def fake_evaluation(examples, **kwargs):
        captured.update(kwargs)
        return {"query_count": len(examples), "methods": {}}

    monkeypatch.setattr("finqa_rag.cli.run_retrieval_evaluation", fake_evaluation)

    result = CliRunner().invoke(
        app,
        [
            "evaluate-retrieval",
            "--data-dir",
            str(data_dir),
            "--split",
            "dev",
            "--methods",
            "dense,hybrid,hybrid_rerank",
            "--candidate-k",
            "10",
            "--bm25-weight",
            "2.0",
            "--dense-weight",
            "0.5",
            "--rrf-k",
            "30",
            "--dense-model",
            "dense-fixture",
            "--reranker-model",
            "reranker-fixture",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "artifacts"),
        ],
    )

    assert result.exit_code == 0
    assert captured["candidate_k"] == 10
    assert captured["bm25_weight"] == 2.0
    assert captured["dense_weight"] == 0.5
    assert captured["rrf_k"] == 30
    assert captured["dense_model_name"] == "dense-fixture"
    assert captured["reranker_model_name"] == "reranker-fixture"
    assert captured["device"] == "cpu"


def test_evaluate_numerical_uses_locked_limit_and_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "finqa"
    data_dir.mkdir()
    (data_dir / "test.json").write_bytes(FIXTURE.read_bytes())
    captured: dict[str, object] = {}

    class FakeQwen:
        def __init__(self, model_id: str, max_new_tokens: int, do_sample: bool) -> None:
            captured.update(
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

    def fake_evaluation(examples, generator, output_dir, **kwargs):
        captured.update(kwargs)
        captured["output_dir"] = output_dir
        return {"query_count": kwargs["sample_size"]}

    monkeypatch.setattr("finqa_rag.cli.QwenGenerator", FakeQwen)
    monkeypatch.setattr("finqa_rag.cli.run_numerical_evaluation", fake_evaluation)
    output = tmp_path / "numerical"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-numerical",
            "--data-dir",
            str(data_dir),
            "--limit",
            "1",
            "--seed",
            "7",
            "--model-id",
            "qwen-fixture",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert captured["sample_size"] == 1
    assert captured["seed"] == 7
    assert captured["model_id"] == "qwen-fixture"
    assert captured["max_new_tokens"] == 64
    assert captured["do_sample"] is False
    assert captured["output_dir"] == output
