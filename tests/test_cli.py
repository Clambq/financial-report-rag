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
