import hashlib
import json
from pathlib import Path

import pytest

from finqa_rag.data.download import FINQA_REVISION, download_finqa
from finqa_rag.data.finqa import load_split


FIXTURE = Path(__file__).parent / "fixtures" / "finqa_mini.json"


def test_load_split_builds_stable_evidence_and_gold_ids() -> None:
    examples = load_split(FIXTURE, "test")

    first = examples[0]
    assert first.example_id == "report-a-0"
    assert first.report_id == "report-a"
    assert first.question == "What was the revenue growth rate?"
    assert [item.evidence_id for item in first.evidence] == [
        "text_0",
        "text_1",
        "table_0",
        "table_1",
        "table_2",
    ]
    assert first.evidence[3].text == "Metric: Revenue | 2023: 100 | 2024: 120"
    assert first.gold_evidence_ids == ("table_1",)
    assert first.table == (
        ("Metric", "2023", "2024"),
        ("Revenue", "100", "120"),
        ("Operating income", "20", "30"),
    )


def test_load_split_rejects_unmatched_gold_evidence(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload[0]["qa"]["gold_inds"] = {"text_99": "not present in report"}
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unmatched gold evidence.*text_99"):
        load_split(path, "test")


def test_load_split_preserves_table_zero_for_headerless_reports(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))[:1]
    payload[0]["table"] = [
        ["balance at december 31 2010", "$ 3711"],
        ["increase based on current period tax positions", "412"],
    ]
    payload[0]["qa"]["gold_inds"] = {
        "table_0": "balance at december 31 2010 the balance was $ 3711"
    }
    path = tmp_path / "headerless.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    example = load_split(path, "test")[0]

    assert example.gold_evidence_ids == ("table_0",)
    assert example.evidence[2].evidence_id == "table_0"
    assert "$ 3711" in example.evidence[2].text


def test_download_finqa_records_revision_size_and_sha256(tmp_path: Path) -> None:
    payloads = {
        "train.json": b"[]\n",
        "dev.json": b"[{}]\n",
        "test.json": b"[{\"id\": \"q1\"}]\n",
    }

    def fetch_bytes(url: str) -> bytes:
        return payloads[url.rsplit("/", 1)[-1]]

    manifest = download_finqa(tmp_path, fetch_bytes=fetch_bytes)

    assert manifest["revision"] == FINQA_REVISION
    for filename, content in payloads.items():
        assert (tmp_path / filename).read_bytes() == content
        assert manifest["files"][filename]["bytes"] == len(content)
        assert manifest["files"][filename]["sha256"] == hashlib.sha256(content).hexdigest()
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
