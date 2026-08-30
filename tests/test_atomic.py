import json
from pathlib import Path

from finqa_rag.common.atomic import atomic_write_json, atomic_write_jsonl


def test_atomic_write_json_replaces_complete_file(tmp_path: Path) -> None:
    target = tmp_path / "metrics.json"

    atomic_write_json(target, {"query_count": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"query_count": 2}
    assert not (tmp_path / "metrics.json.partial").exists()


def test_atomic_write_jsonl_writes_one_object_per_line(tmp_path: Path) -> None:
    target = tmp_path / "predictions.jsonl"

    atomic_write_jsonl(target, [{"id": "q1"}, {"id": "q2"}])

    assert [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
    ] == [{"id": "q1"}, {"id": "q2"}]
