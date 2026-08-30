"""Parse FinQA reports into report-local evidence units."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from finqa_rag.data.models import EvidenceUnit, FinQAExample


_WHITESPACE = re.compile(r"\s+")


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return _WHITESPACE.sub(" ", text).strip()


def _serialize_table_row(header: list[object], row: list[object]) -> str:
    cells = list(row) + [""] * max(0, len(header) - len(row))
    return " | ".join(
        f"{_normalize_text(column)}: {_normalize_text(value)}"
        for column, value in zip(header, cells, strict=False)
    )


def _parse_example(entry: dict[str, object], split: str) -> FinQAExample:
    example_id = _normalize_text(entry.get("id", ""))
    if not example_id:
        raise ValueError("FinQA example is missing id")
    report_id = _normalize_text(entry.get("filename", ""))
    if not report_id:
        report_id = example_id.rsplit("-", 1)[0]

    evidence: list[EvidenceUnit] = []
    aliases: dict[str, list[str]] = {}
    text_rows = list(entry.get("pre_text", [])) + list(entry.get("post_text", []))
    for index, value in enumerate(text_rows):
        text = _normalize_text(value)
        evidence_id = f"text_{index}"
        evidence.append(EvidenceUnit(evidence_id, report_id, split, "text", text))
        aliases.setdefault(_normalize_text(value), []).append(evidence_id)

    table = list(entry.get("table", []))
    if table:
        header = list(table[0])
        for index, raw_row in enumerate(table):
            row = list(raw_row)
            evidence_id = f"table_{index}"
            raw_text = _normalize_text(" | ".join(map(str, row)))
            serialized = raw_text if index == 0 else _serialize_table_row(header, row)
            evidence.append(
                EvidenceUnit(evidence_id, report_id, split, "table", serialized)
            )
            aliases.setdefault(raw_text, []).append(evidence_id)
            normalized_serialized = _normalize_text(serialized)
            if normalized_serialized != raw_text:
                aliases.setdefault(normalized_serialized, []).append(evidence_id)

    qa = entry.get("qa")
    if not isinstance(qa, dict):
        raise ValueError(f"FinQA example {example_id} has invalid qa payload")
    by_id = {item.evidence_id: item for item in evidence}
    gold_ids: list[str] = []
    gold_inds = qa.get("gold_inds", {})
    if not isinstance(gold_inds, dict):
        raise ValueError(f"FinQA example {example_id} has invalid gold_inds")
    for raw_label, raw_value in gold_inds.items():
        label = _normalize_text(raw_label)
        if label in by_id:
            gold_ids.append(label)
            continue
        matches = aliases.get(_normalize_text(raw_value), [])
        if len(matches) == 1:
            gold_ids.append(matches[0])
            continue
        raise ValueError(
            f"FinQA example {example_id} has unmatched gold evidence {label}"
        )

    return FinQAExample(
        example_id=example_id,
        report_id=report_id,
        split=split,
        question=_normalize_text(qa.get("question", "")),
        answer=_normalize_text(qa.get("answer", "")),
        exe_answer=_normalize_text(qa.get("exe_ans", "")),
        program=_normalize_text(qa.get("program", "")),
        evidence=tuple(evidence),
        gold_evidence_ids=tuple(gold_ids),
    )


def load_split(path: Path, split: str) -> list[FinQAExample]:
    """Load a public FinQA split and validate all gold evidence labels."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"FinQA split must be a JSON list: {path}")
    return [_parse_example(entry, split) for entry in payload]
