"""Generate user-facing reports only from verified machine-readable metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from finqa_rag.common.timing import latency_summary
from finqa_rag.evaluation.numerical_metrics import aggregate_numerical


@dataclass(frozen=True, slots=True)
class ReportPaths:
    results: Path
    resume_bullets: Path


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    partial.replace(path)


def _method(summary: Mapping[str, object], name: str) -> Mapping[str, float]:
    methods = summary.get("methods")
    if not isinstance(methods, Mapping) or name not in methods:
        raise ValueError(f"retrieval summary is missing method: {name}")
    value = methods[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid metrics for retrieval method: {name}")
    return value  # type: ignore[return-value]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"artifact must contain JSON objects: {path}")
    return rows


def _assert_close(actual: float, expected: object, label: str) -> None:
    if not math.isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"{label} mismatch: recomputed={actual}, summary={expected}")


def _canonical_hash(
    rows: list[dict[str, object]],
    excluded_keys: set[str],
) -> str:
    cleaned = [
        {key: value for key, value in row.items() if key not in excluded_keys}
        for row in rows
    ]
    encoded = json.dumps(
        cleaned,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_artifacts(artifact_root: Path) -> dict[str, object]:
    """Recompute published summaries and verify deterministic-run hashes."""

    retrieval_dir = artifact_root / "retrieval"
    numerical_dir = artifact_root / "numerical"
    retrieval_summary = _read_json(retrieval_dir / "metrics_summary.json")
    retrieval_manifest = _read_json(retrieval_dir / "experiment_manifest.json")
    retrieval_repro = _read_json(retrieval_dir / "reproducibility.json")
    retrieval_rows = _read_jsonl(retrieval_dir / "predictions.jsonl")
    methods = list(retrieval_manifest["methods"])
    expected_retrieval_rows = int(retrieval_summary["query_count"]) * len(methods)
    if len(retrieval_rows) != expected_retrieval_rows:
        raise ValueError("retrieval prediction count does not match manifest")
    for method in methods:
        rows = [row for row in retrieval_rows if row["method"] == method]
        if len(rows) != int(retrieval_summary["query_count"]):
            raise ValueError(f"retrieval method row count mismatch: {method}")
        published = _method(retrieval_summary, method)
        for source_key, summary_key in (
            ("recall_at_k", "recall_at_5"),
            ("hit_at_k", "hit_at_5"),
            ("mrr", "mrr"),
            ("ndcg_at_k", "ndcg_at_5"),
        ):
            recomputed = sum(float(row["metrics"][source_key]) for row in rows) / len(
                rows
            )
            _assert_close(recomputed, published[summary_key], f"{method}.{summary_key}")
        latency = latency_summary([float(row["latency_ms"]) for row in rows])
        _assert_close(latency["p50_ms"], published["p50_ms"], f"{method}.p50_ms")
        _assert_close(latency["p95_ms"], published["p95_ms"], f"{method}.p95_ms")
    retrieval_hash = _canonical_hash(retrieval_rows, {"latency_ms"})
    if retrieval_hash != retrieval_repro["canonical_predictions_sha256"]:
        raise ValueError("retrieval prediction reproducibility hash mismatch")

    numerical_summary = _read_json(numerical_dir / "metrics_summary.json")
    numerical_manifest = _read_json(numerical_dir / "experiment_manifest.json")
    numerical_repro = _read_json(numerical_dir / "reproducibility.json")
    numerical_rows = _read_jsonl(numerical_dir / "predictions.jsonl")
    if len(numerical_rows) != int(numerical_summary["query_count"]):
        raise ValueError("numerical prediction count does not match summary")
    if [row["example_id"] for row in numerical_rows] != numerical_manifest["sample_ids"]:
        raise ValueError("numerical sample IDs do not match manifest")
    recomputed_numerical = aggregate_numerical(
        [
            {
                "direct_correct": bool(row["direct_correct"]),
                "tool_correct": bool(row["tool_correct"]),
                "direct_latency_ms": float(row["direct_latency_ms"]),
                "tool_latency_ms": float(row["tool_latency_ms"]),
            }
            for row in numerical_rows
        ]
    )
    published_conditions = numerical_summary["conditions"]
    recomputed_conditions = recomputed_numerical["conditions"]
    for condition in ("direct_llm", "python_tool"):
        for key in (
            "exact_match_count",
            "exact_match_rate",
            "arithmetic_error_count",
            "arithmetic_error_rate",
            "p50_ms",
            "p95_ms",
        ):
            _assert_close(
                float(recomputed_conditions[condition][key]),
                published_conditions[condition][key],
                f"{condition}.{key}",
            )
    for key in ("absolute_error_rate_reduction", "relative_error_reduction"):
        _assert_close(
            float(recomputed_numerical[key]),
            numerical_summary[key],
            key,
        )
    numerical_hash = _canonical_hash(
        numerical_rows,
        {"direct_latency_ms", "tool_latency_ms"},
    )
    if numerical_hash != numerical_repro["canonical_predictions_sha256"]:
        raise ValueError("numerical prediction reproducibility hash mismatch")
    publication = _read_json(artifact_root / "experiment_manifest.json")
    root_resolved = artifact_root.resolve()
    for entry in publication["files"]:
        relative_path = Path(entry["path"])
        published_path = (artifact_root / relative_path).resolve()
        if not published_path.is_relative_to(root_resolved):
            raise ValueError(f"publication path escapes artifact root: {relative_path}")
        actual_hash = hashlib.sha256(published_path.read_bytes()).hexdigest()
        if actual_hash != entry["sha256"]:
            raise ValueError(f"publication SHA-256 mismatch: {relative_path}")
    return {
        "retrieval_prediction_rows": len(retrieval_rows),
        "numerical_prediction_rows": len(numerical_rows),
        "retrieval_canonical_sha256": retrieval_hash,
        "numerical_canonical_sha256": numerical_hash,
        "formulas_verified": True,
        "publication_hashes_verified": True,
    }


def analyze_error_artifacts(artifact_root: Path) -> dict[str, object]:
    """Derive descriptive error counts from the published prediction rows."""

    retrieval_rows = _read_jsonl(artifact_root / "retrieval" / "predictions.jsonl")
    per_example: dict[str, dict[str, dict[str, object]]] = {}
    for row in retrieval_rows:
        per_example.setdefault(str(row["example_id"]), {})[str(row["method"])] = row
    comparison = {
        "improved_vs_bm25": 0,
        "regressed_vs_bm25": 0,
        "unchanged_vs_bm25": 0,
        "rerank_complete": 0,
        "rerank_partial": 0,
        "rerank_miss": 0,
    }
    for methods in per_example.values():
        bm25_recall = float(methods["bm25"]["metrics"]["recall_at_k"])
        rerank_recall = float(methods["hybrid_rerank"]["metrics"]["recall_at_k"])
        if rerank_recall > bm25_recall:
            comparison["improved_vs_bm25"] += 1
        elif rerank_recall < bm25_recall:
            comparison["regressed_vs_bm25"] += 1
        else:
            comparison["unchanged_vs_bm25"] += 1
        if rerank_recall == 1:
            comparison["rerank_complete"] += 1
        elif rerank_recall == 0:
            comparison["rerank_miss"] += 1
        else:
            comparison["rerank_partial"] += 1
    reranked_rows = [row for row in retrieval_rows if row["method"] == "hybrid_rerank"]
    representative_miss_ids = [
        str(row["example_id"])
        for row in reranked_rows
        if float(row["metrics"]["recall_at_k"]) == 0
    ][:5]

    numerical_rows = _read_jsonl(artifact_root / "numerical" / "predictions.jsonl")
    mismatches = [row for row in numerical_rows if not bool(row["direct_correct"])]
    numerical_counts = {
        "direct_mismatches": len(mismatches),
        "percent_outputs": 0,
        "verbose_or_truncated": 0,
        "boolean_mismatches": 0,
        "other_numeric_mismatches": 0,
    }
    for row in mismatches:
        parsed = str(row["direct_predicted_answer"]).strip()
        raw = str(row["direct_raw_answer"]).strip()
        if parsed.endswith("%"):
            numerical_counts["percent_outputs"] += 1
        elif "\n" in raw and raw != parsed:
            numerical_counts["verbose_or_truncated"] += 1
        elif parsed.lower() in {"yes", "no"}:
            numerical_counts["boolean_mismatches"] += 1
        else:
            numerical_counts["other_numeric_mismatches"] += 1
    return {
        "retrieval": {
            **comparison,
            "representative_miss_ids": representative_miss_ids,
        },
        "numerical": {
            **numerical_counts,
            "representative_mismatch_ids": [
                str(row["example_id"]) for row in mismatches[:5]
            ],
        },
    }


def write_reports(
    retrieval_summary: Mapping[str, object],
    numerical_summary: Mapping[str, object],
    output_dir: Path,
    error_analysis: Mapping[str, object] | None = None,
) -> ReportPaths:
    retrieval_count = int(retrieval_summary["query_count"])
    numerical_count = int(numerical_summary["query_count"])
    bm25 = _method(retrieval_summary, "bm25")
    reranked = _method(retrieval_summary, "hybrid_rerank")
    bm25_recall = float(bm25["recall_at_5"])
    final_recall = float(reranked["recall_at_5"])
    recall_delta_pp = (final_recall - bm25_recall) * 100
    bm25_mrr = float(bm25["mrr"])
    final_mrr = float(reranked["mrr"])

    conditions = numerical_summary["conditions"]
    if not isinstance(conditions, Mapping):
        raise ValueError("numerical summary is missing conditions")
    direct = conditions["direct_llm"]
    tool = conditions["python_tool"]
    if not isinstance(direct, Mapping) or not isinstance(tool, Mapping):
        raise ValueError("numerical conditions are invalid")
    direct_error = float(direct["arithmetic_error_rate"])
    tool_error = float(tool["arithmetic_error_rate"])
    relative_reduction = numerical_summary.get("relative_error_reduction")

    if final_recall > bm25_recall:
        retrieval_bullet = (
            f"- Built and evaluated BM25, dense, hybrid, and cross-encoder-reranked retrieval "
            f"on {retrieval_count} held-out FinQA queries; improved Recall@5 from "
            f"{format_percent(bm25_recall)} to {format_percent(final_recall)} "
            f"(+{recall_delta_pp:.2f} pp) and MRR@5 from {bm25_mrr:.3f} to "
            f"{final_mrr:.3f}."
        )
    else:
        retrieval_bullet = (
            f"- Built and benchmarked BM25, dense, hybrid, and cross-encoder-reranked retrieval "
            f"on {retrieval_count} held-out FinQA queries; observed Recall@5 of "
            f"{format_percent(bm25_recall)} for BM25 and "
            f"{format_percent(final_recall)} after reranking."
        )
    relative_clause = (
        ""
        if relative_reduction is None
        else f", a {format_percent(float(relative_reduction))} relative reduction"
    )
    numerical_bullet = (
        f"- Integrated a restricted Python executor for financial calculations; in a "
        f"controlled {numerical_count}-query arithmetic-stage evaluation using identical "
        f"gold evidence and programs, reduced "
        f"error rate from {format_percent(direct_error)} to "
        f"{format_percent(tool_error)}{relative_clause}."
    )
    service_bullet = (
        "- Served the pipeline through FastAPI with validated JSON requests, "
        "structured answers, generated-program visibility, verifiable source "
        "citations, latency reporting, and human-review fallback for rejected programs."
    )
    bullets = "# Verified Resume Bullets\n\n" + "\n".join(
        [retrieval_bullet, numerical_bullet, service_bullet]
    )

    error_section = ""
    if error_analysis is not None:
        retrieval_errors = error_analysis["retrieval"]
        numerical_errors = error_analysis["numerical"]
        error_section = f"""
## Error analysis derived from saved predictions

- Hybrid + reranker retrieved all gold evidence for {retrieval_errors['rerank_complete']} queries, partial gold evidence for {retrieval_errors['rerank_partial']}, and no gold evidence in the top five for {retrieval_errors['rerank_miss']}.
- Relative to BM25, per-query recall improved on {retrieval_errors['improved_vs_bm25']} queries, regressed on {retrieval_errors['regressed_vs_bm25']}, and was unchanged on {retrieval_errors['unchanged_vs_bm25']}.
- Representative retrieval misses: {', '.join(f'`{item}`' for item in retrieval_errors['representative_miss_ids'])}.
- The {numerical_errors['direct_mismatches']} direct-Qwen mismatches include {numerical_errors['percent_outputs']} percentage-unit outputs, {numerical_errors['verbose_or_truncated']} verbose or truncated responses, {numerical_errors['boolean_mismatches']} Boolean mismatch, and {numerical_errors['other_numeric_mismatches']} other numerical mismatches.
- Representative direct-Qwen mismatches: {', '.join(f'`{item}`' for item in numerical_errors['representative_mismatch_ids'])}.
"""

    results = f"""# Verified Results

## Retrieval on the locked FinQA test split

| Method | Recall@5 | MRR@5 | P95 latency |
|---|---:|---:|---:|
| BM25 | {format_percent(bm25_recall)} | {bm25_mrr:.3f} | {float(bm25['p95_ms']):.2f} ms |
| Hybrid + cross-encoder reranker | {format_percent(final_recall)} | {final_mrr:.3f} | {float(reranked['p95_ms']):.2f} ms |

The locked comparison contains {retrieval_count:,} held-out queries. Development data selected the candidate count and fusion weight before the public test split was evaluated.

## Controlled numerical-execution experiment

| Condition | Exact match | Arithmetic error rate |
|---|---:|---:|
| Direct {numerical_summary.get('model_id', 'LLM')} calculation | {format_percent(float(direct['exact_match_rate']))} | {format_percent(direct_error)} |
| Safe Python execution | {format_percent(float(tool['exact_match_rate']))} | {format_percent(tool_error)} |

This {numerical_count}-query experiment deliberately gives both conditions the same gold evidence and gold symbolic program. It isolates arithmetic execution; it is not an end-to-end program-generation accuracy claim.
{error_section}
"""
    paths = ReportPaths(
        results=output_dir / "RESULTS.md",
        resume_bullets=output_dir / "RESUME_BULLETS.md",
    )
    _atomic_text(paths.results, results)
    _atomic_text(paths.resume_bullets, bullets)
    return paths
