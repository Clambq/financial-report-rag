# Financial Report RAG System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a reproducible FinQA-based financial-report RAG system that produces defensible retrieval and numerical-execution metrics before packaging the pipeline as a FastAPI service and GitHub portfolio project.

**Architecture:** FinQA examples are normalized into report-local evidence units consumed by interchangeable BM25 and dense retrievers. Reciprocal-rank fusion and a cross-encoder reranker produce cited top-five evidence; a safe symbolic-program executor and Qwen direct-calculation baseline support the numerical comparison. Offline evaluators and the FastAPI service call the same modules and write atomic, machine-readable artifacts.

**Tech Stack:** Python 3.12, PyTorch 2.12.1 with CUDA 13.0 wheels, Transformers 5.x, Sentence Transformers 5.x, NumPy, rank-bm25, FastAPI, Pydantic 2, Typer, PyYAML, pytest, HTTPX.

**Spec:** `docs/superpowers/specs/2026-08-30-financial-report-rag-design.md`

## Global Constraints

- Use official FinQA data pinned to Git revision `0f16e2867befa6840783e58be38c9efb9229d742`.
- Tune retrieval settings only on the development split; run the locked test configuration without test-set tuning.
- Use `Qwen/Qwen3-4B-Instruct-2507` for the controlled direct-calculation condition.
- Use `sentence-transformers/all-MiniLM-L6-v2` for dense retrieval and `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking.
- Use report-local candidate evidence for every retrieval method.
- Never execute generated Python; the calculator accepts only the explicit FinQA operation grammar.
- Write artifacts atomically and preserve raw per-query predictions alongside summaries.
- Do not claim a metric improvement unless the verified test artifact proves it.
- Keep raw FinQA data, model caches, virtual environments, and incomplete artifact files out of Git.

## File Map

```text
financial-report-rag/
├── pyproject.toml                         # package metadata and core/dev/ML dependencies
├── requirements-gpu.txt                  # pinned CUDA-enabled PyTorch wheel
├── .gitignore                            # generated data, caches, environments, partial results
├── configs/
│   ├── retrieval.yaml                    # locked retrieval models and ranking parameters
│   └── numerical.yaml                    # Qwen model and numerical evaluation parameters
├── src/finqa_rag/
│   ├── __init__.py
│   ├── cli.py                            # prepare/evaluate/serve command entry points
│   ├── common/
│   │   ├── atomic.py                     # atomic JSON and JSONL writes
│   │   └── timing.py                     # warm-up-aware latency summaries
│   ├── data/
│   │   ├── models.py                     # immutable example/evidence domain types
│   │   ├── download.py                   # pinned FinQA downloader and checksum manifest
│   │   └── finqa.py                      # parsing, normalization, gold-evidence matching
│   ├── retrieval/
│   │   ├── types.py                      # ranked-evidence protocol types
│   │   ├── bm25.py                       # lexical retriever
│   │   ├── dense.py                      # sentence-transformer retriever
│   │   ├── fusion.py                     # reciprocal-rank fusion
│   │   └── reranker.py                   # cross-encoder reranker
│   ├── reasoning/
│   │   ├── normalize.py                  # financial-number parsing and comparison
│   │   ├── program.py                    # safe FinQA program parser/executor
│   │   └── router.py                     # text-versus-numerical routing
│   ├── generation/
│   │   ├── qwen.py                       # local Qwen direct-calculation adapter
│   │   └── answer.py                     # structured cited-answer formatting
│   ├── evaluation/
│   │   ├── retrieval_metrics.py          # Recall/Hit/MRR/NDCG definitions
│   │   ├── retrieval_runner.py           # comparable retrieval experiments
│   │   ├── numerical_metrics.py          # exact match and error-reduction metrics
│   │   ├── numerical_runner.py           # controlled Qwen-versus-calculator experiment
│   │   └── report.py                     # artifact-to-Markdown and resume-bullet generation
│   └── api/
│       ├── schemas.py                    # validated request/response models
│       ├── pipeline.py                   # shared online RAG orchestration
│       └── app.py                        # FastAPI application factory
├── tests/
│   ├── fixtures/finqa_mini.json          # two-report deterministic test fixture
│   ├── test_atomic.py
│   ├── test_data.py
│   ├── test_retrieval.py
│   ├── test_fusion.py
│   ├── test_retrieval_metrics.py
│   ├── test_program.py
│   ├── test_numerical_metrics.py
│   ├── test_numerical_runner.py
│   ├── test_report.py
│   └── test_api.py
├── data/.gitkeep
├── artifacts/.gitkeep
├── LICENSE
├── README.md
└── docs/interview-guide.md
```

---

### Task 1: Reproducible Project Foundation

**Files:**
- Create: `pyproject.toml`
- Create: `requirements-gpu.txt`
- Create: `.gitignore`
- Create: `src/finqa_rag/__init__.py`
- Create: `src/finqa_rag/common/atomic.py`
- Create: `src/finqa_rag/common/timing.py`
- Create: `tests/test_atomic.py`
- Create: `tests/test_timing.py`
- Create: `data/.gitkeep`
- Create: `artifacts/.gitkeep`

**Interfaces:**
- Consumes: Python executable at `C:/Users/hyzha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`.
- Produces: installed package `finqa_rag`; `atomic_write_json(path: Path, payload: Mapping[str, object]) -> None`; `atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None`; `latency_summary(samples_ms: Sequence[float]) -> dict[str, float]`.

- [ ] **Step 1: Write the failing atomic-write tests**

```python
from pathlib import Path
import json

from finqa_rag.common.atomic import atomic_write_json, atomic_write_jsonl


def test_atomic_write_json_replaces_complete_file(tmp_path: Path) -> None:
    target = tmp_path / "metrics.json"
    atomic_write_json(target, {"query_count": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"query_count": 2}
    assert not (tmp_path / "metrics.json.partial").exists()


def test_atomic_write_jsonl_writes_one_object_per_line(tmp_path: Path) -> None:
    target = tmp_path / "predictions.jsonl"
    atomic_write_jsonl(target, [{"id": "q1"}, {"id": "q2"}])
    assert [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()] == [
        {"id": "q1"},
        {"id": "q2"},
    ]


def test_latency_summary_reports_linear_percentiles() -> None:
    assert latency_summary([10.0, 20.0, 30.0, 40.0, 50.0]) == {
        "p50_ms": 30.0,
        "p95_ms": 48.0,
    }
```

- [ ] **Step 2: Create exact package, GPU, and ignore configuration, then verify the tests fail because the package is absent**

Create `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=80,<81"]
build-backend = "setuptools.build_meta"

[project]
name = "finqa-rag"
version = "0.1.0"
description = "Reproducible hybrid retrieval and numerical reasoning over FinQA"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy>=2.2,<3",
  "rank-bm25>=0.2.2,<0.3",
  "pydantic>=2.11,<3",
  "pyyaml>=6.0.2,<7",
  "typer>=0.16,<1",
  "fastapi>=0.116,<1",
  "uvicorn>=0.35,<1",
]

[project.optional-dependencies]
ml = [
  "torch==2.12.1",
  "transformers>=5.3,<6",
  "sentence-transformers>=5,<6",
  "accelerate>=1.10,<2",
]
dev = [
  "pytest>=8.4,<9",
  "pytest-cov>=6.2,<7",
  "httpx>=0.28,<1",
]

[project.scripts]
finqa-rag = "finqa_rag.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers"
```

Create `requirements-gpu.txt` with:

```text
--index-url https://download.pytorch.org/whl/cu130
torch==2.12.1
```

Create `.gitignore` with:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
data/raw/
data/cache/
model-cache/
artifacts/**/*.partial
artifacts/**/smoke/
artifacts/**/test-run-*/
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_atomic.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'finqa_rag.common'`.

- [ ] **Step 3: Implement atomic artifact writes**

```python
def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    partial.replace(path)


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("latency samples must not be empty")
    values = np.asarray(samples_ms, dtype=float)
    return {
        "p50_ms": float(np.percentile(values, 50, method="linear")),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
    }
```

- [ ] **Step 4: Create the environment and run the foundation tests**

Run:

```powershell
& 'C:/Users/hyzha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -e '.[dev]'
.venv/Scripts/python.exe -m pytest tests/test_atomic.py tests/test_timing.py -v
```

Expected: three tests pass.

- [ ] **Step 5: Commit the foundation**

```powershell
git add pyproject.toml requirements-gpu.txt .gitignore src tests/test_atomic.py tests/test_timing.py data/.gitkeep artifacts/.gitkeep
git commit -m "build: establish reproducible Python project"
```

---

### Task 2: Pinned FinQA Download and Evidence Parsing

**Files:**
- Create: `src/finqa_rag/data/models.py`
- Create: `src/finqa_rag/data/download.py`
- Create: `src/finqa_rag/data/finqa.py`
- Create: `tests/fixtures/finqa_mini.json`
- Create: `tests/test_data.py`
- Modify: `src/finqa_rag/cli.py`

**Interfaces:**
- Consumes: official raw JSON files at `data/raw/finqa/{train,dev,test}.json`.
- Produces: `EvidenceUnit(evidence_id, report_id, split, kind, text)`; `FinQAExample(example_id, report_id, split, question, answer, exe_answer, program, evidence, gold_evidence_ids)`; `load_split(path: Path, split: str) -> list[FinQAExample]`; `download_finqa(output_dir: Path) -> dict[str, object]`.

- [ ] **Step 1: Commit a two-report fixture and write failing parser tests**

The fixture must include narrative evidence, a header plus two table rows, one single-step percentage program, one nested two-step program, and `gold_inds` entries that match both text and table evidence.

```python
def test_load_split_builds_stable_evidence_and_gold_ids() -> None:
    examples = load_split(FIXTURE, "test")
    first = examples[0]
    assert first.example_id == "report-a-0"
    assert first.question == "What was the revenue growth rate?"
    assert {item.kind for item in first.evidence} == {"text", "table"}
    assert set(first.gold_evidence_ids).issubset({item.evidence_id for item in first.evidence})


def test_load_split_rejects_unmatched_gold_evidence(tmp_path: Path) -> None:
    broken = make_fixture_with_gold_text("not present in report")
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="unmatched gold evidence"):
        load_split(path, "test")
```

- [ ] **Step 2: Run the parser tests and observe the missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data.py -v`

Expected: failure importing `finqa_rag.data.finqa`.

- [ ] **Step 3: Implement immutable domain models and normalization**

```python
@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    evidence_id: str
    report_id: str
    split: str
    kind: Literal["text", "table"]
    text: str


@dataclass(frozen=True, slots=True)
class FinQAExample:
    example_id: str
    report_id: str
    split: str
    question: str
    answer: str
    exe_answer: str
    program: str
    evidence: tuple[EvidenceUnit, ...]
    gold_evidence_ids: tuple[str, ...]
```

Normalize whitespace and Unicode punctuation but preserve case and numerical symbols in stored evidence. Serialize each table row as `Header 1: value | Header 2: value` and match `gold_inds` values after whitespace normalization.

- [ ] **Step 4: Implement the pinned downloader and checksum manifest**

Use raw URLs rooted at:

```text
https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/
```

Download `train.json`, `dev.json`, and `test.json` with `urllib.request`, compute SHA-256 for every completed file, and write `data/raw/finqa/manifest.json` atomically with revision, URLs, byte counts, and hashes.

- [ ] **Step 5: Add `finqa-rag prepare-data` and run all data tests**

```python
@app.command("prepare-data")
def prepare_data(output_dir: Path = Path("data/raw/finqa")) -> None:
    manifest = download_finqa(output_dir)
    for split in ("train", "dev", "test"):
        load_split(output_dir / f"{split}.json", split)
    typer.echo(json.dumps(manifest, indent=2))
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_data.py -v`

Expected: all data tests pass without network access.

- [ ] **Step 6: Download and validate official data**

Run: `.venv/Scripts/finqa-rag.exe prepare-data`

Expected: three JSON files and one manifest are created; every split loads without unmatched gold evidence. If the official gold labels expose a parser mismatch, add the smallest fixture reproducing that exact structure before changing the parser.

- [ ] **Step 7: Commit the data layer**

```powershell
git add src/finqa_rag/data src/finqa_rag/cli.py tests/fixtures/finqa_mini.json tests/test_data.py
git commit -m "feat: add reproducible FinQA data layer"
```

---

### Task 3: Retrieval Metrics and BM25 Baseline

**Files:**
- Create: `src/finqa_rag/retrieval/types.py`
- Create: `src/finqa_rag/retrieval/bm25.py`
- Create: `src/finqa_rag/evaluation/retrieval_metrics.py`
- Create: `src/finqa_rag/evaluation/retrieval_runner.py`
- Create: `tests/test_retrieval.py`
- Create: `tests/test_retrieval_metrics.py`
- Modify: `src/finqa_rag/cli.py`

**Interfaces:**
- Consumes: `FinQAExample` and report-local `EvidenceUnit` sequences.
- Produces: `ScoredEvidence(evidence: EvidenceUnit, score: float, rank: int, method: str)`; `BM25Retriever.fit(evidence: Sequence[EvidenceUnit]) -> BM25Retriever`; `BM25Retriever.search(query: str, top_k: int) -> list[ScoredEvidence]`; `evaluate_ranking(ranked_ids: Sequence[str], gold_ids: AbstractSet[str], k: int) -> QueryRetrievalMetrics`; `run_retrieval_evaluation(examples: Sequence[FinQAExample], methods: Sequence[str], output_dir: Path, top_k: int = 5) -> dict[str, object]`.

- [ ] **Step 1: Write failing BM25 ranking and metric tests**

```python
def test_bm25_ranks_exact_financial_terms_first() -> None:
    retriever = BM25Retriever().fit(EVIDENCE)
    ranked = retriever.search("operating income 2024", top_k=2)
    assert ranked[0].evidence.evidence_id == "table_1"
    assert [item.rank for item in ranked] == [1, 2]


def test_retrieval_metrics_use_all_gold_evidence() -> None:
    result = evaluate_ranking(["text_2", "table_1", "text_9"], {"text_2", "table_1"}, k=2)
    assert result.recall_at_k == 1.0
    assert result.hit_at_k == 1.0
    assert result.mrr == 1.0
    assert result.ndcg_at_k == 1.0
```

- [ ] **Step 2: Run the tests and confirm they fail on absent classes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval.py tests/test_retrieval_metrics.py -v`

Expected: import failures for retrieval modules.

- [ ] **Step 3: Implement deterministic tokenization and BM25 ranking**

Use lowercase alphanumeric/percentage tokens from `re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", text.lower())`. Break equal-score ties by stable evidence identifier so repeated runs produce identical rankings.

- [ ] **Step 4: Implement query and aggregate metrics**

```python
def evaluate_ranking(ranked_ids: Sequence[str], gold_ids: AbstractSet[str], k: int = 5) -> QueryRetrievalMetrics:
    top = list(ranked_ids[:k])
    hits = [1 if evidence_id in gold_ids else 0 for evidence_id in top]
    recall = sum(hits) / len(gold_ids)
    first = next((index + 1 for index, value in enumerate(ranked_ids) if value in gold_ids), None)
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(gold_ids), k)))
    return QueryRetrievalMetrics(recall, float(any(hits)), 0.0 if first is None else 1.0 / first, dcg / ideal)
```

Aggregate by arithmetic mean and record the exact query count and unmatched-label count.

- [ ] **Step 5: Add the BM25 evaluation command and atomic artifacts**

`finqa-rag evaluate-retrieval --split dev --methods bm25 --output artifacts/retrieval/dev-bm25` must write `metrics_summary.json`, `predictions.jsonl`, and `experiment_manifest.json`. Each prediction records query ID, gold IDs, ranked IDs, scores, latency, and method.

- [ ] **Step 6: Run tests and the first real BM25 development evaluation**

Run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_retrieval.py tests/test_retrieval_metrics.py -v
.venv/Scripts/finqa-rag.exe evaluate-retrieval --split dev --methods bm25 --output artifacts/retrieval/dev-bm25
```

Expected: tests pass and the summary reports a non-zero query count with Recall@5, Hit@5, MRR, NDCG@5, P50 latency, and P95 latency.

- [ ] **Step 7: Commit the verified BM25 baseline**

```powershell
git add src/finqa_rag/retrieval src/finqa_rag/evaluation src/finqa_rag/cli.py tests/test_retrieval.py tests/test_retrieval_metrics.py
git commit -m "feat: evaluate reproducible BM25 baseline"
```

---

### Task 4: Dense, Hybrid, and Cross-Encoder Retrieval

**Files:**
- Create: `src/finqa_rag/retrieval/dense.py`
- Create: `src/finqa_rag/retrieval/fusion.py`
- Create: `src/finqa_rag/retrieval/reranker.py`
- Create: `tests/test_fusion.py`
- Modify: `tests/test_retrieval.py`
- Modify: `src/finqa_rag/evaluation/retrieval_runner.py`
- Create: `configs/retrieval.yaml`

**Interfaces:**
- Consumes: evidence units, BM25 rankings, model identifiers from configuration.
- Produces: `DenseRetriever.fit(evidence: Sequence[EvidenceUnit]) -> DenseRetriever`; `DenseRetriever.search(query: str, top_k: int) -> list[ScoredEvidence]`; `reciprocal_rank_fusion(rankings: Mapping[str, Sequence[ScoredEvidence]], weights: Mapping[str, float], rrf_k: int, top_k: int) -> list[ScoredEvidence]`; `CrossEncoderReranker.rerank(query: str, candidates: Sequence[ScoredEvidence], top_k: int) -> list[ScoredEvidence]`; comparable method summaries.

- [ ] **Step 1: Write failing fusion tests with exact expected ranks**

```python
def test_rrf_combines_rankings_and_breaks_ties_by_evidence_id() -> None:
    fused = reciprocal_rank_fusion(
        rankings={"bm25": ["a", "b", "c"], "dense": ["b", "a", "d"]},
        weights={"bm25": 1.0, "dense": 1.0},
        rrf_k=60,
        top_k=4,
    )
    assert [item.evidence_id for item in fused] == ["a", "b", "c", "d"]
```

- [ ] **Step 2: Run the fusion tests and confirm the missing implementation failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fusion.py -v`

Expected: import failure for `finqa_rag.retrieval.fusion`.

- [ ] **Step 3: Implement dense retrieval with normalized embeddings**

Load `sentence-transformers/all-MiniLM-L6-v2`, call `encode(..., normalize_embeddings=True, convert_to_numpy=True)`, cache report-local evidence matrices by the ordered evidence-ID tuple, and rank dot products with stable evidence-ID tie breaks.

- [ ] **Step 4: Implement reciprocal-rank fusion and reranking**

```python
score[evidence_id] += weights[method] / (rrf_k + rank)
```

The reranker loads `cross-encoder/ms-marco-MiniLM-L-6-v2`, scores `(query, evidence.text)` pairs, preserves the original fused score as metadata, and returns the final top five.

- [ ] **Step 5: Add model-free unit tests using injected fake encoders**

Tests must inject deterministic embedding and cross-encoder callables so CI does not download models. Verify that dense ranking, RRF, and reranking each change the ordering in an expected way.

- [ ] **Step 6: Install the GPU and ML dependencies and verify CUDA**

Run:

```powershell
.venv/Scripts/python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -m pip install -e '.[ml,dev]'
.venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected: PyTorch `2.12.1`, CUDA availability `True`, and `NVIDIA GeForce RTX 5060 Ti`.

- [ ] **Step 7: Tune only on development data and lock configuration**

Evaluate candidate depths 10 and 20 and RRF lexical weights 0.5, 1.0, and 2.0 with dense weight 1.0. Select the configuration with highest development Recall@5; break ties by MRR, then lower P95 latency. Write the chosen exact values to `configs/retrieval.yaml` together with model identifiers and `top_k: 5`.

- [ ] **Step 8: Run the locked test comparison twice**

Run the same test command twice into `artifacts/retrieval/test-run-1` and `test-run-2`, then compare summary metrics and ranked evidence IDs. Expected: identical rankings and metrics; latency may differ and is compared descriptively rather than for exact equality.

- [ ] **Step 9: Promote the verified retrieval artifacts and commit**

Copy the verified summary and predictions through the report command to `artifacts/retrieval/metrics_summary.json` and `artifacts/retrieval/predictions.jsonl`, then commit code and configuration but not raw models or duplicate experiment caches.

```powershell
git add src/finqa_rag/retrieval src/finqa_rag/evaluation/retrieval_runner.py configs/retrieval.yaml tests
git commit -m "feat: benchmark dense hybrid and reranked retrieval"
```

---

### Task 5: Safe FinQA Numerical Executor and Metrics

**Files:**
- Create: `src/finqa_rag/reasoning/normalize.py`
- Create: `src/finqa_rag/reasoning/program.py`
- Create: `src/finqa_rag/reasoning/router.py`
- Create: `src/finqa_rag/evaluation/numerical_metrics.py`
- Create: `tests/test_program.py`
- Create: `tests/test_numerical_metrics.py`

**Interfaces:**
- Consumes: FinQA program strings and gold `exe_answer` strings.
- Produces: `execute_program(program: str) -> Decimal`; `normalize_answer(value: str | Decimal) -> Decimal`; `answers_match(predicted, gold, tolerance=Decimal("0.0001")) -> bool`; `aggregate_numerical(records) -> dict[str, object]`; `route_question(program: str | None) -> Literal["text", "numerical"]`.

- [ ] **Step 1: Write failing tests for operations and rejected input**

```python
@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("subtract(120, 100), divide(#0, 100)", Decimal("0.2")),
        ("multiply(2.5, 4)", Decimal("10")),
        ("table_average(10, 20, 30)", Decimal("20")),
        ("exp(2, 3)", Decimal("8")),
    ],
)
def test_execute_program(program: str, expected: Decimal) -> None:
    assert execute_program(program) == expected


def test_executor_rejects_arbitrary_python() -> None:
    with pytest.raises(ProgramError, match="unsupported operation"):
        execute_program("__import__('os').system('whoami')")
```

- [ ] **Step 2: Run tests and confirm missing implementation failures**

Run: `.venv/Scripts/python.exe -m pytest tests/test_program.py tests/test_numerical_metrics.py -v`

- [ ] **Step 3: Implement a parser over the fixed operation grammar**

Support `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`, `table_sum`, `table_average`, `table_max`, and `table_min`. Parse comma-separated calls without `eval`, resolve `#0`, `#1`, and later references only against previously computed results, and use `Decimal` arithmetic. Reject division by zero, forward references, non-finite values, unknown operations, and malformed numeric tokens with `ProgramError`.

Recognize the FinQA constant tokens `const_1`, `const_100`, `const_1000`, `const_m1`, and `const_m2` through an immutable lookup table. Reject every other `const_` token until an official-data fixture proves it is required and a failing test defines its value.

- [ ] **Step 4: Implement financial-answer normalization**

Strip currency symbols and commas, convert trailing `%` to its displayed numerical value rather than silently dividing by 100, map parenthesized values to negatives, and quantize comparison through an explicit tolerance. Preserve the raw value in prediction artifacts.

- [ ] **Step 5: Implement numerical aggregate metrics**

Return query count, exact-match count/rate, arithmetic-error count/rate, P50/P95 latency, absolute error-rate difference, and relative error reduction. Return JSON `null` for relative reduction when the direct condition has zero errors.

- [ ] **Step 6: Run all reasoning tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_program.py tests/test_numerical_metrics.py -v`

Expected: all reasoning and numerical-metric tests pass.

```powershell
git add src/finqa_rag/reasoning src/finqa_rag/evaluation/numerical_metrics.py tests/test_program.py tests/test_numerical_metrics.py
git commit -m "feat: add safe financial program execution"
```

---

### Task 6: Controlled Qwen-versus-Calculator Evaluation

**Files:**
- Create: `src/finqa_rag/generation/qwen.py`
- Create: `src/finqa_rag/evaluation/numerical_runner.py`
- Create: `configs/numerical.yaml`
- Create: `tests/test_numerical_runner.py`
- Modify: `src/finqa_rag/cli.py`

**Interfaces:**
- Consumes: held-out FinQA numerical examples with gold evidence and gold symbolic programs.
- Produces: `GenerationResult(raw_text: str, parsed_text: str, latency_ms: float)`; `QwenGenerator.calculate_direct(question: str, evidence: Sequence[EvidenceUnit], program: str) -> GenerationResult`; `QwenGenerator.answer_text(question: str, evidence: Sequence[ScoredEvidence]) -> GenerationResult`; `QwenGenerator.generate_program(question: str, evidence: Sequence[ScoredEvidence]) -> GenerationResult`; `run_numerical_evaluation(examples: Sequence[FinQAExample], generator: DirectCalculator, output_dir: Path) -> dict[str, object]`; numerical summary, predictions, and experiment manifest artifacts.

- [ ] **Step 1: Write a failing runner test using a fake generator**

```python
def test_numerical_runner_compares_identical_examples(tmp_path: Path) -> None:
    generator = FakeGenerator({"q1": "15", "q2": "wrong"})
    summary = run_numerical_evaluation(EXAMPLES, generator, tmp_path)
    assert summary["query_count"] == 2
    assert summary["conditions"]["direct_llm"]["exact_match_count"] == 1
    assert summary["conditions"]["python_tool"]["exact_match_count"] == 2
    assert len(read_jsonl(tmp_path / "predictions.jsonl")) == 2
```

- [ ] **Step 2: Run the test and confirm the absent-runner failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_numerical_runner.py -v`

- [ ] **Step 3: Implement the reusable Qwen adapter**

Load `Qwen/Qwen3-4B-Instruct-2507` with `torch_dtype="auto"`, `device_map="auto"`, and deterministic generation (`do_sample=False`, `max_new_tokens=64`). Format one prompt containing the question, gold evidence, symbolic program, and the instruction `Return only the final numeric answer.` Parse the final non-empty output line and preserve the full raw response.

The same adapter exposes two service methods. `answer_text` receives ranked evidence and instructs the model to answer only from that context while citing evidence IDs in brackets. `generate_program` receives ranked evidence and instructs the model to return one expression composed only of the supported FinQA operations, numeric literals, constant tokens, and backward result references. The safe executor remains the enforcement boundary; no model output is evaluated as Python.

- [ ] **Step 4: Implement the controlled runner and atomic result files**

Filter to examples with a non-empty program and executable gold answer. Both conditions must consume the same ordered example list. Record example ID, question, evidence IDs, program, gold answer, direct raw/predicted answer, tool answer, correctness flags, error category, and both latencies.

- [ ] **Step 5: Configure a deterministic held-out run**

Create `configs/numerical.yaml` with the exact locked values below. Select the first 100 IDs after sorting eligible test examples by SHA-256 of `"42:" + example_id`; store that ordered ID list in the manifest and never resample after seeing accuracy.

```yaml
model_id: Qwen/Qwen3-4B-Instruct-2507
split: test
seed: 42
sample_size: 100
max_new_tokens: 64
do_sample: false
torch_dtype: auto
device_map: auto
```

- [ ] **Step 6: Run a ten-query smoke test and inspect every result**

Run: `.venv/Scripts/finqa-rag.exe evaluate-numerical --limit 10 --output artifacts/numerical/smoke`

Expected: ten complete records, no arbitrary-code execution, parseable calculator outputs, and direct raw responses preserved.

- [ ] **Step 7: Run the locked numerical experiment twice**

Run the configured experiment into `test-run-1` and `test-run-2`. Expected: identical sampled IDs, calculator answers, direct generated answers, and correctness metrics. Confirm the reported relative reduction matches the formula in the spec.

- [ ] **Step 8: Promote verified numerical artifacts and commit**

```powershell
git add src/finqa_rag/generation/qwen.py src/finqa_rag/evaluation/numerical_runner.py src/finqa_rag/cli.py configs/numerical.yaml tests/test_numerical_runner.py
git commit -m "feat: compare Qwen calculation with Python execution"
```

---

### Task 7: Shared RAG Pipeline and FastAPI Service

**Files:**
- Create: `src/finqa_rag/generation/answer.py`
- Create: `src/finqa_rag/api/schemas.py`
- Create: `src/finqa_rag/api/pipeline.py`
- Create: `src/finqa_rag/api/app.py`
- Create: `tests/test_api.py`
- Modify: `src/finqa_rag/cli.py`

**Interfaces:**
- Consumes: the locked retrieval configuration, shared `QwenGenerator`, router/calculator, and one loaded report's evidence.
- Produces: `AskRequest(report_id: str, question: str)`; `Citation(evidence_id: str, evidence_type: str, source_text: str, retrieval_score: float, rank: int)`; `AskResponse(question: str, answer: str, route: Literal["text", "numerical"], citations: list[Citation], needs_human_review: bool, error: str | None, latency_ms: float)`; `RAGPipeline.answer(request: AskRequest) -> AskResponse`; `create_app(pipeline: RAGPipeline) -> FastAPI`; `/health`, `/ask`, and `/metrics`.

- [ ] **Step 1: Write failing API contract tests with a fake pipeline**

```python
def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_structured_citations() -> None:
    response = client.post("/ask", json={"report_id": "report-a", "question": "What changed?"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] in {"text", "numerical"}
    assert body["citations"][0]["evidence_id"] == "table_1"
    assert body["needs_human_review"] is False
    assert body["latency_ms"] >= 0


def test_ask_rejects_blank_question() -> None:
    response = client.post("/ask", json={"report_id": "report-a", "question": "   "})
    assert response.status_code == 422
```

- [ ] **Step 2: Run the API tests and confirm missing schemas/app failures**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v`

- [ ] **Step 3: Implement Pydantic schemas and the shared pipeline**

`AskRequest` validates non-blank `report_id` and `question`. The pipeline uses the same hybrid/reranker modules already evaluated offline and always returns the ranked source citations. Text routes call `QwenGenerator.answer_text`. Numerical routes call `QwenGenerator.generate_program`, pass only the parsed expression into `execute_program`, and format the deterministic result. Unsupported or malformed programs return `needs_human_review=true` with a structured error instead of falling back to unverified arithmetic.

- [ ] **Step 4: Implement the application factory and endpoints**

```python
def create_app(pipeline: RAGPipeline) -> FastAPI:
    app = FastAPI(title="Financial Report RAG", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        return pipeline.answer(request)

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        return pipeline.verified_metrics()

    return app
```

- [ ] **Step 5: Run API and complete-suite tests**

Run:

```powershell
.venv/Scripts/python.exe -m pytest -v
.venv/Scripts/finqa-rag.exe serve --host 127.0.0.1 --port 8000
```

Expected: all tests pass; `/docs` loads; `/health` returns 200; a fixture-backed `/ask` response contains citations.

- [ ] **Step 6: Commit the service**

```powershell
git add src/finqa_rag/api src/finqa_rag/generation/answer.py src/finqa_rag/cli.py tests/test_api.py
git commit -m "feat: serve cited financial answers with FastAPI"
```

---

### Task 8: Verified Results, Resume Bullets, and Interview Guide

**Files:**
- Create: `src/finqa_rag/evaluation/report.py`
- Create: `tests/test_report.py`
- Create: `README.md`
- Create: `docs/interview-guide.md`
- Create: `LICENSE`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: verified retrieval and numerical summaries plus experiment manifests.
- Produces: `write_reports(retrieval_summary: Mapping[str, object], numerical_summary: Mapping[str, object], output_dir: Path) -> ReportPaths`; `verify_artifacts(artifact_root: Path) -> None`; `artifacts/experiment_manifest.json`; `artifacts/RESULTS.md`; `artifacts/RESUME_BULLETS.md`; a portfolio README; and an interview guide whose values come directly from JSON artifacts.

- [ ] **Step 1: Write failing report-generation tests**

```python
def test_resume_bullet_uses_verified_metrics(tmp_path: Path) -> None:
    paths = write_reports(RETRIEVAL_SUMMARY, NUMERICAL_SUMMARY, tmp_path)
    text = paths.resume_bullets.read_text(encoding="utf-8")
    assert str(RETRIEVAL_SUMMARY["query_count"]) in text
    assert format_percent(RETRIEVAL_SUMMARY["methods"]["bm25"]["recall_at_5"]) in text
    assert re.search(r"\[[A-Z]\]", text) is None


def test_report_does_not_claim_improvement_when_hybrid_is_lower(tmp_path: Path) -> None:
    summary = retrieval_summary(bm25_recall=0.70, hybrid_recall=0.68)
    text = write_reports(summary, NUMERICAL_SUMMARY, tmp_path).resume_bullets.read_text(encoding="utf-8")
    assert "improved" not in text.lower()
    assert "benchmarked" in text.lower()
```

- [ ] **Step 2: Run report tests and confirm the absent-report failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_report.py -v`

- [ ] **Step 3: Implement artifact validation and report generation**

Reject summaries missing query counts, model/config identifiers, or matching manifests. Generate an improvement bullet only when hybrid-plus-reranker test Recall@5 exceeds the chosen baseline. Express arithmetic changes as both percentage points and relative reduction, and omit relative reduction when its JSON value is `null`.

Also write `artifacts/experiment_manifest.json` containing the SHA-256 hashes and relative paths of both verified summaries, both verified prediction files, their run manifests, the report generation timestamp, and the Git commit being documented.

- [ ] **Step 4: Create the README from verified artifacts**

The README must include problem statement, architecture flow, why BM25 and dense retrieval are combined, leakage-control policy, reproducible Windows commands, exact models, results tables generated from JSON, one cited example, error analysis, limitations, API example, repository layout, and license/data attribution. It must state that the numerical comparison is a controlled arithmetic-stage experiment using gold evidence and gold programs.

Create an MIT `LICENSE` for the project code. State separately that FinQA remains governed by its upstream repository and is downloaded rather than redistributed by this project.

- [ ] **Step 5: Create the interview guide**

`docs/interview-guide.md` must contain a 30-second pitch, a two-minute walkthrough, definitions of every metric, the origin of every resume number, at least twenty project-specific questions with concise answers, five failure cases from saved predictions, and explanations of why raw PDF parsing and QLoRA were deferred.

- [ ] **Step 6: Run final reproducibility and integrity checks**

Run:

```powershell
.venv/Scripts/python.exe -m pytest -v
.venv/Scripts/finqa-rag.exe verify-artifacts
git status --short
```

Expected: full suite passes; artifact verifier confirms counts, manifests, and formulas; Git status contains only the intentionally versioned reports/documentation or is clean.

- [ ] **Step 7: Inspect representative outputs manually**

Inspect at least five successful and five failed retrieval predictions and every numerical mismatch. Record evidence-missing, retrieval-ranking, direct-arithmetic, program-data, normalization, and unsupported-operation categories with counts in `artifacts/RESULTS.md`.

- [ ] **Step 8: Commit the verified portfolio package**

```powershell
git add README.md LICENSE docs/interview-guide.md src/finqa_rag/evaluation/report.py tests/test_report.py artifacts/RESULTS.md artifacts/RESUME_BULLETS.md artifacts/experiment_manifest.json artifacts/retrieval/metrics_summary.json artifacts/retrieval/predictions.jsonl artifacts/numerical/metrics_summary.json artifacts/numerical/predictions.jsonl .gitignore
git commit -m "docs: publish verified results and interview guide"
```

---

## Completion Verification

After Task 8, verify every success criterion from the spec against authoritative evidence:

1. inspect the data manifest and split counts;
2. compare the two locked retrieval runs query-by-query;
3. compare the two numerical runs query-by-query;
4. recompute all summary metrics independently from prediction JSONL;
5. exercise `/health`, `/ask`, and `/metrics` through HTTPX;
6. run the complete test suite from a clean process;
7. confirm README and resume bullets contain no placeholder metrics;
8. confirm every numerical claim can be traced to an artifact path and manifest;
9. confirm no raw data, model weights, secrets, or partial files are tracked by Git.
