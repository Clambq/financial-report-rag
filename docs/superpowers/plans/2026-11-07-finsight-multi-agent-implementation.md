# FinSight Multi Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the verified financial-report RAG repository into a tested LangGraph-based FinSight application with five real workflow nodes, typed tools, verification, SSE streaming, vLLM integration, reproducible agent evaluation, and Docker deployment material.

**Architecture:** Keep the current retrieval, Qwen, Decimal executor, and metrics modules as the trusted core. Add a typed application layer in `finqa_rag.agents` where a bounded LangGraph state machine routes Supervisor, Research, Quant, Verifier, and Report nodes; expose it through synchronous and SSE FastAPI endpoints and evaluate it through saved per-case predictions.

**Tech Stack:** Python 3.12, LangGraph 1.x, Pydantic 2.x, FastAPI, OpenAI Python client 2.x, existing Transformers/Sentence Transformers stack, pytest, Docker Compose, optional vLLM OpenAI-compatible server.

**Spec:** `docs/superpowers/specs/2026-11-07-finsight-multi-agent-design.md`

## Global Constraints

- Preserve the existing locked retrieval and controlled arithmetic metrics and keep all original tests passing.
- Default unit tests must be network-free and GPU-free; real-model and Docker checks remain opt-in.
- Python remains `>=3.12,<3.13`; add `langgraph>=1.0,<2` and `openai>=2.0,<3` with bounded major versions.
- Every request has a finite graph path, `max_retries=1` by default, and no unbounded agent or tool loop.
- Every tool input and output is a Pydantic model; unknown tools fail before implementation code runs.
- Numerical answers come from the restricted Decimal executor, never free-form LLM arithmetic.
- A citation is valid only when its ID belongs to the current request's retrieved evidence.
- The vLLM adapter uses current structured-output fields and treats prefix caching as an optional deployment optimization.
- No 97 percent tool-success result may be published unless prediction-row recomputation produces it.
- Newly implemented behavior must not be described as having existed during the 2025 internship.
- All Windows pytest commands use `--basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan` to avoid the sandboxed system temporary directory.

## Review Focus

- Duplicate or out-of-order SSE events under async cancellation must still have monotonic sequence numbers; Task 6 adds a disconnect and sequence-order test.
- A model plan that requests an unknown tool must stop before execution and record `unknown_tool`; Task 2 adds a registry rejection test.
- A syntactically valid calculation with an unsupported or forward-referencing program must become human review rather than a silent fallback; Tasks 2 and 5 add these tests.
- A draft that cites an evidence ID from another request must fail verification and exhaust into human review after one repair; Task 5 adds a cross-request citation test.
- One failed retriever in the parallel Research node must produce a traced single-retriever degradation instead of discarding the successful result; Task 4 adds a partial-failure test.

---

## File Structure

### New application modules

- `src/finqa_rag/agents/state.py`: shared LangGraph state, plans, issues, traces, and terminal result models.
- `src/finqa_rag/agents/events.py`: typed execution events and SSE serialization.
- `src/finqa_rag/agents/tools.py`: Pydantic tool contracts, registry, and adapters over existing retrieval/execution services.
- `src/finqa_rag/agents/model.py`: structured-model protocol plus Transformers and vLLM adapters.
- `src/finqa_rag/agents/nodes.py`: five node implementations with no graph wiring.
- `src/finqa_rag/agents/graph.py`: LangGraph construction, conditional routing, retry limits, and run/stream façade.
- `src/finqa_rag/agents/__init__.py`: stable public exports.
- `src/finqa_rag/api/agent_schemas.py`: request/response API models for agent endpoints.
- `src/finqa_rag/api/agent_service.py`: JSON and SSE endpoint integration.
- `src/finqa_rag/evaluation/agent_metrics.py`: metric definitions and aggregation.
- `src/finqa_rag/evaluation/agent_runner.py`: case runner and prediction/artifact writer.
- `configs/agent_eval.yaml`: locked retry, routing, and evaluation settings.
- `tests/fixtures/agent_eval_cases.json`: committed evaluation case definitions.
- `Dockerfile`: CPU-safe application image.
- `compose.yaml`: application plus optional vLLM profile.
- `.env.example`: non-secret runtime configuration.

### Modified modules

- `pyproject.toml`: runtime dependencies and optional vLLM evaluation extra.
- `src/finqa_rag/api/app.py`: register agent JSON and SSE routes when a service is provided.
- `src/finqa_rag/api/bootstrap.py`: construct FinSight tools, model adapter, graph, and API service.
- `src/finqa_rag/cli.py`: add `evaluate-agent` and FinSight serving configuration.
- `README.md`: architecture, commands, results boundary, deployment, and limitations.
- `docs/interview-guide.md`: explain the implemented multi-agent system and evidence boundaries.

---

### Task 1: Typed State and Event Protocol

**Files:**
- Create: `src/finqa_rag/agents/__init__.py`
- Create: `src/finqa_rag/agents/state.py`
- Create: `src/finqa_rag/agents/events.py`
- Create: `tests/test_agent_state.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: existing `Citation`-compatible evidence fields and Python/Pydantic primitives.
- Produces: `AgentPlan`, `AgentIssue`, `NodeTrace`, `FinSightState`, `AgentResult`, `AgentEvent`, and `encode_sse_event(event: AgentEvent) -> str` for all later tasks.

- [ ] **Step 1: Add dependencies and write failing state-model tests**

Create tests asserting that `AgentPlan` accepts only `research`, `quant`, and `report` capability names; `FinSightState` defaults to zero retries and empty evidence; and `AgentResult` requires a final answer or `needs_human_review=True` with a reason.

- [ ] **Step 2: Write failing event tests**

Test `encode_sse_event()` for `run_started` and `run_failed`, asserting `id: <sequence>`, `event: <event_type>`, one compact JSON `data:` line, and a terminating blank line.

- [ ] **Step 3: Run the focused tests and confirm missing-module failure**

Run: `pytest tests/test_agent_state.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: collection fails because `finqa_rag.agents` does not exist.

- [ ] **Step 4: Implement the Pydantic models and SSE encoder**

Use exact public model names from the Interfaces block. `AgentEvent.sequence` is a non-negative integer; event types are a `Literal` containing the ten values in the spec; `encode_sse_event` serializes with `model_dump(mode="json")`.

- [ ] **Step 5: Run state tests and the full baseline suite**

Run: `pytest tests/test_agent_state.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Run: `pytest -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: focused tests pass and all original 66 tests remain green.

- [ ] **Step 6: Commit the approved design and Task 1**

```bash
git add pyproject.toml src/finqa_rag/agents tests/test_agent_state.py docs/superpowers/specs/2026-11-07-finsight-multi-agent-design.md docs/superpowers/plans/2026-11-07-finsight-multi-agent-implementation.md
git commit -m "feat: define FinSight agent state and events"
```

### Task 2: Typed Tool Registry and Core Tool Adapters

**Files:**
- Create: `src/finqa_rag/agents/tools.py`
- Create: `tests/test_agent_tools.py`
- Modify: `src/finqa_rag/retrieval/types.py`

**Interfaces:**
- Consumes: `ReportRetrievalIndex.search(report_id: str, question: str)`, existing report tables, and `execute_program(program: str, table=...)`.
- Produces: `ToolErrorCode`, `ToolFailure`, `ToolCallRecord`, `RetrieveEvidenceInput/Output`, `QueryReportTableInput/Output`, `ExecuteFinancialProgramInput/Output`, `ComposeGroundedReportInput/Output`, `VerifyGroundingInput/Output`, `ToolRegistry.register(name, input_type, output_type, handler)`, and `ToolRegistry.ainvoke(name: str, payload: dict[str, object]) -> BaseModel`.

- [ ] **Step 1: Write failing schema and registry tests**

Test blank report IDs, non-positive `top_k`, extra input fields, duplicate registration, unknown tool rejection with code `unknown_tool`, and validation rejection with code `validation_error` before handler invocation.

Also validate the input and output contracts for `compose_grounded_report` and `verify_grounding`, including rejection of citation IDs not present in their declared allowed-evidence lists.

- [ ] **Step 2: Write failing retrieval and table-query adapter tests**

Use fake retrievers and tables to assert ranked evidence is preserved, a missing report returns `execution_error`, and case-insensitive normalized row matching returns stable row indices and values.

- [ ] **Step 3: Write failing safe-execution adapter tests**

Assert `divide(39, 4)` returns `9.75`; `divide(1, 0)`, a forward reference, and `__import__` produce structured `execution_error` failures and never evaluate Python.

- [ ] **Step 4: Run focused tests to verify failure**

Run: `pytest tests/test_agent_tools.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: collection fails because `finqa_rag.agents.tools` is missing.

- [ ] **Step 5: Implement registry, contracts, and adapters**

Handlers may be synchronous or async; `ToolRegistry.invoke` normalizes both through one async public method `ainvoke(name: str, payload: dict[str, object]) -> BaseModel`. Convert expected domain failures into `ToolFailure` without catching unexpected programmer errors.

- [ ] **Step 6: Run focused and regression tests**

Run: `pytest tests/test_agent_tools.py tests/test_program.py tests/test_retrieval.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/finqa_rag/agents/tools.py src/finqa_rag/retrieval/types.py tests/test_agent_tools.py
git commit -m "feat: add typed FinSight tool registry"
```

### Task 3: Structured Model Protocol and Runtime Adapters

**Files:**
- Create: `src/finqa_rag/agents/model.py`
- Create: `tests/test_agent_model.py`
- Modify: `src/finqa_rag/generation/qwen.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: existing `QwenGenerator` and Pydantic schema classes.
- Produces: `ModelMessage`, `StructuredModel.generate(schema: type[T], messages: Sequence[ModelMessage]) -> T`, `StructuredModel.generate_text(messages) -> str`, `TransformersModelAdapter`, and `VLLMOpenAIAdapter`.

- [ ] **Step 1: Write failing protocol and Transformers adapter tests**

Use a fake `QwenGenerator` to assert structured JSON is parsed into the requested Pydantic type, invalid JSON raises `ModelOutputError`, and text output is passed through without silently repairing content.

- [ ] **Step 2: Write failing vLLM request-contract tests**

Use a fake OpenAI client and assert the adapter sends `response_format={"type":"json_schema", ...}` with `schema.model_json_schema()`, configured model/base URL, timeout, and deterministic temperature. Assert malformed content becomes `ModelOutputError`.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `pytest tests/test_agent_model.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: collection fails because `finqa_rag.agents.model` is missing.

- [ ] **Step 4: Implement the protocol and both adapters**

The vLLM adapter uses the OpenAI-compatible client but imports it lazily. Do not add vLLM itself as a Windows application dependency. The Transformers adapter builds JSON-only prompts and validates the returned content with `model_validate_json`.

- [ ] **Step 5: Run model and existing Qwen tests**

Run: `pytest tests/test_agent_model.py tests/test_qwen.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: all selected tests pass without loading a real model.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/finqa_rag/agents/model.py src/finqa_rag/generation/qwen.py tests/test_agent_model.py
git commit -m "feat: add structured model adapters"
```

### Task 4: Supervisor, Research, Quant, and Report Nodes

**Files:**
- Create: `src/finqa_rag/agents/nodes.py`
- Create: `tests/test_agent_nodes.py`
- Modify: `src/finqa_rag/agents/state.py`
- Modify: `src/finqa_rag/agents/tools.py`

**Interfaces:**
- Consumes: `StructuredModel`, `ToolRegistry.ainvoke`, and Task 1 state models.
- Produces: `supervisor_node(state, runtime)`, `research_node(state, runtime)`, `quant_node(state, runtime)`, and `report_node(state, runtime)`, each returning explicit state updates.

- [ ] **Step 1: Write failing Supervisor tests**

Assert structured plans route text to research/report, numerical to research/quant/report, mixed to all capabilities, and an unknown tool request becomes an `AgentIssue(code="unknown_tool")` without invocation.

- [ ] **Step 2: Write failing Research concurrency and degradation tests**

Use two async fake retrievers with synchronization events to prove both start before either finishes. Assert two successes are fused; one expected failure records a degradation issue and preserves the successful ranking; two failures produce human review.

- [ ] **Step 3: Write failing Quant tests**

Assert table query precedes program execution, the returned value is stored as a verified calculation candidate, and invalid program output records an execution issue instead of calculating in the model.

- [ ] **Step 4: Write failing Report tests**

Assert prompts contain only retrieved evidence and calculation results, returned citation IDs are stored for later verification, and Report has no registry capability for retrieval or calculation tools.

Assert Report invokes the typed `compose_grounded_report` contract rather than returning unvalidated model text.

- [ ] **Step 5: Run focused tests and confirm failure**

Run: `pytest tests/test_agent_nodes.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: collection fails because `finqa_rag.agents.nodes` is missing.

- [ ] **Step 6: Implement the four node functions and runtime dependency container**

Add `AgentRuntime` containing model, registry, optional separate lexical/dense search callables, clock, and event sink. Research uses `asyncio.gather(..., return_exceptions=True)` and deterministic fusion ordering.

- [ ] **Step 7: Run focused tests**

Run: `pytest tests/test_agent_nodes.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: all selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/finqa_rag/agents/nodes.py src/finqa_rag/agents/state.py src/finqa_rag/agents/tools.py tests/test_agent_nodes.py
git commit -m "feat: implement FinSight workflow nodes"
```

### Task 5: Verifier and Bounded LangGraph Workflow

**Files:**
- Create: `src/finqa_rag/agents/graph.py`
- Create: `tests/test_agent_graph.py`
- Create: `tests/test_agent_verifier.py`
- Modify: `src/finqa_rag/agents/nodes.py`
- Modify: `src/finqa_rag/agents/__init__.py`

**Interfaces:**
- Consumes: four Task 4 nodes plus `FinSightState` and `AgentRuntime`.
- Produces: `verifier_node(state, runtime)`, `build_finsight_graph(runtime, max_retries: int = 1)`, `FinSightAgent.ainvoke(request) -> AgentResult`, and `FinSightAgent.astream(request) -> AsyncIterator[AgentEvent]`.

- [ ] **Step 1: Write failing citation and numerical verification tests**

Assert valid current-request citations pass; an invented ID and an ID retrieved in a different request fail; deterministic numerical claims must match the Quant result; unsupported comparison is marked unverifiable rather than accepted.

Assert Verifier invokes the typed `verify_grounding` contract and stores every returned issue before choosing a route.

- [ ] **Step 2: Write failing graph happy-path tests**

Assert text path visits Supervisor, Research, Report, Verifier; numerical path visits Supervisor, Research, Quant, Report, Verifier; and successful verification returns a final answer with the ordered node trace.

- [ ] **Step 3: Write failing correction and termination tests**

Assert one verification failure routes to the responsible node once, `retry_count` becomes one, a second failure terminates with `needs_human_review=True`, and total node visits remain bounded.

Assert transient model/retrieval failures use the configured bounded retry policy, while unexpected programmer errors surface without being mislabeled as a model-recoverable failure.

- [ ] **Step 4: Run focused tests and confirm failure**

Run: `pytest tests/test_agent_verifier.py tests/test_agent_graph.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: collection fails because verifier and graph functions are missing.

- [ ] **Step 5: Implement Verifier and compile the LangGraph state machine**

Use explicit conditional edges after Supervisor and Verifier. Compile with an in-memory checkpointer for tests and accept a caller-supplied checkpointer for deployment. Graph state must carry request ID so evidence cannot cross request boundaries.

- [ ] **Step 6: Run focused and original API tests**

Run: `pytest tests/test_agent_verifier.py tests/test_agent_graph.py tests/test_api.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/finqa_rag/agents tests/test_agent_verifier.py tests/test_agent_graph.py
git commit -m "feat: orchestrate bounded FinSight graph"
```

### Task 6: FastAPI JSON and SSE Endpoints

**Files:**
- Create: `src/finqa_rag/api/agent_schemas.py`
- Create: `src/finqa_rag/api/agent_service.py`
- Create: `tests/test_agent_api.py`
- Modify: `src/finqa_rag/api/app.py`
- Modify: `src/finqa_rag/api/bootstrap.py`

**Interfaces:**
- Consumes: `FinSightAgent.ainvoke` and `FinSightAgent.astream`.
- Produces: `AgentAskRequest`, `AgentAskResponse`, `AgentApiService`, `POST /agent/ask`, and `POST /agent/stream` with media type `text/event-stream`.

- [ ] **Step 1: Write failing request and JSON endpoint tests**

Assert blank report/question validation, extra-field rejection, response trace serialization, human-review responses, and existing `/ask`, `/health`, and `/metrics` compatibility.

- [ ] **Step 2: Write failing SSE order and cancellation tests**

Assert event IDs are strictly increasing, first event is `run_started`, terminal event is exactly one of `run_finished` or `run_failed`, heartbeats are comments rather than data events, and disconnect cancellation closes the agent iterator without subsequent tool events.

Assert unexpected exceptions return a sanitized `run_failed` payload without a Python stack trace.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `pytest tests/test_agent_api.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: collection or route assertions fail because agent API integration is missing.

- [ ] **Step 4: Implement schemas, service, and route registration**

Use FastAPI `StreamingResponse` with an async generator and `media_type="text/event-stream"`. Register agent routes only when `create_app(..., agent_service=...)` receives the service so old unit tests remain lightweight.

- [ ] **Step 5: Run API and graph tests**

Run: `pytest tests/test_agent_api.py tests/test_api.py tests/test_agent_graph.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/finqa_rag/api src/finqa_rag/agents tests/test_agent_api.py
git commit -m "feat: expose FinSight JSON and SSE APIs"
```

### Task 7: Reproducible Agent Evaluation and Metric Verification

**Files:**
- Create: `src/finqa_rag/evaluation/agent_metrics.py`
- Create: `src/finqa_rag/evaluation/agent_runner.py`
- Create: `tests/test_agent_evaluation.py`
- Create: `tests/fixtures/agent_eval_cases.json`
- Create: `configs/agent_eval.yaml`
- Modify: `src/finqa_rag/cli.py`
- Modify: `src/finqa_rag/evaluation/report.py`

**Interfaces:**
- Consumes: `FinSightAgent.ainvoke`, `ToolCallRecord`, node traces, and fixture cases.
- Produces: `AgentCase`, `AgentPrediction`, `compute_agent_metrics(predictions) -> dict`, `run_agent_evaluation(...)`, CLI `evaluate-agent`, and artifact verification for agent prediction rows.

- [ ] **Step 1: Write failing metric-definition tests**

Use fixed prediction objects to assert task completion, all-attempt tool success, first-attempt success, after-retry success, citation validity, numerical verification, error-category counts, model/tool-call counts, and interpolated P50/P95 latency.

- [ ] **Step 2: Write failing fixture and runner tests**

Cover text, numerical, mixed, malformed arguments, unknown tool, missing evidence, unsupported operation, retriever degradation, verification retry, and retry exhaustion. Assert prediction rows preserve denominators and case IDs.

- [ ] **Step 3: Write failing recomputation and CLI tests**

Assert metrics reconstructed from JSONL equal the summary JSON, incomplete `.partial` files are not promoted, and CLI prints the output directory and query count.

- [ ] **Step 4: Run focused tests and confirm failure**

Run: `pytest tests/test_agent_evaluation.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: collection fails because agent evaluation modules are missing.

- [ ] **Step 5: Implement metric aggregation, runner, atomic artifacts, and CLI command**

Follow the existing retrieval/numerical artifact pattern. Save `metrics_summary.json`, `predictions.jsonl`, `experiment_manifest.json`, and `reproducibility.json` under the caller's output directory.

- [ ] **Step 6: Run evaluation and report regression tests**

Run: `pytest tests/test_agent_evaluation.py tests/test_report.py tests/test_cli.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: all selected tests pass.

- [ ] **Step 7: Run the deterministic fixture evaluation**

Run: `finqa-rag evaluate-agent --cases tests/fixtures/agent_eval_cases.json --output artifacts/agent/smoke --model fake`

Expected: artifacts exist and `verify-artifacts` recomputes every agent metric. Label these results as deterministic application tests, not real-model tool-use accuracy.

- [ ] **Step 8: Commit**

```bash
git add configs/agent_eval.yaml tests/fixtures/agent_eval_cases.json src/finqa_rag/evaluation src/finqa_rag/cli.py tests/test_agent_evaluation.py
git commit -m "feat: evaluate FinSight agent reliability"
```

### Task 8: Docker and vLLM Deployment Path

**Files:**
- Create: `Dockerfile`
- Create: `compose.yaml`
- Create: `.env.example`
- Create: `tests/test_deployment_contract.py`
- Modify: `src/finqa_rag/api/bootstrap.py`
- Modify: `src/finqa_rag/cli.py`

**Interfaces:**
- Consumes: vLLM base URL/model configuration and the FinSight application factory.
- Produces: environment-driven `build_finsight_app(...)`, application container health check, and an optional `vllm` Compose profile started with automatic prefix caching.

- [ ] **Step 1: Write failing environment and deployment-contract tests**

Assert missing required vLLM model/base URL produces a clear configuration error, secrets are absent from committed defaults, Compose contains health checks, and the vLLM command contains the current prefix-cache flag.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `pytest tests/test_deployment_contract.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: failure because deployment files and factory configuration are missing.

- [ ] **Step 3: Implement environment-driven bootstrap and container files**

The default container starts the application. The `vllm` profile is Linux/GPU-only, exposes an OpenAI-compatible endpoint, enables prefix caching, and does not run in the default unit-test workflow.

- [ ] **Step 4: Run deployment tests and Docker validation when available**

Run: `pytest tests/test_deployment_contract.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Run: `docker compose config`

Expected: tests pass; Docker config validates if Docker is installed. If Docker is unavailable, record the check as not executed rather than claiming success.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile compose.yaml .env.example src/finqa_rag/api/bootstrap.py src/finqa_rag/cli.py tests/test_deployment_contract.py
git commit -m "build: package FinSight application runtime"
```

### Task 9: Documentation, Real-Model Gate, and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/interview-guide.md`
- Create: `docs/finsight-architecture.md`
- Create: `docs/finsight-evaluation.md`
- Create: `tests/test_finsight_documentation.py`
- Modify: `artifacts/RESULTS.md` only when a verified new evaluation exists.

**Interfaces:**
- Consumes: all implemented commands, routes, metrics, limitations, and optional vLLM results.
- Produces: reproducible operator documentation, architecture explanation, interview guide, and a verified final package.

- [ ] **Step 1: Write failing documentation-contract tests**

Assert README contains five node names, `/agent/ask`, `/agent/stream`, agent evaluation reproduction, Docker/vLLM commands, result-boundary language, and no unverified `97%` claim.

- [ ] **Step 2: Run documentation tests and confirm failure**

Run: `pytest tests/test_finsight_documentation.py -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Expected: failure until documentation is updated.

- [ ] **Step 3: Write architecture, evaluation, README, and interview documentation**

Document exact data flow, each node's boundary, retry and verification logic, SSE event examples, vLLM current structured-output request, prefix caching, evaluation denominators, and the difference between deterministic smoke results and real-model results.

- [ ] **Step 4: Run the optional real-model evaluation only when a compatible runtime is available**

Start vLLM with prefix caching, then run `evaluate-agent` with the vLLM adapter and a new output directory. Publish the measured tool-success value only after `verify-artifacts` succeeds. If no compatible runtime is available, leave the value unclaimed and document the exact unexecuted command.

- [ ] **Step 5: Run the complete verification suite**

Run: `pytest -q --basetemp C:\Users\hyzha\Documents\Codex\2026-11-07\xian\work\pytest-finsight-plan`

Run: `finqa-rag verify-artifacts`

Run: `finqa-rag evaluate-agent --cases tests/fixtures/agent_eval_cases.json --output artifacts/agent/final --model fake`

Run: `git diff --check`

Expected: all tests pass, existing hashes verify, deterministic agent artifacts recompute exactly, and no whitespace errors remain.

- [ ] **Step 6: Inspect repository status and commit documentation**

```bash
git add README.md docs artifacts/RESULTS.md tests/test_finsight_documentation.py
git commit -m "docs: explain verified FinSight multi-agent system"
```

- [ ] **Step 7: Perform final review against the approved spec**

Confirm every acceptance criterion maps to passing tests or an explicitly documented optional-runtime limitation. Do not mark the implementation complete while an in-scope deterministic criterion remains unverified.

