# FinSight Multi Agent System Design

## Intent

Extend the existing reproducible financial-report RAG project into a real FinSight multi-agent application whose code, tests, deployment path, and evaluation artifacts support the technical claims already present in the submitted resume. The existing retrieval and controlled numerical results remain the evidence base; new claims are published only after a reproducible evaluation produces them.

The target reader is an engineer or interviewer who needs to understand how a financial research request moves through orchestration, retrieval, calculation, verification, reporting, streaming, and failure handling.

## Goals

- Implement a LangGraph workflow with Supervisor, Research, Quant, Verifier, and Report nodes.
- Preserve and reuse the verified FinQA parser, hybrid retrieval, reranker, safe Decimal executor, FastAPI schemas, and artifact verification.
- Give every tool a Pydantic-validated request and response contract.
- Support bounded retries, explicit failure state, evidence-restricted citations, numerical claim checks, and human-review fallback.
- Expose synchronous JSON and Server-Sent Events APIs over the same application service.
- Add an OpenAI-compatible vLLM adapter using current structured-output APIs and document prefix-cache startup configuration.
- Measure task completion, tool-call success, error categories, latency, and token/call usage without preselecting the result.
- Provide Docker packaging, operational documentation, and interview-oriented architecture documentation.

## Non Goals

- Do not claim that newly written code existed during the 2025 internship.
- Do not claim production users, business ROI, cloud scale, arbitrary PDF parsing, or online availability without evidence.
- Do not replace the existing locked retrieval or arithmetic benchmarks with easier data.
- Do not report 97 percent tool success unless the committed evaluation actually produces that result.
- Do not require vLLM or a GPU for unit tests. The vLLM path is an optional runtime adapter with contract tests and a documented Linux GPU deployment path.
- Do not make the LLM the authority for arithmetic, citation validity, or retry limits.

## Constraints

- Python remains 3.12 and the current package layout remains intact.
- Existing 66 baseline tests and verified artifact hashes must continue to pass.
- Network-free unit tests use fakes; model downloads and GPU inference stay outside the default test suite.
- Windows development must work for the deterministic test path, while vLLM deployment is documented for a Linux CUDA container or host.
- The public FinQA data model remains report-local. Persistent multi-tenant indexing and authorization are future work.
- A single request has a fixed retry budget and a finite graph path; no unbounded tool or agent loops are allowed.

## Approaches Considered

### Extend the existing repository

Selected. Add orchestration and application modules around the verified core. This preserves reproducible metrics, minimizes duplicate code, and makes it possible to trace every agent action back to tested retrieval and execution components.

### Create a separate orchestration repository

Rejected for this iteration. It gives a clean service boundary but duplicates packaging and makes a one-machine reproducible demonstration harder.

### Wrap the pipeline with simulated agent labels

Rejected. It would produce a visual demo without meaningful state management, failure recovery, or independent node contracts, and would not withstand technical questioning.

## System Architecture

```text
POST /agent/ask or POST /agent/stream
                 |
                 v
        Supervisor Agent
     classify intent and plan
        /          |          \
       v           v           v
 Research Agent  Quant Agent  Report-only path
 hybrid RAG      data + math
       \           /
        \         /
         v       v
        Verifier Agent
 evidence and number checks
          | pass
          v
        Report Agent
 grounded final response
          |
          v
 answer + citations + trace + metrics
```

The system uses LangGraph as a deterministic application state machine rather than allowing autonomous agents to call each other without limits. Nodes receive raw shared state, invoke typed services, and return explicit state updates. Conditional edges are based on validated routing decisions and verification outcomes.

## Shared State

`FinSightState` stores only information that must survive between nodes:

- request ID, report ID, question, and optional conversation context;
- supervisor plan and required capabilities;
- retrieved evidence and structured data results;
- proposed numerical program and execution result;
- draft answer and final answer;
- citation IDs and numerical claims extracted from the draft;
- verification issues, retry count, and human-review reason;
- node timings, tool attempts, error categories, and event sequence.

Prompts are built on demand from raw state. Formatted prompt strings and derived summaries are not treated as authoritative state.

## Agent Responsibilities

### Supervisor

- Classifies a request into research, numerical, mixed, or unsupported.
- Produces a structured plan containing required nodes and tool intents.
- Routes with explicit conditional edges.
- Rejects unknown tool names and stops after the configured retry budget.

The first implementation uses structured model output when a model adapter is configured and a deterministic rules fallback for offline tests.

### Research

- Runs BM25 and dense retrieval concurrently over the selected report.
- Applies weighted RRF, cross-encoder reranking, and returns top evidence units.
- Preserves evidence IDs, source text, type, score, and rank.
- Never fabricates citations outside the retrieved evidence set.

### Quant

- Reads relevant table rows or retrieved numeric evidence.
- Requests a fixed-grammar FinQA program from the model adapter.
- Executes the program with the existing restricted Decimal executor.
- Rejects unsupported operations, malformed arguments, forward references, division by zero, and non-finite values.

### Verifier

- Confirms every citation ID belongs to the retrieval result for this request.
- Extracts numerical claims from the draft and compares them with tool results or cited source text where deterministic comparison is possible.
- Emits structured issues rather than silently rewriting evidence.
- Routes recoverable failures back to Research, Quant, or Report once; exhausted failures become human-review responses.

### Report

- Generates a concise answer from verified evidence and calculation results.
- Uses only allowed citation IDs.
- Returns structured sections, citations, caveats, and optional calculation trace.
- Cannot directly execute tools or expand the evidence set.

## Tool Contracts

Each tool has Pydantic input and output models and a stable tool name:

- `retrieve_evidence`: report ID, query, top-k -> ranked evidence.
- `query_report_table`: report ID, row selector -> matching structured rows.
- `execute_financial_program`: fixed-grammar program and report table -> Decimal or yes/no result.
- `compose_grounded_report`: question, verified evidence, verified calculations -> draft response.
- `verify_grounding`: draft, allowed evidence, calculations -> verification result.

The registry rejects unknown names before execution. Validation failures, unavailable tools, timeouts, execution errors, and retry exhaustion use distinct error codes for evaluation.

## Model Runtime

Define a model protocol used by agents rather than binding nodes directly to Transformers or vLLM.

- `DeterministicFakeModel` supports network-free tests.
- `TransformersModelAdapter` preserves the current local Qwen demonstration path.
- `VLLMOpenAIAdapter` calls an OpenAI-compatible vLLM server and sends Pydantic JSON schemas through `response_format` or `structured_outputs` according to the installed server contract.

The documented vLLM server command enables automatic prefix caching. Prompts keep stable system instructions and report context prefixes in a consistent order so shared-prefix reuse is possible. Prefix caching is treated as a deployment optimization and must be benchmarked separately from correctness.

## Streaming API

Keep the existing `/ask` endpoint unchanged and add:

- `POST /agent/ask`: returns the complete multi-agent result.
- `POST /agent/stream`: returns `text/event-stream`.

SSE event types are `run_started`, `node_started`, `tool_started`, `tool_finished`, `node_finished`, `verification_failed`, `retrying`, `answer_chunk`, `run_finished`, and `run_failed`. Every event contains request ID, sequence number, timestamp, node, and a typed payload. Heartbeats keep long requests alive. Client disconnect cancellation is handled at await points and recorded without continuing expensive downstream work.

## Concurrency and Performance

- BM25 and dense retrieval execute concurrently through an async coordinator; RRF waits for both successful results.
- If one retriever fails, configuration determines whether to degrade to the remaining retriever or fail closed. The default is a traced single-retriever degradation for the demonstration service.
- Cross-encoder reranking remains bounded to the fused candidate set.
- Node and tool timers report P50 and P95 in evaluation artifacts.
- vLLM prefix caching is enabled in deployment configuration; benchmark output compares cold and repeated-prefix runs when a compatible GPU environment is available.

## Failure Handling

- Pydantic validation failures never reach the tool implementation.
- Transient model or retrieval failures receive bounded exponential-backoff retries.
- Model-generated invalid tool arguments are returned to the Supervisor once for repair.
- Verifier failures route to the responsible node with structured feedback and at most one correction attempt.
- Unexpected exceptions surface as `run_failed` events and structured HTTP responses without leaking stack traces.
- Retry exhaustion, insufficient evidence, or unresolved verification produces `needs_human_review=true` with the reason.

## Evaluation Design

Add a committed agent evaluation fixture with representative text, numerical, mixed, malformed-tool, missing-evidence, unsupported-operation, and retry-exhaustion cases.

Metrics:

- task completion rate;
- tool-call success rate, defined as valid requested tool executions divided by all attempted tool executions;
- first-attempt and after-retry success rates;
- error counts by validation, unknown tool, timeout, execution, verification, and retry exhaustion;
- citation validity and numerical verification pass rates;
- end-to-end and per-node P50 and P95 latency;
- tool attempts and model-call counts per request.

The denominator, fixture IDs, configuration, prediction rows, and metric recomputation command are committed. A published 97 percent value is allowed only if recomputation from prediction rows produces it.

## Testing Strategy

- Unit tests for state models, every tool schema, registry behavior, routing, retry limits, verifier rules, and SSE formatting.
- Graph contract tests covering text, numerical, mixed, verification retry, and human-review paths.
- API tests for JSON responses, SSE ordering, cancellation, and error sanitization.
- Regression tests for existing retrieval, executor, metrics, and artifact verification.
- Optional integration smoke test against a real vLLM server, excluded from the default suite and guarded by environment variables.
- Docker build and container health smoke test when Docker is available.

Implementation follows test-driven development: each behavior begins with a failing test, then the minimum production code, then refactoring while tests remain green.

## Packaging and Deployment

- Add LangGraph and the OpenAI client as application dependencies with bounded versions.
- Add a CPU-safe application image and a compose profile that points to an external or companion vLLM service.
- Include health, readiness, and metrics endpoints.
- Configuration comes from environment variables or a checked-in example file; secrets are never committed.
- README documents local fake-model tests, Transformers demonstration, vLLM deployment, SSE usage, evaluation reproduction, and limitations.

## Delivery Stages

1. Typed state, events, tool contracts, and registry.
2. Agent nodes and LangGraph routing with fake-model graph tests.
3. Integration of existing retrieval and Decimal execution services.
4. Verifier, correction loop, and human-review fallback.
5. Synchronous and SSE FastAPI endpoints.
6. vLLM structured-output adapter and prefix-cache deployment configuration.
7. Agent evaluation runner, artifacts, and metric verifier.
8. Docker packaging, documentation, architecture diagram, and project interview guide.

## Acceptance Criteria

- All original tests pass without modified expected metrics.
- New graph tests demonstrate all five real nodes and bounded conditional routes.
- Tool calls are schema validated and unknown tools cannot execute.
- Citations outside retrieved evidence are rejected by the Verifier.
- Numerical results come from the restricted executor, not free-form model arithmetic.
- Both JSON and SSE APIs pass contract tests.
- Agent evaluation artifacts can be recomputed from prediction rows.
- Docker and vLLM instructions are explicit about what was locally verified.
- README separates verified results, optional runtime claims, and limitations.

