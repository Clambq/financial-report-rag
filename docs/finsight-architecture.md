# FinSight Multi-Agent Architecture

## Scope

FinSight turns the verified financial-report RAG and safe FinQA executor into a bounded application workflow. LangGraph provides explicit conditional edges and an in-memory checkpointer in tests. Pydantic validates every state transition, model response, and tool boundary. The graph never executes a tool name invented by the model.

## Data flow

```text
request
  -> Supervisor
  -> Research
  -> Quant (numerical or mixed only)
  -> Report
  -> Verifier
       | pass -> final answer
       | first recoverable failure -> responsible node -> Report -> Verifier
       | second failure -> human review
```

The graph state carries `request_id`, `report_id`, question, plan, evidence, calculation records, draft/final answer, citation IDs, issues, `retry_count`, repair target, human-review state, and ordered node traces. Evidence is accepted by Verifier only when its `report_id` equals the current request's report; this prevents evidence from another request or report from being reused accidentally.

## Node boundaries

- **Supervisor** returns an `AgentPlan`: route, allowed capabilities, and requested tool names. Unknown tools and unsupported routes stop before execution.
- **Research** accepts two async retrievers and starts them with `asyncio.gather`. Successful rankings are fused by RRF. One expected retriever failure becomes a traced degraded result; two failures become human review. Deployment may instead bind the existing combined hybrid retriever to the same typed contract.
- **Quant** asks the structured model only for a table-row selector and fixed-grammar FinQA program. It queries the report table, then calls the restricted Decimal executor. It never asks the model to perform the arithmetic. Unsupported or forward-referencing programs do not fall back to free-form calculation.
- **Report** invokes `compose_grounded_report` with evidence, calculations, and the exact allowed evidence IDs. The typed output rejects citations outside that set.
- **Verifier** invokes `verify_grounding`, rechecks current-report citation scope, compares numeric tokens against cited source text and Quant output, and records every verification issue.

## Bounded correction

`max_retries=1` is the default. On the first verification failure, `retry_count` becomes one and the graph chooses Research for citation problems, Quant for numerical/comparison problems, or Report for drafting problems. A second failure exceeds the budget and returns `needs_human_review=true` with a reason. Unexpected programmer exceptions are not relabeled as recoverable model errors; they propagate to the API boundary, where the SSE service emits a sanitized failure.

## Model and tool contracts

The local Transformers adapter adds the Pydantic schema to the prompt and validates returned JSON. The vLLM OpenAI-compatible adapter sends:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {"name": "AgentPlan", "schema": "<Pydantic JSON schema>"}
  }
}
```

Deployment starts the optional vLLM profile with `--enable-prefix-caching`. Prefix caching reduces repeated prompt-prefill work; it does not change output correctness, and this repository does not claim a measured latency improvement from it.

## HTTP and SSE

`POST /agent/ask` returns the final answer, citation IDs, human-review state, issues, and node traces. `POST /agent/stream` uses FastAPI `StreamingResponse` with `text/event-stream`.

Example event sequence:

```text
id: 0  event: run_started
id: 1  event: node_started
id: 2  event: node_finished
...
id: 9  event: answer_chunk
id: 10 event: run_finished
```

The service rewrites source sequence numbers monotonically, sends idle heartbeats as SSE comments, produces exactly one terminal `run_finished` or `run_failed` event, closes the agent iterator on disconnect, and removes exception/stack details from client-visible errors.

## Human review conditions

Human review is an explicit terminal state for unsupported plans, exhausted retrieval, non-retryable financial programs, missing drafts, or verification failure after the repair budget. It is not a silent generic answer and does not execute arbitrary Python.

