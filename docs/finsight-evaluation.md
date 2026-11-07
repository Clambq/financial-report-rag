# FinSight Evaluation Contract

## Two different evidence levels

The committed fixture run is a deterministic, fault-injection **application test**. It verifies routing, typed failures, retry termination, citation accounting, atomic artifacts, and metric recomputation. It is **not a real-model tool-use accuracy** measurement. A real-model claim requires a compatible vLLM run in a new directory plus successful artifact verification.

The ten fixture scenarios are text, numerical, mixed, malformed arguments, unknown tool, missing evidence, unsupported operation, retriever degradation, verification retry, and retry exhaustion. Failures are deliberately present, so the aggregate is not intended to look like a production KPI.

## Saved denominators

Every prediction row stores its case ID, completion flag, every tool attempt with attempt number and error code, citation validity when applicable, numerical verification when applicable, error category, latency, and model-call count.

- **Task completion** denominator: all cases.
- **Tool success** denominator: all tool attempts, including retry failures and successes.
- **First-attempt success** denominator: first attempts only (`attempt == 1`).
- **After-retry success** denominator: retry attempts only (`attempt > 1`).
- **Citation denominator**: cases for which citation validity is applicable, not all cases.
- **Numerical denominator**: cases for which deterministic numerical verification is applicable, not all cases.
- P50/P95 use NumPy linear interpolation over per-case latency.

## Deterministic smoke result

The current ten-case fault-injection fixture has 14 successful calls over 22 all tool attempts: **63.64%**. First attempts are 13/20; retry attempts are 1/2. These numbers demonstrate that the aggregator retains failures and the correct denominators. They are not a real-model tool-use accuracy and must not replace a separately measured vLLM result.

Reproduce it with:

```powershell
.venv\Scripts\finqa-rag.exe evaluate-agent `
  --cases tests/fixtures/agent_eval_cases.json `
  --output artifacts/agent/final --model fake
.venv\Scripts\finqa-rag.exe verify-artifacts
```

The runner writes `metrics_summary.json`, `predictions.jsonl`, `experiment_manifest.json`, and `reproducibility.json` atomically. The verifier reloads prediction rows, recomputes every metric, checks case order against the manifest, and verifies a canonical SHA-256 that excludes wall-clock latency.

## Real-model gate

No compatible vLLM/GPU runtime was available in the final Windows environment. Therefore a real-model success rate is unmeasured. The intended run, after starting vLLM with prefix caching, is:

```bash
docker compose --profile vllm up --build
# Run the real-model evaluator into a NEW directory once the CLI gains/uses the
# selected vLLM runtime, then verify that directory before publishing results.
finqa-rag verify-artifacts --artifact-root artifacts
```

The current `evaluate-agent --model fake` command intentionally refuses to masquerade as a real model. Adding a vLLM evaluation adapter must preserve the same prediction schema and denominators.

## Existing verified upstream results

FinSight reuses two separately verified components: retrieval Recall@5 72.09% to 81.46% and MRR 0.622 to 0.746 over 1,147 held-out FinQA queries; and a controlled 100-query arithmetic-stage comparison that reduced error from 52% to 0% with the restricted executor. Neither result is an end-to-end multi-agent accuracy measurement.

