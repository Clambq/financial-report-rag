# Verified Results

## Retrieval on the locked FinQA test split

| Method | Recall@5 | MRR@5 | P95 latency |
|---|---:|---:|---:|
| BM25 | 72.09% | 0.622 | 0.36 ms |
| Hybrid + cross-encoder reranker | 81.46% | 0.746 | 15.18 ms |

The locked comparison contains 1,147 held-out queries. Development data selected the candidate count and fusion weight before the public test split was evaluated.

## Controlled numerical-execution experiment

| Condition | Exact match | Arithmetic error rate |
|---|---:|---:|
| Direct Qwen/Qwen3-4B-Instruct-2507 calculation | 48.00% | 52.00% |
| Safe Python execution | 100.00% | 0.00% |

This 100-query experiment deliberately gives both conditions the same gold evidence and gold symbolic program. It isolates arithmetic execution; it is not an end-to-end program-generation accuracy claim.

## FinSight deterministic application test

The committed `artifacts/agent/final` run covers ten deterministic success and fault-injection scenarios. It recorded 14 successful calls across 22 tool attempts (63.64%), 13/20 successful first attempts, and 1/2 successful retry attempts. Five of ten cases terminate with a final answer because the fixture deliberately includes malformed arguments, unknown tools, missing evidence, an unsupported operation, and retry exhaustion.

This is a contract and failure-handling test, not a real-model tool-use accuracy result. No compatible vLLM/GPU runtime was available for the final run, so no real-model success rate is published. `verify-artifacts` reconstructs the summary from `predictions.jsonl`, checks case order, and verifies the canonical prediction hash.

## Error analysis derived from saved predictions

- Hybrid + reranker retrieved all gold evidence for 800 queries, partial gold evidence for 267, and no gold evidence in the top five for 80.
- Relative to BM25, per-query recall improved on 219 queries, regressed on 76, and was unchanged on 852.
- Representative retrieval misses: `GS/2015/page_188.pdf-2`, `CMCSA/2015/page_112.pdf-2`, `ADBE/2008/page_89.pdf-2`, `GS/2015/page_188.pdf-1`, `CDNS/2018/page_66.pdf-4`.
- The 52 direct-Qwen mismatches include 19 percentage-unit outputs, 6 verbose or truncated responses, 1 Boolean mismatch, and 26 other numerical mismatches.
- Representative direct-Qwen mismatches: `MSI/2007/page_70.pdf-1`, `SLG/2001/page_48.pdf-2`, `GS/2014/page_51.pdf-1`, `MS/2013/page_132.pdf-1`, `GIS/2015/page_62.pdf-1`.
