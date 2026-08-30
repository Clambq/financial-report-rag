# Verified Results

## Retrieval on the locked FinQA test split

| Method | Recall@5 | MRR | P95 latency |
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
