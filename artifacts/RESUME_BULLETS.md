# Verified Resume Bullets

- Built and evaluated BM25, dense, hybrid, and cross-encoder-reranked retrieval on 1147 held-out FinQA queries; improved Recall@5 from 72.09% to 81.46% (+9.37 pp) and MRR@5 from 0.622 to 0.746.
- Integrated a restricted Python executor for financial calculations; in a controlled 100-query arithmetic-stage evaluation using identical gold evidence and programs, reduced error rate from 52.00% to 0.00%, a 100.00% relative reduction.
- Served the pipeline through FastAPI with validated JSON requests, structured answers, generated-program visibility, verifiable source citations, latency reporting, and human-review fallback for rejected programs.
