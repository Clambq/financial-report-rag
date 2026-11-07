# Financial Report RAG 面试指南

这份材料只解释仓库中已经实现和验证的内容。面试时不要把受控算术实验说成端到端准确率，也不要声称项目解析了任意原始 PDF。

## 30 秒介绍

I built a financial-report RAG system on FinQA that combines BM25, dense retrieval, reciprocal-rank fusion, and cross-encoder reranking. On 1,147 held-out test queries, it improved Recall@5 from 72.09% to 81.46% and MRR@5 from 0.622 to 0.746. For numerical questions, I added a restricted Python executor instead of letting the LLM calculate directly; in a controlled 100-query experiment using the same gold evidence and program, it reduced arithmetic error rate from 52% to 0%. I exposed the pipeline through FastAPI with structured responses, source citations, generated-program visibility, and human-review fallback.

## 两分钟展开

FinQA 的每条样本包含金融报告正文、表格、问题、支持证据和符号化计算程序。我先把正文段落和表格行转换成稳定的 evidence units，并限制在单份报告内检索。BM25 对精确指标、年份和专有词很强，dense retriever 对同义表达更强，所以我分别取候选，用 weighted reciprocal-rank fusion 合并名次，再用 cross-encoder 对 query–evidence pair 做精排。

为了防止数据泄漏，我只在 883 条 development queries 上选择 candidate count 和 fusion weight，冻结配置后才在 1,147 条 test queries 上跑两遍。最终 Recall@5 是 81.46%，比 BM25 的 72.09% 高 9.37 个百分点，MRR@5 从 0.622 提升到 0.746。

数值部分我把“能否找证据/生成程序”和“能否正确执行算术”拆开。固定 100 个 test IDs，并让 Qwen 和 Python 条件都拿到相同 gold evidence 与 gold symbolic program。Qwen 直接计算正确 48 条，restricted Decimal executor 正确 100 条，所以算术错误率从 52% 降到 0%。这个实验只证明 tool execution 的价值，不代表端到端 program generation 是 100%。

在线服务复用同一套 hybrid retrieval 和 reranker。文本问题由 Qwen 基于证据回答；数值问题先生成受限 FinQA program，再交给不使用 `eval` 的安全执行器。非法运算、前向引用、除零和非有限结果会被拒绝并标记人工复核。API 返回答案、生成程序、五条引用、错误信息和延迟。

## 每个简历数字来自哪里

| 简历数字 | 含义 | 权威文件 |
|---|---|---|
| 1,147 | 锁定 test retrieval queries | `artifacts/retrieval/metrics_summary.json` |
| 72.09% | BM25 Recall@5 | `artifacts/retrieval/metrics_summary.json` |
| 81.46% | Hybrid + reranker Recall@5 | `artifacts/retrieval/metrics_summary.json` |
| +9.37 pp | 81.46% − 72.09% | `artifacts/RESUME_BULLETS.md`，由 JSON 自动计算 |
| 0.622 → 0.746 | BM25 与最终方法的 MRR@5 | `artifacts/retrieval/metrics_summary.json` |
| 100 | 固定 numerical sample size | `artifacts/numerical/experiment_manifest.json` |
| 48% → 100% | Direct Qwen 与 Python exact match | `artifacts/numerical/metrics_summary.json` |
| 52% → 0% | Direct Qwen 与 Python arithmetic error rate | `artifacts/numerical/metrics_summary.json` |
| 100% relative reduction | `(0.52 - 0) / 0.52` | `artifacts/numerical/metrics_summary.json` |
| 8,281/8,281 | safe executor 对全部公开 gold programs 的复现结果 | `scripts/validate_official_programs.py` 的验证输出 |
| 15.18 ms | 最终检索 P95，不含模型加载 | `artifacts/retrieval/metrics_summary.json` |

`finqa-rag verify-artifacts` 会从 4,588 条 retrieval prediction rows 和 100 条 numerical prediction rows 重新计算这些指标，并校验两次复跑的 canonical SHA-256。

## 指标解释

- **Recall@5**：前五条结果覆盖了多少 gold evidence。一个问题可能有多个 gold evidence，所以它比简单的 hit 更严格。
- **Hit@5**：前五条里是否至少出现一条 gold evidence，只取 0 或 1。
- **MRR@5**：前五条结果里第一条相关证据名次的倒数。相关证据排第一得 1，排第二得 0.5；前五条没有相关证据时为 0。
- **NDCG@5**：考虑多个相关证据及其位置的折损累计增益，再用理想排序归一化。
- **Exact Match**：经过固定金融数字 normalization 后，预测与 gold 是否在 `0.0001` tolerance 内一致；yes/no 必须完全一致。
- **P50/P95 latency**：50%/95% 请求不超过的延迟。模型下载和一次性加载不算在检索延迟中。
- **Percentage points**：两个百分率直接相减，例如 81.46% − 72.09% = 9.37 pp。
- **Relative error reduction**：`(baseline error − tool error) / baseline error`。

## 常见项目问题与回答

### 1. Why not use vector search only?

金融问题常包含精确年份、指标名称、缩写和金额。BM25 对这些 lexical matches 很稳；dense retrieval 能找语义改写。测试集上 dense Recall@5 是 78.43%，hybrid 是 78.87%，说明两者有互补，但增益不应该被夸大。

### 2. What is reciprocal-rank fusion?

RRF 根据每个候选在不同排序中的名次加权：`score(d)=Σ weight_i/(k+rank_i(d))`。它不直接混合 BM25 和 cosine 的原始分数，因此避免了尺度不一致问题。本项目 `rrf_k=60`。

### 3. Where is the reranker placed?

先由 BM25 和 dense 各自召回候选，RRF 合并成 10 条，再让 cross-encoder 对 query 和每条 evidence 联合编码并重排，最终返回 top 5。这样把昂贵的 cross-encoder 限制在小候选集。

### 4. Why did reranking help?

Bi-encoder 在构建 evidence embedding 时看不到 query，速度快但交互有限。Cross-encoder 同时看到 query 与 evidence，能判断更细的匹配关系。测试 Recall@5 从 hybrid 的 78.87% 提升到 81.46%。

### 5. How did you choose candidate_k and fusion weights?

只在 development split 比较 `candidate_k={10,20}` 和 BM25 weight `{0.5,1,2}`；先按 Recall@5，再按 MRR@5，再按较低 P95 选。最终是 candidate 10、BM25 0.5、dense 1.0。

### 6. Did you tune on the test set?

没有。所有超参数在 883 条 dev queries 上冻结，然后才评估 1,147 条 test queries。两次 test run 只用于复现，不用于再调参。

### 7. How did you define chunks?

FinQA 已提供结构化正文和表格。本项目把每个正文段落和每个表格行作为 evidence unit，而不是固定 token window。这样引用能直接落到文本段或表格行，也避免切断一行财务数据。

### 8. Why can larger chunks increase recall but reduce precision?

大 chunk 更可能包含答案词，因此可能增加召回；但也混入更多无关内容，使 embedding 和 reranker 难以定位真正证据，并增加生成上下文噪音。本项目没有声称完成 arbitrary chunk-size study，因为使用的是结构化 evidence units。

### 9. How do you distinguish retrieval failure from generation failure?

先检查 top-k 是否包含 gold evidence。没有就是 retrieval miss；有 gold evidence 但答案错误，才属于 downstream generation/program/execution。检索 artifact 保存每条 ranked IDs 和 gold IDs，可以直接分类。

### 10. Why use a calculator tool for financial reasoning?

LLM 的 token generation 不是精确算术引擎。固定实验中，Qwen 即使拿到 gold program 仍有百分比单位、长输出截断和算术错误；受限 Decimal executor 对相同 100 条达到 100/100。

### 11. Is the 100% result end-to-end accuracy?

不是。它是 arithmetic-stage execution accuracy。两种条件都拿到 gold evidence 和 gold program，所以没有测 program generation。面试时必须主动说明这个边界。

### 12. Why give Qwen the gold program?

这是为了做 controlled experiment：唯一变化是“由模型直接计算”还是“由 Python 执行”。如果同时让模型自己找证据和生成程序，错误来源混在一起，就无法把下降归因于 calculator。

### 13. How is the executor safe?

它只解析固定 operation grammar，不用 `eval`。参数只能是数字、允许的 const token、表格行名或 backward `#` reference。未知运算、恶意 Python、前向引用、除零、NaN/Infinity 都抛出 `ProgramError`。

### 14. Why use Decimal instead of float?

Decimal 让十进制财务计算和 tolerance 行为更可控，也避免常见二进制浮点展示误差。执行上下文 precision 设为 40，最终比较用显式 `0.0001` tolerance。

### 15. How are percentages handled?

在 program operand 里，`23.6%` 按 FinQA 官方语义解析为 0.236；在最终显示答案 normalization 中，`20%` 保留显示数值 20，而不是静默除以 100。两种语境必须分开。

### 16. How did you validate compatibility with FinQA?

我扫描所有公开 program，发现最初常量列表不完整，还遇到重复表格行名和数字形式行名。加入真实 regression fixtures 后，安全执行器与 8,281/8,281 gold programs 一致。

### 17. What happens when a generated program is invalid?

API 不做自由形式计算回退。它返回空答案、原始 generated program、引用、结构化错误，并把 `needs_human_review` 设为 true。

### 18. Can a valid program still be wrong?

可以。语法安全不等于语义正确。真实 smoke 中最初模型曾生成合法但错误的判断程序。后来通过更明确的 operation definitions 和 few-shot 改善，但端到端 program accuracy 仍未正式评测，README 将其列为限制。

### 19. What does the FastAPI layer add?

它提供 request validation、统一 response schema、health/metrics endpoints、citations、generated-program visibility、latency 和 human-review fallback。它不是只把 notebook 包一层，而是复用同一检索和执行模块。

### 20. What is returned as a citation?

每条 citation 包含 evidence ID、text/table 类型、原始 source text、reranker score 和 rank。用户可以看到答案依据的具体表格行或文本段。

### 21. How reproducible are the results?

FinQA commit、模型 ID、Python 版本、参数、sample IDs 和预测都保存。两次 retrieval 的 4,588 条 method-level rows 在 rankings/scores/metrics 上一致；两次数值实验在 IDs/raw outputs/answers/correctness 上一致。校验器从 prediction JSONL 重算汇总。

### 22. Why not parse raw PDFs?

当前目标是可解释地比较 retrieval 和 numerical execution。原始 PDF 会引入 OCR、layout、merged cells 和 reading order 等独立问题，使误差来源混杂。后续可以增加 Docling/Unstructured 或自定义 layout parser，并单独建立 extraction benchmark。

### 23. Why was QLoRA deferred?

这个项目的研究问题是 hybrid retrieval 和 tool execution，不是领域微调。如果加入 QLoRA，就需要另一套 base-vs-fine-tuned held-out evaluation、训练数据污染检查和 calibration。它更适合独立项目，不能为了“技术多”硬塞进来。

### 24. Why not report RAGAS faithfulness?

本次没有完成足够可靠的 generated-answer annotation 或 RAGAS benchmark，所以不发布 faithfulness claim。现有结论只覆盖 evidence retrieval 和受控算术。宁可少报一个指标，也不报无法复现的漂亮数字。

### 25. What would you improve next?

第一，建立 end-to-end program-generation test 并加入 constrained decoding；第二，对 table-aware embedding 和 hard-negative reranker 微调做消融；第三，解析真实 PDF 并单独测 extraction recall；第四，增加 confidence calibration、review threshold、persistent vector index 和并发压测。

### 26. What surprised you most?

Reranker 确实提高总体 Recall，但也让 76 条 query 相对 BM25 退步，说明平均指标会隐藏局部回归。另一个发现是 Qwen 有时算对比例却输出百分数单位；产品中答案单位 normalization 和可见程序同样重要。

## 保存预测中的代表性案例

### 检索成功

1. `INTC/2015/page_41.pdf-4`：两个 gold table rows 在 rank 1 和 2。
2. `ADI/2011/page_61.pdf-2`：gold narrative evidence 排名第一。
3. `FIS/2010/page_70.pdf-2`：两个 gold table rows 都在 top 5。
4. `MAS/2017/page_27.pdf-2`：跨两行表格的股东回报问题，gold rows 排名 3 和 4。
5. `SYY/2006/page_71.pdf-1`：租赁费用变化的 gold text 排名第一。

### 检索失败

1. `GS/2015/page_188.pdf-2`：gold 是 `table_3/table_7`，top 5 全被相似 narrative text 占据。
2. `CMCSA/2015/page_112.pdf-2`：需要 `table_8`，只召回相邻的 `table_5`。
3. `ADBE/2008/page_89.pdf-2`：需要 `table_6`，相似税务正文和其他表行排名更高。
4. `GS/2015/page_188.pdf-1`：一个 gold row 到第六名之外，top 5 证据不足。
5. `CDNS/2018/page_66.pdf-4`：需要组合 `table_1/table_6`，复杂长问题导致错误表行进入 top 5。

### Direct Qwen 数值失败

1. `MSI/2007/page_70.pdf-1`：模型输出解释并在 64 tokens 截断，parsed answer 变成 `$5,038`，gold 为 `6055.66667`。
2. `SLG/2001/page_48.pdf-2`：多步平均计算输出在中途截断，parsed answer 为 `2`。
3. `ETR/2011/page_17.pdf-3`：program 输出应为比率 `-0.0291`，模型返回显示百分数 `-2.91%`。
4. `GIS/2015/page_62.pdf-1`：`divide(-36.5, 7.7)` 的 gold 为 `-4.74026`，模型错误输出 `5.00`。
5. `EOG/2018/page_75.pdf-1`：表格平均问题输出长解释并截断，parsed answer 为 `2`，gold 为 `338928.0`。

所有 52 条 direct-Qwen mismatch 都保存在 `artifacts/numerical/predictions.jsonl`，没有删除失败样本。

## 面试中的表述底线

- 可以说：built、implemented、evaluated、improved、deployed locally、reproduced。
- 不要说：production system、private FinQA leaderboard、100% end-to-end accuracy、arbitrary PDF support、QLoRA fine-tuning、RAGAS faithfulness，除非后续真的补做并产生新 artifact。
- 如果被问到某个细节不知道，先说明实验边界，再回到仓库里可验证的实现和数字；不要编造线上用户、业务收益或公司部署。

## FinSight Multi-Agent 面试补充

### 30 秒版本

我把原来的金融报告 RAG 和安全数值执行器扩展成了一个 LangGraph 多智能体应用。Supervisor 只负责结构化路由；Research 做检索并支持 BM25 与 dense 并发和降级；Quant 只生成受限 FinQA program，再交给 Decimal 执行器；Report 只能引用当前请求返回的 evidence ID；Verifier 再检查引用范围和回答数字。验证失败最多允许一次修复，第二次转人工，避免 Agent 无限循环。服务提供 `/agent/ask` 和 SSE `/agent/stream`，并有 Docker 和可选 vLLM prefix-cache 部署路径。

### 为什么要五个节点，而不是一个大 Prompt？

因为每个错误的责任边界不同：找不到证据回 Research，数字不一致回 Quant，表达或引用格式问题回 Report。拆开后每个节点只有一个结构化输出，工具名由注册表白名单控制，Verifier 还能决定修复目标。一个大 Prompt 很难区分错误来源，也很难证明最多重试几次。

### 为什么 Supervisor 不直接执行工具？

Supervisor 只产出 `AgentPlan`。代码在任何调用之前检查 `tool_names` 是否存在于 `ToolRegistry.names`；未知工具直接记录 `unknown_tool` 并终止。因此模型的文本不能变成任意函数调用。

### Research 的并发和降级怎么做？

两个 async 检索器用 `asyncio.gather(..., return_exceptions=True)` 同时启动。两边成功就做 RRF；一边出现预期的 timeout/connection/validation 错误时保留另一边结果并写 `retriever_degraded`；两边都失败才转人工。非预期程序异常继续抛出，不会伪装成可重试网络问题。

### Verifier 到底验证什么？

第一层是作用域：只把 `report_id` 等于当前请求的 evidence ID 放入 allowlist，跨报告或编造 ID 在 Pydantic 工具入参处就失败。第二层是数字：从被引用原文和 Quant calculation 取可信数字，与回答里的数字集合比较。第三层是重试：第一次失败按 issue code 回 Research、Quant 或 Report；第二次失败超过 `max_retries=1`，转人工。

### 简历中的 97% 怎么回答？

仓库已经实现了工具调用记录、首试/重试分母和可重算评估，但 **97% 尚未复现**。当前提交的 10 条是故障注入 smoke，故意包含失败，不能拿它替代真实模型准确率。面试时应该说：这个数字来自当时实习/原设计口径，现仓库尚缺兼容 vLLM 的真实模型复跑；如果对方继续追问，就展示 `predictions.jsonl` 的分母定义，不要把 smoke 指标说成 97%。

### SSE 如何保证可调试？

事件经过服务层重新编号，ID 严格递增；心跳是注释，不污染数据事件；正常和异常都只有一个 terminal event；客户端断开时会取消 pending `__anext__` 并 `aclose` 上游迭代器；异常只返回 `internal agent error`，不会把堆栈泄露给客户端。

### vLLM 的结构化输出和 prefix cache 分别解决什么？

结构化输出通过 OpenAI-compatible `response_format=json_schema` 约束 Supervisor、Quant 等节点的 JSON，再由 Pydantic二次校验。prefix caching 通过 `--enable-prefix-caching` 复用重复系统提示和 schema 的 prefill KV；它是性能优化，不提升准确率，本机没有 vLLM 环境，所以仓库没有发布它带来的延迟改善数字。

### 当前实现最诚实的边界是什么？

LangGraph 编排、类型契约、引用/数字验证、一次修复上限、JSON/SSE API、故障注入评估和部署文件都已有测试。真实 vLLM tool-use accuracy、Docker 实机启动、生产并发压测和任意 PDF 解析尚未验证；这些不能说成已经上线。
