# context-router-lab

**Relation-aware, calibrated and provenance-preserving context routing for interleaved agent conversations.**

A research harness that asks one falsifiable question:

> In a long agent conversation that interleaves several unrelated tasks, can routing the
> query to the right logical contexts cut the tokens the main model reads, without
> losing answer quality?

The project does not try to be another memory OS or vector-RAG framework. It exists to
measure whether **selective context assembly** is worth building at all, and only then to
build it.

---

## The problem

Long conversations are not one linear task. They look like this:

```text
A1  A2  B1  A3  C1  B2  A4  B+C  A5
```

A conventional agent puts all of that into one conversation context, which produces
context pollution, wasted tokens, attention dilution across tasks, compaction that mixes
task states, and poor recovery when the user returns to an older task.

This harness turns that linear log into **multiple logical contexts plus a dynamically
selected working context**, so the main model sees only the minimal sufficient evidence
for the current turn.

## Architecture

```text
Append-only Event Store          (SQLite is the single source of truth)
        |
Causal Context Projection        (only events with sequence <= as_of)
        |
Relation Lite + Candidate Generation    (dense + BM25 + entity, fused with RRF)
        |
Calibrated Context Ranker        (grouped logistic regression + Platt calibration)
        |
Soft Selection / Abstention      (t_high, margin, t_low, correctness calibrator)
        |
Within-context Evidence Retrieval
        |
Token-budgeted Context Assembly  (provenance spans, tool atomic groups)
        |
Main Model  ->  Answer + Routing Trace + Evaluation Record
```

Two deep modules are the whole public surface. Embedding, BM25, the relation model, the
ranker, storage and provider calls all hide behind them:

```python
route(request: RouteRequest) -> RouteDecision

assemble_context(request: AssemblyRequest, decision: RouteDecision) -> WorkingContext
```

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows
# .venv/bin/python -m pip install -e ".[dev]"       # POSIX
```

Python 3.11–3.13, Pydantic 2, SQLite (FTS5 not required), jieba, scikit-learn, Typer.

## Quickstart

```bash
ctxlab generate-synthetic run/dataset --sessions 60
ctxlab ingest            run/events.sqlite run/dataset/records.jsonl
ctxlab validate-dataset  run/events.sqlite run/dataset/benchmark.jsonl
ctxlab compare-baselines run/events.sqlite run/dataset/benchmark.jsonl run/arms.json
ctxlab report            run/arms.json run/arms_report.md
```

`generate-synthetic --sessions 60` produces 60 sessions, 1 020 events and 600 labelled
query checkpoints. Everything above runs offline and deterministically — no API key, no
network.

Routing-only commands against a real pinned model live separately:

```bash
ctxlab build-index run/events.sqlite run/index.json
ctxlab route       run/events.sqlite syn-000 "回到之前的 Context Router，RRF 怎么算？"
ctxlab assemble    run/events.sqlite syn-000 "回到之前的 Context Router，RRF 怎么算？"
ctxlab benchmark   run/events.sqlite run/dataset/benchmark.jsonl run/bench.json
ctxlab train-ranker run/bench.json run/ranker.json
ctxlab tune-policy  run/bench.json run/policy.json
```

## Baseline arms

`compare-baselines` scores every memory strategy on the same token counter, the same
causal cutoff and the same labelled evidence sets, so the columns are directly
comparable.

| Arm | Mean memory tokens | Median | vs Full history | Evidence recall | Future leakage |
|---|---:|---:|---:|---:|---:|
| `query_recent_only` | 59.4 | 58.5 | 73.6% | 0.300 | 0 |
| `full_history` | 225.4 | 221.5 | 0.0% | 1.000 | 0 |
| `sliding_window` | 154.2 | 154.5 | 31.6% | 0.700 | 0 |
| `summary_recent` | 161.4 | 160.5 | 28.4% | 0.300 | 0 |
| `global_bm25` | 182.1 | 188.5 | 19.2% | 1.000 | 0 |
| `global_dense` | 208.5 | 202.5 | 7.5% | 1.000 | 0 |
| `global_hybrid` | 209.7 | 202.5 | 7.0% | 1.000 | 0 |
| `hybrid_router` | 212.8 | 255.5 | 5.6% | 1.000 | 0 |
| `oracle_router` | 128.8 | 131.5 | **42.9%** | 1.000 | 0 |

Measured on the 600-checkpoint synthetic dataset at a 2 048-token budget.

### Gate one: is selective context worth a router?

| Criterion | Result |
|---|---|
| Oracle token reduction vs full history | **42.9%** (target >= 30%) |
| Oracle evidence recall vs full history | not worse (1.000 vs 1.000) |
| Future-event leakage, all arms | 0 |
| Answer quality | **not yet measured** |
| Verdict | **continue** |

The gate is deliberately answerable offline. Its answer-quality criterion needs a run
against a pinned main model and is reported as unverified rather than assumed.

**Read the table before trusting it.** The honest summary of this first slice:

- The oracle proves the architecture has headroom — 42.9% of memory tokens can go
  without losing a single labelled evidence set.
- The learned router does **not** yet capture that headroom: it reaches full-history
  recall but saves only 5.6%, because it selects nearly every context and the 2 048-token
  budget never binds on 17-event sessions.
- `global_bm25` is competitive with the router while touching no context structure at all.
  That is the strongest signal in the table, and it says the routing problem here is
  currently too easy.

## Scope of Phase 0–1

Implemented: append-only SQLite event store with causal replay and lossless JSONL
round-trip, bilingual and code-aware lexical analysis, BM25 and hashing-vector retrieval
fused by RRF, rule-based relation classification, a pluggable Platt-calibrated ranker,
soft-routing policy tuning, a token-budgeted context builder with provenance spans and
tool-call atomic groups, the nine comparison arms, leakage-aware metrics, and a CLI.

Deliberately **not** implemented yet, because gate one only licenses them if it passes:
memory writer, automatic context creation, merge/split, context hierarchy or graph, and
any Codex/MCP integration.

## Honest limitations

1. **Answer quality is unmeasured.** Every number here is routing or token accounting.
   The claim "quality does not drop" is untested until a pinned model runs.
2. **The default embedder is not semantic.** `HashEmbeddingProvider` is a deterministic
   offline placeholder that hashes tokens into a 256-dimension vector. It makes
   `global_dense` and `global_hybrid` *reproducible smoke baselines*, not real dense
   retrievers. Wire `OpenAICompatibleEmbeddingProvider` before drawing conclusions about
   dense or hybrid retrieval.
3. **The synthetic dataset is too easy.** Three contexts per session, short sessions, a
   budget that never binds. The plan calls for 3–6 contexts and much longer sessions;
   until the generator is widened, the router has little room to lose and the comparison
   understates the difficulty.
4. **Reference solutions share the model.** The oracle uses the same assembly path as the
   router, so it bounds *implementation* error, not *modelling* error.

## Evaluation metrics

Router: exact-context-set accuracy, micro/macro precision, recall and F1, MRR, NDCG,
cross-context recall, relation accuracy, Brier score, ECE, high-confidence error rate and
risk–coverage curves.

Retrieval and builder: acceptable-evidence-set recall, future-leakage count (must be
zero), provenance completeness, memory tokens, total input tokens and latency.

Statistics: paired bootstrap over conversations for confidence intervals.

## Development

```bash
.venv/Scripts/python -m pytest         # 36 tests
.venv/Scripts/python -m ruff check .   # lint
.venv/Scripts/python -m ruff format .
.venv/Scripts/python -m mypy src       # strict
```

`mypy` type-checks at the development interpreter's version. The 3.11 runtime floor is
enforced by ruff's `target-version`, because numpy's bundled stubs use PEP 695 syntax that
mypy cannot parse while targeting 3.11.

## Related work

Useful prior art, and what to take from each: [TRACE](https://github.com/husain34/TRACE)
(topic-branch two-level retrieval), [Mem0](https://github.com/mem0ai/mem0) (hybrid
semantic + BM25 + entity recall), [Graphiti](https://github.com/getzep/graphiti)
(incremental events with valid-time and transaction-time), [Letta](https://github.com/letta-ai/letta)
(layered memory blocks and context compilation), [RASPUTIN Memory](https://github.com/jcartu/rasputin-memory)
(SQLite FTS5 + vector + RRF engineering reference), [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
and [LoCoMo](https://github.com/snap-research/LoCoMo) (long-term memory benchmarks).

The differentiating claim: existing systems mostly retrieve content inside a flat memory
space. This one first recovers *which task context the turn belongs to and how it relates
to the previous turn*, then retrieves the minimal sufficient evidence inside it.

## License

Apache-2.0. See [LICENSE](LICENSE).

---

## 中文摘要

面向**交错任务对话**的关系感知上下文路由研究框架。要验证的命题是：在答案质量不下降的
前提下，把 query 路由到正确的逻辑 Context 能否减少主模型读到的无关 token。

`compare-baselines` 在统一的 token 计数器、统一的因果截断和统一的标注证据集下横向对比
九个策略（离线、确定性、不需要 API key）。当前 600 个检查点的结果：**Oracle 可省
42.9% memory token 且证据召回不降**（第一道门槛通过，说明架构有上限），但**学习型
Router 只省 5.6%**，因为合成数据集太简单（每会话仅 3 个 Context、预算从不触及）。
答案质量尚未测量——必须先固定主模型才能下结论。
