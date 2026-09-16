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

`generate-synthetic --sessions 60` produces 60 sessions, 8 520 events, 360 contexts and
780 labelled query checkpoints — six interleaved contexts per session, ~142 events each.
Full history reaches ~4 400 tokens by the last checkpoint, which is what makes the
2 048-token budget bite. Everything above runs offline and deterministically — no API
key, no network. `compare-baselines` takes about 35 s for all 780 checkpoints.

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
| `query_recent_only` | 135.5 | 138.0 | 95.4% | 0.462 | 0 |
| `full_history` | 2 968.7 | 2 920.5 | 0.0% | 1.000 | 0 |
| `sliding_window` | 335.5 | 334.0 | 88.7% | 0.538 | 0 |
| `summary_recent` | 221.3 | 236.0 | 92.5% | 0.462 | 0 |
| `global_bm25` | 773.0 | 793.5 | 74.0% | 1.000 | 0 |
| `global_dense` | 789.7 | 821.5 | 73.4% | 0.923 | 0 |
| `global_hybrid` | 789.2 | 807.5 | 73.4% | 0.949 | 0 |
| `hybrid_router` | 912.9 | 779.5 | 69.2% | 0.974 | 0 |
| `oracle_router` | 603.4 | 560.0 | **79.7%** | 1.000 | 0 |

Measured on the 780-checkpoint synthetic dataset at a 2 048-token budget, with the
router on its default configuration.

### Gate one: is selective context worth a router?

| Criterion | Result |
|---|---|
| Oracle token reduction vs full history | **79.7%** (target >= 30%) |
| Oracle evidence recall vs full history | not worse (1.000 vs 1.000) |
| Future-event leakage, all arms | 0 |
| Answer quality | **not yet measured** |
| Verdict | **continue** |

The gate is deliberately answerable offline. Its answer-quality criterion needs a run
against a pinned main model and is reported as unverified rather than assumed.

**Read the table before trusting it.** Two things in it matter more than the ranking:

- **Cheap is not the same as good.** The three cheapest arms save 89–95% of memory
  tokens and recover only 46–54% of the labelled evidence. `query_recent_only` at 95.4%
  savings is not a result, it is a failure that happens to be small. Any claim resting on
  token reduction alone is worthless here.
- **The oracle proves the architecture, and by a wide margin.** 79.7% of memory tokens
  can go while keeping every evidence set. So selective context assembly is worth
  building — gate one passes on substance, not on a technicality.

A third signal sits in the dense rows: `global_dense` (0.923) and `global_hybrid`
(0.949) land *below* `global_bm25` (1.000), so fusing the embedding in actively hurts.
That is measured evidence that the default hashing embedder carries no semantics — see
limitation 2 below — and it means the dense and hybrid arms should not be read as
dense-retrieval results at all.

### Out-of-sample: the configured router matches BM25

The default row above is not the router's ceiling. With a ranker trained on sessions
0–35 and a policy tuned on sessions 36–47, scored on the disjoint sessions 48–59:

| Arm on the held-out split | Evidence recall | Mean memory tokens |
|---|---:|---:|
| `hybrid_router`, default | 0.974 | 913 |
| `hybrid_router`, trained + tuned | **1.000** | **780** |
| `global_bm25` | 1.000 | 773 |
| `oracle_router` | 1.000 | 603 |

```bash
ctxlab generate-synthetic run/train --sessions 36 --session-start 0
ctxlab generate-synthetic run/dev   --sessions 12 --session-start 36
ctxlab generate-synthetic run/test  --sessions 12 --session-start 48
# ingest each, then benchmark/train-ranker on train, tune-policy on dev,
# and finally compare-baselines on test with --ranker-file and --policy-file
```

The router reaches full evidence recall at within 1% of BM25's token cost, and unlike
BM25 it also produces selected contexts, relation labels, an abstention decision and a
per-turn routing trace. On this dataset, that is the honest state of the claim: equal
recall and equal cost, plus structure — not a token saving.

### What the first measurement got wrong

The first version of this table reported the router at 0.744 recall, *dominated* by
BM25. That number was real but the conclusion drawn from it was not, and the four causes
are worth recording because three of them were latent defects the small dataset had
hidden:

1. **The relation rules were too narrow.** `cross_context` was matched by the literal
   `应用到`, so a natural paraphrase ("把 X 的 Y 思路用到 Z 上") fell through to
   `unknown`, and `切换到` had no rule at all.
2. **`unknown` collapsed to a single context.** The selection policy applied its
   confident single-context branch whenever the relation was not `cross_context` — which
   includes `unknown`. An admission of ignorance became the most aggressive possible
   narrowing, turning every classifier miss into silently discarded evidence.
3. **The recent window was concatenated into the retrieval query.** Three recent turns
   (~120 tokens) swamp a short query, so BM25 and the dense channel favoured the context
   being left. This was the largest single cause and it is exactly the context stickiness
   the plan warns about: recency belongs in reranking, never in candidate retrieval.
4. **`relation_match` rewarded the contexts a return was leaving.** It gave +0.8 to
   contexts in `recent_context_ids` for `switch_or_return`, but a return target is by
   construction not recently used, so the feature rewarded the answer's opposite.

Fixing 1–2 took the router from 0.731 to 0.831; fixing 3–4 took it to 0.977 in-sample and
1.000 out-of-sample. The general lesson, and the reason the harness now measures this: a
benchmark whose budget never binds and whose sessions are too short cannot tell a broken
router from a working one. **All four defects were invisible until the dataset was
widened.**

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
   retrievers — and the table above shows it directly, since both score below plain
   `global_bm25`. Wire `OpenAICompatibleEmbeddingProvider` before drawing any conclusion
   about dense or hybrid retrieval, or about the router's dense candidate generator.
3. **The relation classifier is keyword rules, and it says so.** `relation-rules-v2`
   covers explicit Chinese and English cues and lets everything else fall to `unknown`,
   which the policy now widens on. English cross-context phrasing without an explicit cue
   ("Can the WAL result be shown in the matplotlib figure?") is deliberately left
   `unknown` rather than guessed — that case is what the plan reserves a small model for,
   and `tests/test_relation.py` pins the behaviour so a later rule change cannot silently
   overreach.
4. **A residue of within-context retrieval failures remains.** After the routing fixes,
   the only failures left in the diagnostic loop are cases where the correct contexts are
   selected but the token budget or the within-context ranking drops the evidence
   (3 of 110 answerable checkpoints in the diagnostic sample). That layer has not been
   examined yet.
5. **The synthetic sessions are still templated.** Six interleaved contexts, evidence up
   to 17 events behind the checkpoint, tool call/result pairs, cross-context and
   unresolvable checkpoints are all covered. Not yet covered, from the plan's harder-case
   list: a conclusion later overturned by a newer turn, the same identifier meaning two
   different things in two contexts, and a wrong assistant conclusion that must not be
   trusted. Until those exist, the benchmark cannot separate "found the evidence" from
   "found the *current* evidence".
6. **Reference solutions share the model.** The oracle uses the same assembly path as the
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
九个策略（离线、确定性、不需要 API key）。当前 780 个检查点（60 段会话 × 13）的结果有
两条最要紧：

1. **省 token 不等于做对了。** 三个最便宜的臂省掉 89–95% 的 memory token，却只召回
   46–54% 的标注证据。只看 token 降幅的结论在这里没有价值。
2. **Oracle 以很大余量证明架构成立**：省 79.7% 且证据召回一条不丢，第一道门槛通过。

**Router 曾经被最朴素的 BM25 全面压过**（召回 0.744 vs 1.000，token 还更多）。那 0.744
是真的，但从它得出的结论是错的：诊断后定位到四个缺陷——关系规则太窄、`unknown` 被当成
确信而收窄到单一上下文、**近期窗口被拼进检索 query**（最大元凶，正是计划警告的"上下文
黏滞"）、以及 `relation_match` 反而奖励"正在离开的上下文"。修完后 Router 在**留出的
测试会话**上取得 **召回 1.000 / 780 token**，与 `global_bm25`（1.000 / 773）基本持平，
另附 BM25 没有的上下文选择、关系标签、弃答决策与路由轨迹。**这四个缺陷在数据集加宽前
全部不可见。**

另外 `global_dense`(0.923) 与 `global_hybrid`(0.949) **低于** `global_bm25`(1.000)，
这正是默认哈希嵌入器不含语义的直接证据：融合进去反而变差。答案质量尚未测量——必须
先固定主模型才能下结论。
