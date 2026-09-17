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

Claude Code transcripts can be imported into a separate local event store without
committing the source conversations:

```powershell
ctxlab ingest-claude run/claude-history.sqlite "$env:USERPROFILE\.claude"
```

The importer streams visible user/assistant messages, tool calls and tool results; preserves
tool lineage and source metadata; replaces binary images with metadata-only placeholders;
and deliberately omits hidden thinking blocks. Event IDs are stable, so repeating the command
only imports newly appended records. Generated SQLite databases are ignored by Git.

## Baseline arms

`compare-baselines` scores every memory strategy on the same token counter, the same
causal cutoff and the same labelled evidence sets, so the columns are directly
comparable.

| Arm | Mean memory tokens | Median | vs Full history | Evidence recall | Future leakage |
|---|---:|---:|---:|---:|---:|
| `query_recent_only` | 135.5 | 138.0 | 95.4% | 0.462 | 0 |
| `full_history` | 2 976.9 | 2 928.0 | 0.0% | 1.000 | 0 |
| `sliding_window` | 336.7 | 335.0 | 88.7% | 0.538 | 0 |
| `summary_recent` | 221.3 | 236.0 | 92.6% | 0.462 | 0 |
| `global_bm25` | 749.7 | 775.0 | 74.8% | 1.000 | 0 |
| `global_dense` | 779.5 | 810.0 | 73.8% | 0.949 | 0 |
| `global_hybrid` | 779.9 | 804.0 | 73.8% | 0.974 | 0 |
| `hybrid_router` | 714.3 | 587.0 | **76.0%** | 1.000 | 0 |
| `oracle_router` | 440.2 | 439.5 | **85.2%** | 1.000 | 0 |

Measured on the 780-checkpoint synthetic dataset at a 2 048-token budget, with the
router on its default configuration.

### Gate one: is selective context worth a router?

| Criterion | Result |
|---|---|
| Oracle token reduction vs full history | **85.2%** (target >= 30%) |
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
- **The oracle proves the architecture, and by a wide margin.** 85.2% of memory tokens
  can go while keeping every evidence set. So selective context assembly is worth
  building — gate one passes on substance, not on a technicality.
- **The router now beats the strongest baseline on both axes.** 714.3 memory tokens at
  evidence recall 1.000, against `global_bm25` at 749.7 for the same recall. That is the
  project's claim — same quality, fewer tokens — measured against the baseline that
  previously beat it.

A third signal sits in the dense rows: `global_dense` (0.949) and `global_hybrid`
(0.974) still land *below* `global_bm25` (1.000), so fusing the embedding in still hurts.
That is measured evidence that the default hashing embedder carries no semantics — see
limitation 2 below — and it means the dense and hybrid arms should not be read as
dense-retrieval results at all.

### Out-of-sample: the claim, on held-out sessions

With a ranker trained on sessions 0–35 and a policy tuned on sessions 36–47, scored on the
disjoint sessions 48–59:

| Arm on the held-out split | Evidence recall | Mean memory tokens | vs `global_bm25` |
|---|---:|---:|---:|
| `hybrid_router`, default | 1.000 | 714 | −4.8% |
| `hybrid_router`, trained + tuned | 1.000 | **642** | **−14.4%** |
| `global_bm25` | 1.000 | 750 | — |
| `oracle_router` | 1.000 | 440 | −41.3% |

Full evidence recall at **14.4% fewer memory tokens** than the strongest baseline, on
sessions the router was never trained or tuned on. The oracle's 440 says roughly another
third is still on the table, so this is a first result rather than a ceiling.

Two caveats on how this was obtained. The minimal-sufficiency floor (`0.5`) was chosen by
sweep on sessions 0–29 and then **held on the disjoint 48–59 range, where it reproduced** —
a setting that had been fitted to the test split would not survive that. And the floor uses
only the system's own relevance scores, never the labels, so it is a standard relevance
cutoff rather than a label-fitted oracle.

Gate two is **not** passed. On the dev split the tuned policy reaches a high-confidence
error rate of 0.013 (target ≤ 0.02) and cross-context recall of 0.958 (target ≥ 0.90), but
required-context recall of 0.840 against a target of 0.95.

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

Fixing 1–2 took the router from 0.731 to 0.831; fixing 3–4 took it to 0.977.

Chasing the last 2.7% found two more, and the second was **not in the system at all**:

5. **The Builder spent the budget in global evidence order.** Admitting groups by score
   let one context's long tail exhaust the tokens before another *selected* context
   contributed anything, so a correct multi-context decision could still lose the answer
   it routed for. Evidence groups are now round-robined across contexts, each context's
   best group first. This is what "minimal sufficient" has to mean once more than one
   context is in play, and it halved the residual (10 → 5 of 330).
6. **The label was unanswerable, not the system wrong.** The remaining 5 were all
   `cross_context`, and all failed for one reason: the query named its two contexts
   ("把 Context Router 的 RRF 思路用到 SQLite migration 上") while the labelled evidence
   was one specific episode of each — but every episode of a context shares the same
   anchor file and marker, so **nothing in the query distinguished the labelled episode
   from its siblings**. No retrieval system could have found it. The generator now names
   what it wants from each side. This took the residual to **0 of 330**.

7. **The Builder treated the token budget as a target.** With recall already perfect, token
   accounting showed where the memory actually went — 631 of 821 tokens to evidence, with
   the mean *minimal* prefix that still covers a labelled set costing 296. So **64% of
   everything spent was waste**, admitted only because tokens remained. Worse, the router
   was selecting three contexts in 85 checkpoints when **no checkpoint in the benchmark
   ever requires three**. Evidence groups now stop at minimal sufficiency: a group is
   dropped when it scores below half its own context's best. On the 780-checkpoint
   benchmark that took the router from 915 to 714 tokens with recall unchanged at 1.000 —
   which is what finally put it ahead of `global_bm25`, 714 against 750.

The general lesson, and the reason the harness now measures this: a benchmark whose budget
never binds and whose sessions are too short cannot tell a broken router from a working
one. **Defects 1–4 were invisible until the dataset was widened, 6 was invisible until the
evidence layer was instrumented, and 7 was invisible until the tokens were counted by
section rather than reported as one number.** A benchmark bug and a system bug look
identical from the score alone, so the diagnostic loop classifies every failure by the
layer that dropped the evidence instead of reporting a single score.

## Answer quality: exploratory run

Everything above measures **evidence recall**, which is a proxy for answer quality and not
a substitute for it. `ctxlab answer-experiment` and `ctxlab judge-answers` now measure the
real thing: a main model answers each arm's memory, and a blinded judge compares the
answers pairwise in both orders.

The answer run (60 answer calls) and six judge iterations established the **direction** and
nothing more:

- **Evidence recall is a valid proxy here.** The arm with 0.55 recall scores far worse on
  both instruments, and the mechanism is refusal — it declines 60% of the time rather than
  fabricating an answer from parametric knowledge.
- **Router and oracle remain indistinguishable on answer quality**, matching their
  identical evidence recall, so the oracle's remaining 2× memory advantage buys no
  measurable quality on this sample.

It did **not** establish magnitudes, and a free control shows why to be careful with the
headline. `ctxlab judge-answers --judge coverage` runs the identical swap protocol with a
deterministic lexical scorer and no model: it produces the **same 10–2–8 tally**, but agrees
with the paid judge on only **14 of 20 pairs**. The matching total is a coincidence of
symmetric disagreement, not agreement — comparing methods on the win table alone would have
concluded the paid judge confirms the free heuristic.

The six judge iterations exposed a limit rather than a stable quality estimate. Anchoring the
rubric raised order agreement from 80% to 100%, but the judge still got the one manually
adjudicable pair wrong; bounding its analysis cut output tokens 64% and eliminated cap hits,
but agreement fell to 85%. Prompting the judge not to reward an echoed refusal did not work.

The structural fix is now implemented: `ctxlab judge-answers` wraps either judge in a narrow
refusal gate, so if **both** answers explicitly decline it records a tie without making either
position-swapped model call. A single refusal still goes to the delegate because `is_refusal`
is a lexical heuristic. Results now include separate checkpoint strata (`answerable` versus
`must_refuse`) and response strata (`neither`, `one`, or `both` refusing). Use
`--no-refusal-gate` only for an ablation. The original answer artefact and private credentials
were not retained, so this implementation has offline regression coverage but no claimed
seventh live judge run.

**These numbers are illustrative and not reproducible.** The model used is a private proxy
alias, not a documented identifier. Read
[`docs/answer-quality-experiment.md`](docs/answer-quality-experiment.md) for the full
report, including the three instrument failures that made the first two runs unusable —
the lexical metric scoring fluent refusals as correct answers, the judge discarding the
replies it needed for diagnosis, and a thinking block consuming the output budget so the
verdict was never emitted.

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

1. **The magnitude of answer quality is still unmeasured.** A small private-model run
   corroborates the direction of the evidence-recall result, but its six judge iterations
   never produced a simultaneously correct, stable and reproducible instrument. The model
   alias was private, n was 20 and the saved answer artefact was not retained. The claim
   "quality does not drop" therefore still needs a fresh run against a documented pinned
   model; the current result is evidence, not a final estimate.
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
4. **A clean diagnostic sample is not a clean system.** The diagnostic loop now reports
   0 lost evidence sets out of 330 answerable checkpoints, but that is a sample of a
   synthetic benchmark whose labels the generator itself writes. This benchmark has
   already shown it can hide several defects at once, so read a clean score as "nothing
   left to see here", not "correct". The remaining measured gap is the distance to the
   oracle: 642 tokens against 440 out of sample, which is the next thing worth chasing.
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
.venv/Scripts/python -m pytest
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
是真的，但从它得出的结论是错的。逐层诊断后定位到**七个**缺陷：**(1)** 关系规则太窄；
**(2)** `unknown` 被当成确信而收窄到单一上下文；**(3) 最大元凶——近期窗口被拼进检索
query**（正是计划警告的"上下文黏滞"）；**(4)** `relation_match` 反而奖励"正在离开的
上下文"；**(5)** Builder 按全局得分顺序花预算，一个上下文的长尾能挤掉另一个*已选中*
上下文的唯一证据；**(6)** 最后 5 例的真凶**竟是标签本身不可回答**（query 只写了两个
Context 的名字，而每个 episode 共用同一 anchor 与标记，**没有任何线索能区分被标注的
episode**，换任何系统都检索不到）；**(7)** Builder 把 token 预算当成了目标——按 section
记账后发现证据占 631/821 token，而覆盖标注集所需的**最小前缀只要 296**，即 **64% 是纯
浪费**；且 Router 在 85 个检查点选了 3 个 Context，而**全数据集没有任何检查点需要 3 个**。

七项修完后：**残差 0/330**，且 Router 在留出的测试会话上取得 **召回 1.000 / 642 token**，
**比 `global_bm25`（1.000 / 750）少 14.4% token**——**项目的命题「同等质量、更少 token」
首次成立**。Oracle 的 440 说明还有约三分之一空间。最小充分性的阈值 0.5 是在 0–29 会话上
扫出来的，**在完全不相交的 48–59 上复现**，且只用系统自身分数、不用标签。**第二道门槛
仍未通过**：dev 上高置信错误率 0.013 ✓、跨上下文召回 0.958 ✓，但 required-context 召回
0.840 < 0.95。

另外 `global_dense`(0.949) 与 `global_hybrid`(0.974) 仍**低于** `global_bm25`(1.000)，
这是默认哈希嵌入器不含语义的直接证据。答案质量尚未测量——必须先固定主模型才能下结论。
