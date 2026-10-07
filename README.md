# context-router-lab

**English** · [中文介绍](#中文介绍)

**Relation-aware, calibrated and provenance-preserving context routing for interleaved agent conversations.**

A research harness that asks one falsifiable question:

> In a long agent conversation that interleaves several unrelated tasks, can routing the
> query to the right logical contexts cut the tokens the main model reads, without
> losing answer quality?

The project does not try to be another memory OS or vector-RAG framework. It exists to
measure whether **selective context assembly** is worth building at all, and only then to
build it.

**Maintenance resumed, 2026-10-07.** The historical research verdict below is
preserved. The external LongMemEval experiment now has a portable, tested CLI,
paired all-gold metrics, interruption-safe checkpoints, and an optional pinned
joint query-passage reranker. See
[the maintenance plan and reproduction commands](docs/maintenance-2026-10-07.md).
Generated-answer quality remains unestablished; session retrieval is measured separately.

The first [full external reranking run](docs/longmemeval-joint-2026-10-07.md)
raises all-gold session recall from **68.3% to 81.7%** on 278 answerable questions,
with 44 paired wins against 7 losses. The scorer stays opt-in pending an
end-to-end answer-quality gate.

> **The project is closed as a router and kept as a record.** The token half is established and
> stronger than the claim; the quality half is a null result that the available judge could not have
> resolved. Read **[`docs/termination-2026-10-02.md`](docs/termination-2026-10-02.md)** first: it
> states what was established, every explanation that was proposed and refuted, the four findings
> that outlive the question, and what a future reader should do instead.
>
> Two measurements made after that statement was written, and folded back into it, are what a reader
> should take next.
> **[`docs/discriminating-mass-2026-10-02.md`](docs/discriminating-mass-2026-10-02.md)** withdraws the
> claim that retrieval is nearly solved -- that came from an any-gold reading of a benchmark whose
> multi-session questions need a *set* of sessions -- and shows SCALE-QA saturating at 20%
> discriminating, with recall monotone in string overlap.
> **[`docs/longmemeval-decomposition-2026-10-02.md`](docs/longmemeval-decomposition-2026-10-02.md)**
> then uses the substrate that does discriminate and finds the front end nearly free (0.068 of the
> ceiling) and the ordering stage leaving 0.247. The router **does** beat plain BM25 there by an
> established margin, but the gain belongs to the fusion and the shipped configuration does not beat
> it.
>
> To find out whether *your* benchmark can answer the question you are asking it:
> `ctxlab audit <package>` ([`docs/audit-2026-10-02.md`](docs/audit-2026-10-02.md)).

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

Optional neural retrieval and joint reranking:

~~~sh
python -m pip install -e ".[dev]"
ctxlab longmemeval datasets/fixtures/longmemeval-smoke.json .local/smoke.json
python -m pip install -e ".[dev,neural,rerank]"
ctxlab longmemeval /path/to/longmemeval_s_cleaned.json .local/lme.json --embedder neural
ctxlab longmemeval /path/to/longmemeval_s_cleaned.json .local/joint.json --embedder neural --rerank
~~~

Use --limit 30 for a first smoke test and --resume to continue an interrupted
run. The reports explicitly mark answer quality as not measured.
The included fixture is fictional and only tests the workflow; it is not
benchmark evidence.

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
ctxlab audit-secondary-annotations primary.json secondary.json events.sqlite audit.json
ctxlab audit-answer-adjudication primary.json secondary.json adjudication.json events.sqlite audit.json
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
| Answer quality | exploratory direction only; formal non-inferiority unverified |
| Verdict | **continue** |

The offline part of the gate passes. A small private-model run later supported the direction
of the evidence-recall result, but its model alias and judge were not reproducible, so formal
answer-quality non-inferiority remains unverified rather than assumed.

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

That 0.840 and the later 1.000 stage diagnostic measure different operating points. The
default policy keeps every required context but over-selects heavily; the tuned development
policy narrows the set and loses required contexts. The problem to solve is therefore the
recall–precision calibration trade-off, not an unexplained contradiction between runs.

```bash
ctxlab generate-synthetic run/train --sessions 36 --session-start 0
ctxlab generate-synthetic run/dev   --sessions 12 --session-start 36
ctxlab generate-synthetic run/test  --sessions 12 --session-start 48
# ingest each, then benchmark/train-ranker on train, tune-policy on dev,
# and finally compare-baselines on test with --ranker-file and --policy-file
```

The default router reaches full evidence recall at 4.8% below BM25's token cost; training and
policy tuning widen that held-out reduction to 14.4%. It also produces selected contexts,
relation labels, an abstention decision and a per-turn routing trace. These are synthetic
results, so the next milestone is real causal replay rather than another synthetic claim.

### Stage diagnosis: retrieval is not the present synthetic bottleneck

The new stage metrics separate a required context that never entered the candidate set from
one that entered and was later dropped. On the same 780 checkpoints, both candidate and
selected required-context recall are 1.000, with zero candidate misses and zero selection
losses. The problem is the other direction: selection precision is 0.591, 280 cases select
too broadly and the router includes 540 excess contexts. `new_context`, `unanswerable` and
`cross_context` account for most of them.

That result changes the implementation order. A Transformer is not justified here as a
synthetic recall fix; a pinned semantic embedding remains a conditional real-replay ablation.
The current priority is a sanitized, manually labelled replay of real Claude histories, then
calibration and narrower soft selection where those labels support it. See
[`docs/v0.2-real-replay.md`](docs/v0.2-real-replay.md).

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
`--no-refusal-gate` only for an ablation. Gated pairs are excluded from the judge's order-
agreement denominator, and the primary `tally` contains answerable checkpoints only;
must-refuse results are available only in their separate stratum. The original answer artefact
and private credentials were not retained, so this implementation has offline regression
coverage but no claimed seventh live judge run.

**These numbers are illustrative and not reproducible.** The model used is a private proxy
alias, not a documented identifier. Read
[`docs/answer-quality-experiment.md`](docs/answer-quality-experiment.md) for the full
report, including the three instrument failures that made the first two runs unusable —
the lexical metric scoring fluent refusals as correct answers, the judge discarding the
replies it needed for diagnosis, and a thinking block consuming the output budget so the
verdict was never emitted.

## Answer quality: the result, on a documented model

The fresh run this section used to ask for has now happened, on `kimi-k2.5`, over two real
conversations, with the judge asked three times per pair in both orders. **It is a null
result.**

| pooled, 35 decided pairs | `hybrid_router` | `query_recent_only` | ties | p |
|---|---:|---:|---:|---:|
| terse judge | 16 | 22 | 30 | 0.42 |
| **anchored judge** | **18** | **17** | 33 | **1.00** |

**A blinded judge cannot distinguish relevance-selection from recency-selection.** The 6–3
lead the project reported earlier came from a weaker instrument: 42% of its ties were the two
orders contradicting each other and the protocol collapsing that into a tie. Asking each pair
three times removed those ties and flipped no decided pair, and the lead did not survive.

Two things did survive, and neither depends on a judge:

- `full_history` **ran on 5 of 24 checkpoints** and failed the other 19 with HTTP 400 context
  refusals, against a documented model's own ceiling — 79%, where the generous 1M proxy alias
  had reported 10%;
- `hybrid_router` reads **1,671** memory tokens against `full_history`'s median **370,102**.

The judge itself was then fixed. Order disagreement turned out to be in how it *read* the
answers rather than where they sat — all thirteen self-contradicting pairs on the published
conversation contradicted themselves within a single order, with no position bias at all —
and the prompt forbade the one thing that would have grounded the reading. Requiring a quote
per requirement took `mean_consistency` from 0.797 to 0.891, left the tie count unchanged, and
moved no decided pair; independent hand reads agree with the anchored verdicts on 8 of 8 pairs
against 2 of 8 for the old prompt. `--judge-style terse` keeps the old prompt available, and
every judge artefact now records which one produced it. → `docs/judge-anchoring-2026-10-01.md`

## Scope of Phase 0–1

Implemented: append-only SQLite event store with causal replay and lossless JSONL
round-trip, bilingual and code-aware lexical analysis, BM25 and hashing-vector retrieval
fused by RRF, rule-based relation classification, a pluggable Platt-calibrated ranker,
soft-routing policy tuning, a token-budgeted context builder with provenance spans and
tool-call atomic groups, the nine comparison arms, leakage-aware stage metrics, local Claude
Code transcript import, and a CLI.

Deliberately **not** implemented yet, because gate one only licenses them if it passes:
memory writer, automatic context creation, merge/split, context hierarchy or graph, and
any Codex/MCP integration.

## Honest limitations

1. **Answer quality is a null result, and a null is not a proof of no difference.** The run
   against a documented pinned model has happened: two conversations, 35 decided pairs,
   p = 1.00 under the anchored judge. That bounds any quality advantage above roughly the
   effect this sample could have seen — it does not establish that the arms are equivalent,
   and it says nothing about the arms on a third conversation. The project's earlier
   direction-only reading, from a private alias with n = 20 and no retained artefact, is
   superseded rather than confirmed.
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

## 中文介绍

**面向交错任务对话的关系感知上下文路由研究框架。**

### 要验证的命题

> 在一段交错着多个无关任务的长对话里，把 query 路由到正确的逻辑上下文，**能否减少主模型
> 读到的 token，同时不损失答案质量？**

这个仓库存在的目的是**先测量"选择性上下文组装"值不值得做**，值得再去做——不是再写一个
记忆系统或向量 RAG 框架。

### 问题长什么样

长对话不是一条线性任务：

```text
A1  A2  B1  A3  C1  B2  A4  B+C  A5
```

常规 agent 把这一切塞进同一个上下文，于是产生上下文污染、token 浪费、注意力被稀释、
压缩时把任务状态搅在一起，以及用户回到旧任务时恢复不良。本框架把线性日志变成**多个逻辑
Context + 一个动态选出的工作上下文**，让主模型每轮只看到最小充分证据。

### 现在测到了什么

**一、省 token 这一半：成立，而且比原命题更强。**

真实对话上 `hybrid_router` 每轮读 **1,671** 个 memory token，`full_history` 中位数
**370,102**——**221 倍**。但真正的结论不是"更便宜"：换成**公开文档化模型**的真实上下文
上限（`kimi-k2.5` 260,096 / `qwen3-max` 258,048，都是网关自己拒绝时报出的数），
`full_history` 的中位输入是 **485,966**，**18 个检查点里 13 个超限**；在 24 检查点的完整
gate 上它**只跑通 5 个**，其余 19 个 HTTP 400（**79%**）。

**读全文不是"更贵"，而是在文档化模型的窗口下根本不是一个可选项。**

**二、质量这一半：零结果，而且仪器越好越平。**

盲判两两对比、交换顺序、每对问 3 次取多数：

| | `hybrid_router` | `query_recent_only` | 平局 | 已决 | p |
|---|---:|---:|---:|---:|---:|
| 合并（terse 判定器） | 16 | 22 | 30 | 38 | 0.42 |
| **合并（anchored 判定器）** | **18** | **17** | 33 | 35 | **1.00** |

**相关性选择与近因选择在答案质量上分不出来**：35 个已决对、p = 1.00。项目此前报过的
"6–3 领先路由"来自更弱的仪器——那个判定器 42% 的平局是它**自相矛盾**之后被协议折算成的
平局。

**三、合成数据上，门槛一通过、门槛二没过。** Oracle 路由省 **85.2%** 的 memory token 且
标注证据召回一条不丢，说明架构有余量；但校准（门二）的 required-context 召回
**0.840 < 0.95**。

**四、终止陈述发布后，又被项目自己的测量修正了两次——两次都记在原文旁边。**

- **"检索几乎已解决"这个结论站不住。** 它来自 **any-gold** 读法（"至少一个金标会话进了
  top-5"），而 LongMemEval_S 平均每题需要 **1.90** 个金标会话，`multi-session` 是 **2.59** 个。
  换成 **all-gold@5**（每个必需会话都要在前五）：**42.9% 的 `multi-session` 题**有必需会话不在
  前五，整体 22.6%。这是这个项目**第一次有了能真正区分"损失在检索还是在下游"的基底**。
- **在那个基底上：前端几乎免费，排序不是。** 把整个语料交给路由器当候选（于是 candidate miss
  构造上为 0），ceiling 只从 **0.922 升到 0.990**——前端值 0.068；而排序层只拿到 0.922 里的
  **0.676，丢掉 0.247**。路由器在那里**确实**以 established 的幅度赢过纯 BM25（**+0.078**，
  31 胜 8 负，p<0.001；单看 `multi-session` 是 **+0.135**），**但增益属于等权 RRF 融合**——
  `router:everything − hybrid_full` 是 +0.034、**p = 0.052**，区间排除 0 而精确检验没有，
  按项目自己的规则只能算方向；**出厂配置仍然打不过融合**（−0.017，15 胜 20 负）。
- 顺带修正：SCALE-QA 的判别质量**封顶在 20.1%**（十个包全扫过，填充轴完全平坦、对话数轴饱和），
  没有任何包越过项目自己声明的 25% 门槛；而 **融合效应的符号随标签反转**——在 SCALE-QA 的逐字
  字符串上是 **−0.085**，在会话 ID 上是 **+0.051**。

### 方法上值得单独说的一件事

这个项目**反复推翻自己**，并且每次都记下来：

- 第一版表格报 Router 召回 0.744、被最朴素的 BM25 全面压过。逐层诊断后定位到**七个**
  缺陷——最大元凶是**近期窗口被拼进了检索 query**，正是计划自己警告过的"上下文黏滞"。
  修完残差 **0/330**，在留出的测试会话上取得**召回 1.000 / 642 token**，比 `global_bm25`
  （1.000 / 750）**少 14.4% token**。这不是答案质量非劣效结论。
- 真实对话上的"路由质量 6–3 领先"，把判定器修好后变成 **7–5、p = 1.00**。
- 词法上下文标签那个 **F1 0.786 复现不出来**：它是三个未记录自由参数（粒度、渲染方式、
  阈值）的一个取值，其中**粒度一项**就能把它从 0.463 拉到 0.786。**永远不要在不写明读法
  规模的情况下引用一致率。**

判定器本身也修了一轮。顺序分歧被拆开后发现：published 会话上**全部 13 个自相矛盾对都是
在同一个顺序内矛盾**，位置偏置为 **0**——噪声在"怎么读"，不在"排在哪"。而 prompt 里那句
`Do not restate either answer` 恰好**禁止了唯一能锚定这个读法的东西**。改成"每条要求必须
引用决定它的原文"之后：`mean_consistency` **0.797 → 0.891**（published）、
**0.853 → 0.864**（海冰），平局数不变、已决对**翻转 0 对**，独立盲读 **8/8** 支持新判定器
而旧的只有 **2/8**。

### 明确的局限

1. **答案质量的幅度仍未测出。** 两个会话、35 个已决对，这是一个零结果，不是"证明无差异"。
2. **判定器仍不是好仪器。** `mean_consistency` 0.86–0.89，八次调用里有一次自相矛盾——这是
   质量结论的分辨率天花板。残余噪声在"读一条要求"，不在"判谁赢"：已用 29 次能读到汇总
   条数的调用证明，判定器的结论**0 次**违背它自己写下的条数。
3. **默认嵌入器不含语义。** `HashEmbeddingProvider` 是确定性的离线占位实现，所以
   `global_dense`(0.949) 与 `global_hybrid`(0.974) 是**可复现的冒烟基线**，不是真正的稠密
   检索结果——表里它们低于 `global_bm25`(1.000) 正是这一点的直接证据。
4. **合成会话仍是模板生成的。** 尚未覆盖：结论被更新的轮次推翻、同一标识符在两个上下文里
   含义不同、错误的助手结论不可信。在这些补上之前，benchmark 分不清"找到了证据"和
   "找到了**当前**证据"。

### 快速开始

全部离线、确定性、不需要 API key：

```bash
ctxlab generate-synthetic run/dataset --sessions 60
ctxlab ingest            run/events.sqlite run/dataset/records.jsonl
ctxlab compare-baselines run/events.sqlite run/dataset/benchmark.jsonl run/arms.json
```

780 个检查点全跑约 35 秒。详细命令见上文 Quickstart，架构见 Architecture。

### 许可

Apache-2.0，见 [LICENSE](LICENSE)。
