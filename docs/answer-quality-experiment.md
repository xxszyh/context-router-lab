# Answer-quality experiment — first run and its three instrument failures

Date: 2026-09-16. Model: `deepseek-v4.1-flash` (see *Reproducibility* — this identifier is a
private proxy alias and is **not** reproducible by anyone else).

This document records the first attempt to answer the question the routing benchmark only
*proxies*:

> Does an arm that recalls the right evidence actually produce better answers — or is
> evidence recall a metric that moves independently of what the model can do?

It is written as a report of what happened, including three instrument failures that made
the first two runs unusable. Those failures are the most transferable part: each one is a
case of measuring something other than what was claimed.

---

## 1. What was run

```bash
ctxlab generate-synthetic run/dataset --sessions 60
ctxlab ingest run/events.sqlite run/dataset/records.jsonl

# 20 checkpoints x 3 arms = 60 calls
ctxlab answer-experiment run/events.sqlite run/dataset/benchmark.jsonl run/answers.json \
  --limit 20 --model deepseek-v4.1-flash --max-tokens 800

# 20 pairs x 2 orders = 40 calls, run three times
ctxlab judge-answers run/answers.json run/judge.json \
  --arm-a hybrid_router --arm-b query_recent_only --limit 20 \
  --model deepseek-v4.1-flash --max-tokens 2048
```

The 20 checkpoints are sampled twice-stratified: across all seven query types, and spread
within each type so the sample spans sessions 0, 30 and 59 rather than three adjacent ones.
15 are answerable and 5 require a refusal.

Arms: `query_recent_only` (135 memory tokens), `hybrid_router` (668), `oracle_router` (338).
The first exists to be wrong; the third is the ceiling.

---

## 2. The answer run

| arm | evidence recall | coverage | strict coverage | refused | attempted | memory tokens (local) |
|---|---:|---:|---:|---:|---:|---:|
| `query_recent_only` | 0.550 | 0.400 | **0.250** | **0.600** | 0.667 | 135 |
| `hybrid_router` | 1.000 | 0.800 | **0.600** | 0.350 | 0.833 | 668 |
| `oracle_router` | 1.000 | 0.800 | 0.650 | 0.350 | 0.909 | 338 |

- **coverage** = deterministic lexical coverage of `answer_requirements`.
- **strict coverage** = coverage with a declining answer counted as zero on an answerable
  checkpoint.
- **attempted** = coverage among answers that neither declined nor were unanswerable.

### Failure 1 — the metric scored fluent refusals as correct answers

The first scoring pass gave the cheap arm 0.30 and the router 0.70. Spot-checking two
answers showed why that number was not trustworthy:

```
requirements : 覆盖 routing.py 的「RRF 融合」结论, 覆盖 migration.py 的「批处理提交」结论
router answer: 回答中没有说明 RRF 融合思路能否迁移到 SQLite migration 的批处理提交上。
               它只在 syn-000-router 的 routing.py 中提到 RRF 融合，还有 ... 但未给出依据。
scored       : 1.00
```

The answer **declined to answer while quoting the requirement's own terms**, and the
lexical check counted the quotation as coverage. On an answerable checkpoint that is a
false positive, and it ran at **20–25% across every arm** — roughly uniform, so it inflated
absolute numbers without reversing the ordering, but it made the metric unusable for
magnitudes.

`is_refusal` now detects the stance and records carry `strict_coverage` and `refused`
alongside `coverage`, so the reader can see which way the instrument leans instead of
having the two collapsed into one number. Re-scoring the **saved** answers (no new calls)
moved the cheap arm 0.30 → 0.25 and the router 0.70 → 0.60.

The cue list is a heuristic and errs in both directions: it will also flag a real answer
phrased as "无法确定 X，但 Y 是...". That residual error is the measured reason the judge
below is still required.

### What the answer run established

**The arm without the evidence mostly declines rather than guessing.** `query_recent_only`
refused 60% of the time. The model under test does not paper over missing memory with
parametric knowledge, so **evidence recall genuinely gates answerability** on this
benchmark. The worry that motivated this experiment — that a 135-token arm would score
~90% answer quality and invalidate the whole optimisation target — did not materialise.

**Router and oracle are indistinguishable** (strict 0.600 vs 0.650), consistent with both
recalling every evidence set. The oracle's remaining advantage is **tokens, not quality**.

---

## 3. The judge, and two more instrument failures

### Failure 2 — the first judge run threw away its own evidence

Run 1 reported **14 of 40 replies unparseable**. `LLMJudge` counted the failures and
discarded the text, so finding out what they said would have cost another 40 calls. That
is the same mistake as the earlier ones: the layer under suspicion left no evidence behind.
`replies` now keeps every raw reply and the CLI writes them into the result.

### Failure 3 — the thinking block ate the token budget

Run 2, with replies captured, showed **all 7 remaining failures were the empty string**.
Not malformed output: no output. Probing a live call explained it:

```
blocks = [('thinking', 595), ('text', 177)]
blocks = [('thinking', 886), ('text', 193)]
```

The model emits a thinking block on every call; thinking counts against the same output
budget; and `_extract_text` deliberately ignores thinking blocks. When the deliberation ran
long, the reply had no text block at all. This is the truncation trap one level up — a
large part of the output was invisible to the code reading it, and unbudgeted.

Fixed by recovering a reply with no text block from its thinking (verdict model only — the
answer provider must not, because a human must not be shown deliberation as an answer), by
retrying an unreadable reply once, and by raising the judge budget to 2048.

| judge run | unparseable replies | final parse failures |
|---|---:|---:|
| 1 | 14 / 40 | 14 |
| 2 | 7 / 40 | 7 |
| 3 | 4 / 40 | **1** |

In run 3 the four unreadable replies were ~8 500-character thinking transcripts that never
reached a verdict — the model thought past its budget. The retry recovered three of them.

### Judge result (run 3, stable with run 2)

| | count |
|---|---:|
| `hybrid_router` wins | **10** |
| `query_recent_only` wins | **2** |
| ties | 8 |
| order agreement | **16 / 20 (80%)** |
| judge tokens | 8 539 in / 30 576 out |

**Four of the eight ties are position-bias artefacts, not equivalence.** A verdict that
flips when the answers are exchanged is a verdict about the position, which is why every
pair is judged twice and a disagreement is recorded as `tie` with `agreement=False` rather
than averaged away.

**Five of the eight ties involve a refusal** (2 both refused, 3 one side refused). The judge
is largely saying "neither answer answered" — a correct verdict that says nothing about the
relative quality of the two memories.

---

## 4. Cross-validation

| instrument | `hybrid_router` | `query_recent_only` |
|---|---:|---:|
| blind pairwise judge | 10 wins / 2 losses | 2 / 10 |
| deterministic strict coverage | 0.600 | 0.250 |
| evidence recall | 1.000 | 0.550 |

The judge and the lexical metric agree on **14 of 20 pairs**, and **5 of the 6 disagreements
involve a refusal** — the one place both instruments are known to be weak, for opposite
reasons. `syn-000-q-08` is the clearest case: both arms declined, the judge correctly called
it a tie, and the lexical metric scored it `a` because the router's refusal echoed the
requirement's terms. **There the judge was right and the metric was wrong.**

So the two instruments corroborate each other on **direction**, which is the validation this
run was for. They do not corroborate magnitudes.

---

## 5. Findings

Established:

1. **Evidence recall is a directionally valid proxy for answer quality on this benchmark.**
   The arm with 0.55 recall scores far lower on both answer-quality instruments than the
   arms with 1.00, and the mechanism is refusal: it declines 60% of the time rather than
   fabricating.
2. **The router's memory is worth its tokens relative to the cheapest arm** (10–2 on the
   judge, 0.60–0.25 strict).
3. **Router and oracle remain indistinguishable on answer quality**, matching their
   identical evidence recall. The oracle's 2× memory advantage buys nothing measurable here.

Not established:

- Any trustworthy **magnitude** of answer-quality difference. The judge's order agreement
  is 80%, ~20% of its calls produced no verdict, and the tie bucket is dominated by refusals.
- **Anything about a documented model.** See below.
- The **cost axis**: the quality–token frontier is not computable from this run (see
  limitation 4).

---

## 6. Threats to validity

1. **The model is a private proxy alias.** `deepseek-v4.1-flash` is served through a
   third-party Anthropic-compatible endpoint. Nobody else can reproduce these numbers, the
   operator can change what the alias resolves to at will, and the request log cannot show
   what actually served them. **These results are illustrative, not publishable.** A
   documented model identifier is required before any of this becomes a claim.
2. **Generation and judging use the same model.** Self-preference is a known effect. Here
   both arms' answers come from the same model so the preference is applied symmetrically
   and should not favour one arm — but the judge is not independent, and that is a
   methodological weakness, not a detail.
3. **n = 20.** Seven-versus-twelve successes is a small sample. Scaling it before fixing the
   judge would only measure the noise more precisely.
4. **The provider reports inconsistent usage for byte-identical requests.** Eleven pairs of
   judge calls sent identical bytes (same context digest) and ten reported different input
   token counts — 399 vs 143, 522 vs 138, 803 vs 163 — and different output counts too. The
   proxy reports `cache_read_input_tokens: 0`, so this is **not** prompt caching. The cause
   is **not known**, so the cost axis in this report uses the deterministic local memory
   count and the provider's numbers are marked unusable rather than averaged in.
5. **The judge replies in the language of the corpus.** Requirement terms include ASCII
   identifiers, which makes the lexical check partly language-robust, but nothing here
   tests a monolingual English or Chinese conversation.

---

## 7. What to do next, in order

1. **Constrain the judge's output format** so it must emit its verdict in a text block
   after analysis. This targets both remaining failures directly: the 20% position
   disagreement and the no-verdict replies.
2. **Run `CoverageJudge` over the same pairs** — zero calls. If the paid judge's verdicts
   largely reproduce a lexical heuristic, it is not adding information and should be
   replaced rather than trusted.
3. **Report refusal checkpoints separately** instead of mixing them into the win table,
   since they are 5 of the 8 ties.
4. **Only then scale the sample**, and only against a documented model.

---

## 8. 中文摘要

这是**答案质量实验的第一次运行**，目的是验证：证据召回到底是不是答案质量的合格代理。

**结论（方向已确立）**：不合格的证据召回臂 **60% 的时候选择拒答而不是胡编**，两套独立
仪器（盲化 Judge 10:2、词法 strict coverage 0.60:0.25）都测到它明显更差。所以**证据召回
是合格的代理，项目方向成立**——"廉价臂几乎不降质从而推翻整个项目"的可能性没有发生。
Router 与 Oracle 在答案质量上**无法区分**，与两者证据召回都是 1.000 一致：**Oracle 的优势
在 token，不在质量**。

**三轮仪器故障（本报告最有价值的部分）**：
1. 词法指标**把"复述要求词然后拒答"判成满分**，假阳性率 20–25% → 引入 `is_refusal` 与
   `strict_coverage`，把"覆盖"和"拒答"分开报而不是合并成一个数。
2. 第一轮 Judge **把失败的原始回复丢掉了**，想查就得重买 40 次调用 → 改为保存全部原始回复。
3. 模型每次都先输出 **thinking 块，thinking 计入输出预算**，思考一长就**根本不产出 text 块**，
   回复为空 → 兜底从 thinking 捞回（仅判词模型；**答案 provider 刻意不这么做，因为人不能把
   思考过程当答案看**）+ 空回复重试 + 预算提到 2048。解析失败 14 → 7 → **1**。

**量级仍不可信**：Judge 顺序一致性仅 **80%**（8 个平局里 4 个是位置偏差产物），约 20% 调用
产不出判词，8 个平局里 5 个涉及拒答。**n = 20，且生成与评判是同一个模型（自偏好混淆）。**

**最重要的限制**：`deepseek-v4.1-flash` 是**私有代理别名**，他人不可复现、运营方可随时改
指向。**这批数字是示例性的，不是可发表的。** 另外代理对**逐字节相同的请求**回报不一致的
用量（399 vs 143），且 `cache_read_input_tokens: 0` 说明**不是缓存**——原因**未知**，故本
报告的成本轴一律使用确定性的本地 memory token。
