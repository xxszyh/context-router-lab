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
| 3 | 4 / 40 | 1 |
| 4 | 7 / 40 | **2** |

In runs 3 and 4 the unreadable replies were 8 000–9 000-character thinking transcripts that
never reached a verdict — the model thought past its 2 048-token budget. The retry recovered
most of them. The count rose in run 4 because the procedure below asks the judge to work
through the requirements one at a time, which is more thinking, and the budget did not grow
with it.

### Fixing the order agreement

80% is not a request to be more consistent. It means the decision had nothing to rest on but
position, so the instructions now give it something:

1. work through the requirements one at a time, deciding for each whether A satisfies it and
   whether B does, so the verdict derives from a content assessment;
2. an answer satisfies a requirement only if it is actually addressed — echoing the wording
   while saying the answer cannot be given does **not** count, closing the same loophole the
   lexical metric fell into;
3. `tie` is defined (both satisfy the same requirements, neither more correct) instead of
   being an undefined third option a judge picks when undecided — which is precisely when
   position decides.

The reply must end with a `WINNER: <a|b|tie>` line, and `parse_verdict` now takes the **last**
mention: with the verdict pinned to the end, an earlier mention is reasoning about it.

| | run 3 | run 4 |
|---|---:|---:|
| order agreement | 16 / 20 (80%) | **18 / 20 (90%)** |
| wins `hybrid` / `cheap` / ties | 10 / 2 / 8 | 9 / 2 / 9 |
| ties that were *agreed* | 4 | **7** |
| verdicts that changed at all | — | **1 of 20** |
| agreement with the free `CoverageJudge` | 14 / 20 | 13 / 20 |

**The improvement is substantive rather than a hedge.** Only one of twenty verdicts changed,
so the judge did not simply start calling more ties; three pairs that previously reached
their verdict *via* a swap disagreement now reach the same verdict stably, and the tied
outcome is unchanged. The consistency came from the mechanism, not from the answer.

One pair went the other way: `syn-000-q-03` was a stable `a` and is now an unstable `tie`.
Both answers declined there, so rewarding one of them — which run 3 did — was the
loophole instruction 2 closes. **The new behaviour is the more correct one even though it is
the less consistent one**, which is worth stating plainly rather than presenting the 90% as
an unqualified win.

The judge also moved slightly further from the free heuristic (70% → 65% agreement), which is
what exercising more independent judgement looks like.

**90% is still not good.** One verdict in ten still flips with the order the answers appear
in, and the no-verdict rate rose because the procedure is more thinking than the budget
allowed.

### Run 5: raising the budget reached 100%, and that is not the whole story

The budget went to 4 096 with `JudgeCall` now recording `stop_reason` and output tokens.

| | run 3 | run 4 | run 5 |
|---|---:|---:|---:|
| order agreement | 80% | 90% | **100% (20/20)** |
| final parse failures | 1 | 2 | **0** |
| wins `hybrid` / `cheap` / ties | 10 / 2 / 8 | 9 / 2 / 9 | 11 / 2 / 7 |
| judge output tokens | 30 576 | 47 220 | 45 830 |
| agreement with the free `CoverageJudge` | 70% | 65% | 65% |

Three things stop this being a clean win.

**The thinking expands to fill whatever budget it is given.** `stop_reason` shows 3 of 41
calls still hit the 4 096 cap, and one reply still needed a retry. Raising the budget again
is a treadmill rather than a fix: the model thinks until the cap, so the cap sets the
thinking and not the other way round. The next change has to **bound the analysis**, not
enlarge its allowance.

**It cost about 50% more.** Output tokens went from 30 576 in run 3 to 45 830 in run 5 for
the same twenty pairs. Perfect self-consistency was bought with more deliberation.

**And on the one pair that can be adjudicated by hand, the extra deliberation produced a
worse verdict.** Only two verdicts changed between runs 4 and 5, both from `tie` to `a`
(`syn-000-q-03`, `syn-000-q-05`). In `syn-000-q-03` **both answers declined to answer** — and
run 5 rewards the router for it, because its refusal echoed the requirement's terms. That is
precisely the loophole instruction 2 was written to close, and run 4 had it right. The
longer thinking talked the judge out of the correct verdict.

> **100% self-consistent is not 100% correct.** The consistency metric the fix was aimed at
> cannot see this, and reporting the 100% without it would be reporting the instrument's
> agreement with itself as if it were agreement with the truth.

Agreement with the free heuristic stayed at 65%, so run 5 did not simply converge on the
lexical scorer — the judge has not become a slower copy of a free heuristic. What it has
become is stable and, on the evidence available, no more right than run 4.

### Run 6: bounding the analysis fixed the cost and broke the consistency

The hypothesis from run 5 was that the extra deliberation had talked the judge out of the
correct verdict, so the instruction was changed to bound the analysis — one short line per
requirement, no restating, four lines total — while keeping the 4 096 budget.

| | run 3 | run 4 | run 5 | run 6 |
|---|---:|---:|---:|---:|
| order agreement | 80% | 90% | **100%** | **85%** |
| parse failures | 1 | 2 | 0 | 0 |
| calls hitting the cap | — | — | 3 | **0** |
| judge output tokens | 30 576 | 47 220 | 45 830 | **16 396** |
| `syn-000-q-03` (correct = `tie`) | `a` ✗ | `tie` ✓ | `a` ✗ | `a` ✗ |

**Two things it fixed, cleanly.** Output tokens fell **64%** and the cap hits went to zero:
bounding the analysis does control the thinking, which the budget never did.

**And one thing it broke.** Order agreement fell from 100% back to **85%**. The judge's
self-consistency evidently depends on being allowed to deliberate, so there is a real
tension between the cost and the consistency and neither setting is acceptable.

**The hypothesis itself was wrong.** `syn-000-q-03` is still `a`. Bounding the deliberation
did not restore the correct verdict, so run 5's long thinking was not the cause. Across four
runs that pair has been `a`, `tie`, `a`, `a` — **v4 is the outlier and `a` is the base rate**.
The judge gets this pair wrong by default.

That is the important result, and it is not about deliberation at all: **instruction 2 does
not work.** Telling the judge "repeating a requirement's wording while saying the answer
cannot be given does NOT satisfy it" does not stop it happening. The refusal loophole is a
*content* judgement that a prompt cannot reliably impose.

### What this means for the next step

Prompt engineering has reached its limit on this judge. The thing that demonstrably detects
the loophole is already in the repository: `is_refusal` found the 20–25% false-positive rate
in the lexical metric and it is what `strict_coverage` is built on. It should be applied as a
**hard gate on the judge's verdict** rather than asked for in prose — if both answers decline,
the pair is a tie whatever the judge concluded. That is the one mechanism here with a
measured track record.

A gate would also make the remaining prompt tuning safe to stop, since correctness would no
longer depend on the instruction being obeyed.

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

## 5. The control: does the paid judge add anything?

`CoverageJudge` is a deterministic stand-in that runs the identical swap protocol using
`deterministic_coverage` and no model. It runs for free:

```bash
ctxlab judge-answers run/answers.json run/judge_coverage.json \
  --judge coverage --arm-a hybrid_router --arm-b query_recent_only --limit 20
```

| judge | wins `hybrid` | wins `cheap` | ties | order agreement | parse failures |
|---|---:|---:|---:|---:|---:|
| LLM (`deepseek-v4.1-flash`) | 10 | 2 | 8 | 16 / 20 | 1 |
| `CoverageJudge` (free) | **10** | **2** | **8** | 20 / 20 | 0 |

**The headline tally is identical — and that is a coincidence, not agreement.** Comparing
pairs rather than totals:

- the two judges agree on **14 of 20 pairs** — **30% disagreement**;
- the disagreements run both ways and happen to cancel in the aggregate;
- **all six disagreements involve a refusal**, so the two instruments differ *only* on how
  to treat a declining answer.

This is the trap the control was for. Anyone comparing methods on the win table alone would
conclude the paid judge confirms the free heuristic. It does not: it disagrees on nearly a
third of the individual comparisons, and the matching total hides that. **An aggregate can
mask 30% disagreement.**

### A hypothesis this refuted

The obvious explanation was that the LLM judge's extra signal is refusal-awareness, which
`strict_coverage` already provides for free. So the control was run a third way — a judge
scoring pairs by refusal-aware strict coverage:

| comparator | agreement with the LLM judge |
|---|---:|
| raw `deterministic_coverage` | 14 / 20 (70%) |
| refusal-aware `strict_coverage` | **13 / 20 (65%)** |

Making the lexical metric refusal-aware did **not** close the gap; it moved slightly further
away. So the judge's disagreement with a lexical heuristic is **not** explained by refusals
after all, and the earlier framing of this section's hypothesis was wrong.

What the remaining disagreement *is* cannot be settled at n=20. In four of the six cases
both lexical scores are equal (0.00 vs 0.00 — the metric has no signal at all) while the
LLM judge still expressed a preference. That is either signal the metric lacks or the
judge's own 20% positional noise, **and 20 samples cannot tell those apart.**

### Verdict on the judge as configured

**It does not earn its cost on this benchmark.** It reproduces a free heuristic's headline,
disagrees with it on 30% of pairs without either being demonstrably right, carries 20%
positional disagreement with itself, and left one pair unjudged across three runs.

There is evidence it *can* add signal — `syn-000-q-08` is a pair where both arms declined,
the judge correctly called a tie, and the lexical metric scored the router a win because its
refusal echoed the requirement's terms. But a judge whose verdict changes with answer order
one time in five cannot be used to establish that. **Fix the order agreement first; without
it the sample size is irrelevant.**

## 6. Findings

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
- **Whether the paid judge contributes anything a free heuristic does not.** Section 5
  shows the two agree on totals and disagree on 30% of pairs, which is a stand-off, not a
  result.

---

## 7. Threats to validity

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

## 8. What to do next, in order

1. ~~Run `CoverageJudge` over the same pairs~~ — **done, section 5.** Result: identical
   totals, 30% pairwise disagreement, and it refuted the refusal hypothesis.
2. **Fix the judge's order agreement before anything else.** It is 80%, and a verdict that
   changes with answer order one time in five makes every downstream number unusable.
   Constrain the output format so the verdict must appear in a text block after the
   analysis, then re-measure the order agreement on the same 20 pairs.
3. **Report refusal checkpoints separately** instead of mixing them into the win table,
   since they are 5 of the 8 ties and all 6 judge disagreements.
4. ~~Bound the judge's analysis~~ — **done, run 6.** It fixed the cost (−64% output tokens,
   zero cap hits) and broke the order agreement (100% → 85%). Left in place, because
   truncation and cost are real defects and the consistency was not trustworthy at either
   setting, but recorded as a trade rather than an improvement.
5. **Gate the judge's verdict on the deterministic refusal check** instead of instructing for
   it. Run 6 showed the instruction does not work, and `is_refusal` is already the one
   mechanism here with a measured track record. This is the next change to make.
6. **Keep using `syn-000-q-03` as the correctness probe.** It has now been wrong in three of
   four runs and it is the only pair adjudicable from the answer text alone. Order agreement
   cannot substitute for it: run 5 scored 100% on consistency while getting this pair wrong.
7. **Only then scale the sample**, and only against a documented model.

---

## 9. 中文摘要

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
