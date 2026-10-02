# The decomposition on the substrate that discriminates: the front end is free and the ranking is not

2026-10-02. Offline, deterministic, no judge, no API. LongMemEval_S
(`xiaowu0162/longmemeval-cleaned`), restricted to the three question types that carry a retrieval
deficit, BM25 and a pinned static encoder over session texts.

`docs/discriminating-mass-2026-10-02.md` found the substrate: under all-gold@5, 42.9% of
`multi-session` questions have a required session outside the top five, and that deficit is invisible
to the any-gold metric this project had been reporting. It found the benchmark and stopped there.

This runs this project's own decomposition on it -- a router miss is either the front end never
offering the evidence or the selection dropping it -- and pairs every arm against every other.

## The setup

**The unit is a session, not a window.** LongMemEval's label is `answer_session_ids`, a set of
session ids, so a unit maps one-to-one onto the label and no windowing choice can manufacture or hide
a miss. It also means the metric is not verbatim: a session counts as retrieved if any part of it is,
so the measurement does not reward a retriever for reproducing a string. That is the paraphrase-
tolerant reading `docs/termination-2026-10-02.md` asked for as its second open item.

The three types, because the others are at or above 0.98 under every metric and would only dilute the
table:

| type | questions | gold sessions each |
|---|---:|---:|
| `multi-session` | 133 | 2.59 |
| `temporal-reasoning` | 133 | 2.20 |
| `single-session-preference` | 30 | 1.00 |
| **total** | **296** | |

Haystack 38 to 61 sessions, median 47. Retrieval depth 5. `all@5` means every gold session is in the
top five; `coverage@5` is the mean fraction of gold sessions retrieved.

The candidate budget is swept, because the SCALE-QA result was that the front-end/selection split is
a property of the budget rather than of the corpus. Four configurations:

| policy | `top_per_retriever` | `max_candidates` |
|---|---:|---:|
| `default` | 10 | 20 |
| `tight` | 5 | 5 |
| `wide` | 10 | 1,000 |
| `everything` | 1,000 | 1,000 |

## Where the loss is: ranking, and the front end is nearly free

| policy | recovered | selection loss | candidate miss | sel % | miss % | mean candidates | oracle all-gold@5 | router | headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | 200 | 75 | 21 | 25.3% | 7.1% | 15.3 | **0.922** | 0.676 | **0.247** |
| `wide` | 200 | 75 | 21 | 25.3% | 7.1% | 15.3 | 0.922 | 0.676 | 0.247 |
| `everything` | 213 | 83 | 0 | 28.0% | 0.0% | 47.3 | **0.990** | 0.720 | **0.270** |

The `oracle` column is the ceiling over that policy's own candidate list: covering the gold set needs
`|gold|` slots, so a question is coverable at depth 5 exactly when it has at most five gold sessions
and the candidate list already holds all of them.

**The front end costs 0.068.** Handing the router the entire haystack as its candidate list --
`candidate_miss` zero by construction -- raises the ceiling from 0.922 to 0.990. Everything the
front end drops is worth six points of a hundred.

**The ranking leaves 0.247.** Against its own candidate list the router reaches 0.676 of a 0.922
ceiling: 73.3% claimed, a quarter of the coverable questions missed. Under `everything` it claims
72.7% of a 0.990 ceiling. So the stage that fails on the one benchmark whose hard types carry a
retrieval deficit is the ordering stage, and it fails by a wide margin.

This is SCALE-QA's decomposition arriving at the same place from a different direction and on a
different label representation. There, the candidate list's own ceiling was 0.982 and the best
ordering reached 0.589; here 0.990 and 0.720.

**`wide` is identical to `default`, and that is a finding.** Raising `max_candidates` from 20 to
1,000 changes nothing, because the mean candidate list is 15.3 either way: the binding constant is
`top_per_retriever=10`, not the cap. `docs/candidate-budget-2026-10-02.md` attributed the miss rate
to "`top_per_retriever=10` and `max_candidates=20`"; on this substrate only the first of those is
doing anything.

## The arms, paired

Means over 296 questions are not results until the pairing is used, so every comparison below is
paired on the same questions, with a bootstrap interval and an exact McNemar p. A difference is
called established only when both agree, which is this repo's `is_established` rule.

| arm | any@5 | all@5 | coverage@5 |
|---|---:|---:|---:|
| `dense` | 0.878 | 0.618 | 0.756 |
| `lexical` (plain BM25) | 0.912 | 0.642 | 0.787 |
| `router:default` | 0.919 | 0.676 | 0.814 |
| `router:wide` | 0.919 | 0.676 | 0.814 |
| `hybrid_full` | 0.929 | 0.686 | 0.822 |
| `hybrid` | 0.919 | 0.693 | 0.817 |
| `router:tight` | 0.919 | 0.693 | 0.817 |
| `router:everything` | 0.929 | **0.720** | **0.838** |

Paired, on `all@5`:

| comparison | difference | won | lost | exact p | 95% interval | verdict |
|---|---:|---:|---:|---:|---|---|
| `router:everything - lexical` | **+0.078** | 31 | 8 | **<0.001** | [+0.037, +0.118] | **established** |
| `hybrid - lexical` | **+0.051** | 25 | 10 | **0.017** | [+0.014, +0.091] | **established** |
| `router:everything - hybrid_full` | +0.034 | 16 | 6 | 0.052 | [+0.003, +0.064] | not established |
| `router:default - lexical` | +0.034 | 23 | 13 | 0.132 | [-0.007, +0.074] | not established |
| `router:everything - hybrid` | +0.027 | 14 | 6 | 0.115 | [-0.003, +0.057] | not established |
| `hybrid_full - lexical` | +0.044 | 30 | 17 | 0.079 | [+0.000, +0.091] | not established |
| `router:default - hybrid` | -0.017 | 15 | 20 | 0.500 | [-0.057, +0.024] | not established |
| `dense - lexical` | -0.024 | 37 | 44 | 0.505 | [-0.084, +0.037] | not established |

On the more sensitive `coverage@5`, `hybrid_full - lexical` (+0.034, 51 against 24, p = 0.002) and
`router:default - lexical` (+0.027, 39 against 21, p = 0.027) do reach established.

### What the pairing says

**The router does beat plain BM25 here, by an established margin, and that has not happened before in
this project.** `router:everything - lexical` is +0.078 at p < 0.001; on `multi-session` alone it is
**+0.135, 21 won against 3 lost, p < 0.001**, and even the shipped `default` configuration is
**+0.090, 16 against 4, p = 0.012** there.

**But the gain belongs to the fusion, not to the router.** `hybrid - lexical` is +0.051 and
established, and `router:everything - hybrid_full` is +0.034 at p = 0.052 -- the interval excludes
zero and the exact test does not, so by this repo's own rule it is a direction and not a result. The
one comparison where the router's own machinery does clear the bar is `router:everything -
hybrid_full` on `temporal-reasoning` (+0.060, 9 against 1, p = 0.021), which is one type out of three.

**The shipped configuration still does not beat plain fusion.** `router:default - hybrid` is
**-0.017**, 15 won against 20 lost. That is the same verdict SCALE-QA gave, on a substrate where the
fusion itself is working.

### A consistency check that passed

`router:tight` and the `hybrid` arm return **the same five sessions on all 296 questions**: difference
+0.000, 0 won, 0 lost, on `all@5`, `any@5` and `coverage@5` alike. That is expected and is worth
stating. At `top_per_retriever=5, max_candidates=5` the router takes each channel's top five, scores
them by RRF, truncates to five, and returns all five; a set metric cannot see the reordering by
calibrated probability that happens in between. So `router:tight` is not an independent arm -- it is
`hybrid` under another name -- and its row is a check that the harness reproduces the router's own
fusion exactly, not a measurement of the router.

## The sign of the fusion effect flips between benchmarks

| benchmark | comparison | difference | won | lost | p | verdict |
|---|---|---:|---:|---:|---:|---|
| SCALE-QA, 602 discriminating | `rrf_score - lexical_rank` | **-0.085** | 43 | 94 | <0.001 | fusion **hurts** |
| LongMemEval_S, 296 hard | `hybrid - lexical` | **+0.051** | 25 | 10 | 0.017 | fusion **helps** |

Both are depth-5 evidence recall on the questions only one of the two arms gets right. Both are
established. They point opposite ways.

The difference between the two setups is what the label is. SCALE-QA's evidence is a string that must
appear literally in a window, so BM25 is scoring against its own representation and a second channel
can only dilute it. LongMemEval's evidence is a session id, so BM25 is being asked a question it was
not constructed to answer and a semantic channel adds something. This project has been quoting
"the fusion costs the ordering" as a finding since `docs/fusion-weights-2026-10-02.md`; the honest
version is that it is a finding **about verbatim labels**, and the sign reverses when the label stops
being a string.

## What this does to the termination statement

The statement's reasoning for why routing cannot help was that retrieval is very nearly solved, so
the loss must be downstream. The first half of that is false on this substrate, and
`docs/discriminating-mass-2026-10-02.md` withdrew it. What replaces it is a sharper and less
comfortable position:

> On the benchmark whose hard types carry a retrieval deficit, the deficit is **inside** the retrieval
> stage. The front end costs 0.068 of the ceiling and the ordering stage leaves 0.247. The router
> captures 73% of the ceiling its own candidate list offers, and does not beat an equal-weight RRF
> fusion of the same two channels.

So the answer to "is the loss in retrieval or downstream of it" is: **in retrieval, on the only
substrate where the question can be asked with any power.** Whether the *answer* improves is still
unmeasured, and `docs/quality-verdict-2026-10-01.md` says why that instrument is not available here.

## What this does and does not establish

**Does not establish** that `router:everything` is a shippable configuration. Giving the router the
whole haystack as its candidate list is a diagnostic that isolates the ranking stage; it is not a
policy anyone would deploy, and nothing here says it should be.

**Does not establish** that the router adds nothing. It establishes that the additions over plain
fusion are not resolvable at n = 296 on a pooled test, and that on `multi-session` specifically the
fusion is what carries the gain. A larger sample could resolve `router:everything - hybrid_full`,
which sits at p = 0.052.

**Does not establish** any answer-quality effect. This is evidence recall. The distinction is the one
the termination statement spent its length on, and it is not resolved here.

**Establishes** that on a third-party benchmark with session-level, non-verbatim evidence, the
failure to retrieve is dominated by the ordering stage rather than by the front end, by a margin of
roughly four to one in ceiling terms; and that the sign of the fusion effect depends on whether the
evidence label is a string.

```bash
# the decomposition and the arms
HF_ENDPOINT=https://hf-mirror.com python .scratch/longmemeval_router.py

# the paired tests, read off the saved per-question records
python .scratch/longmemeval_paired.py
```

*Grade: exact, deterministic, offline; no model call beyond the pinned static encoder, and no judge.
Unit is a session, so the metric is set membership over `answer_session_ids` and not string
containment. The router runs this project's own `ContextRouter` unmodified; the `hybrid` and
`hybrid_full` arms are RRF with `k=60` and equal weights over BM25 and the static encoder, which then
matched `router:tight` on all 296 questions, confirming the two are the same construction. Paired
tests resample the 296 questions with a seeded RNG and call a difference established only when the
bootstrap interval excludes zero and the exact McNemar p is below 0.05, the rule
`scale_qa.is_established` uses. `.scratch/` is local to this machine and is not part of the published
tree.*
