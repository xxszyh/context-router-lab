# The corpus-size effect on an independent benchmark, and where it lives

2026-10-02. Offline, deterministic, no judge, no API. LongMemEval_S
(`xiaowu0162/longmemeval-cleaned`), 500 questions, BM25 over session texts.

`docs/composition-sweep-2026-10-02.md` established on SCALE-QA that the front-end/selection split
tracks the number of evidence-bearing sessions and not the amount of filler.
`docs/candidate-budget-2026-10-02.md` then showed the corpus axis only matters because the retrieval
depth is fixed, which is the mechanism rather than a repudiation. Both were measured on one
benchmark, built one way, and the project's own synthetic generator turned out to be useless for it
(see below).

This is the same question asked of a benchmark this project did not build and did not choose.

## Why LongMemEval is a real test rather than a repeat

* **There is no filler in it at all.** Every competing session is a real user-assistant
  conversation. The mechanism says the distractors that matter are the ones that look like the
  answer; LongMemEval's haystack is nothing but those, so if the effect needs filler to appear it
  will not appear here.
* **It has per-question ground truth** (`answer_session_ids`) naming the session the answer lives
  in, so a hit is checkable by string search.
* **It comes from a different lineage.** It is the benchmark Engram reports on, and its sessions are
  chat logs rather than a flat stream of documents.

Structure: 500 questions; haystack sessions per question from **38 to 62, median 48**; 1.90 gold
sessions per question on average; six question types (`multi-session` and `temporal-reasoning` 133
each, `knowledge-update` 78, `single-session-user` 70, `single-session-assistant` 56,
`single-session-preference` 30).

## The design

The corpus is varied **under control and paired**: every gold session is kept, and the remaining
haystack is sampled down to N sessions with a seeded RNG. The same question is therefore measured at
every size, and only the number of competitors changes. This is SCALE-QA's corpus axis, run on data
nobody built for it.

Retrieval is a plain BM25 over concatenated session text, which is the cheapest real retriever and
the same baseline the rest of this project uses. The metric is whether any gold session is in the
top k.

| sessions in haystack | n | hit@1 | hit@5 | hit@10 |
|---:|---:|---:|---:|---:|
| 5 | 500 | 0.920 | **1.000** | 1.000 |
| 10 | 500 | 0.890 | 0.990 | 1.000 |
| 20 | 500 | 0.844 | 0.974 | 0.996 |
| 40 | 496 | 0.810 | **0.950** | 0.984 |

**Monotone at every depth, and stable across three sampling seeds** (hit@1 at N=40: 0.815, 0.815,
0.810; hit@5 at N=40: 0.950, 0.954, 0.950). Growing the corpus eightfold costs **0.107 at depth 1**,
0.049 at depth 5 and 0.016 at depth 10.

The median rank of a gold session is **1** at every size, at every seed. The effect is entirely in
the tail: most LongMemEval questions are found immediately, and what the corpus size moves is the
share of hard ones.

For completeness, the same measurement on each question's *natural* haystack size (41 to 55
sessions, buckets of 6 to 59 questions) shows hit@5 between 0.86 and 0.98 with no readable slope.
The natural range is too narrow and too confounded with question type to carry the claim; the paired
truncation is what does.

## The second thing this measurement says

> **Withdrawn 2026-10-02, later the same day.** The conclusion below was drawn from an **any-gold**
> metric -- one gold session retrieved is a hit -- and LongMemEval_S averages **1.90 gold sessions
> per question**, 2.59 for `multi-session`. Under **all-gold@5**, which is what a question needing
> three sessions actually asks for, **42.9% of `multi-session` and 28.6% of `temporal-reasoning`
> questions have a required session outside the top five**, and 22.6% of the benchmark does. The
> numbers below are unchanged and reproduce exactly; what is withdrawn is the conclusion drawn from
> them. See `docs/discriminating-mass-2026-10-02.md`.
>
> The measurement error is exactly the one the grade note at the end of this document already
> flagged -- "a question with 1.90 gold sessions is scored the same as one with a single gold
> session" -- and the consequence was not followed through.

**Retrieval is very nearly solved on LongMemEval_S too.** A plain BM25 over sessions puts a gold
session in the top five for **95% to 100%** of questions, at every corpus size tested.

The published best result on that benchmark -- Engram, at 83.6% answer accuracy against 73.2% for
full context -- is an *end-to-end* number. So on a benchmark this project did not build, with its own
retrieval protocol and its own readers, the evidence is in the retrieved context for almost every
question and the reported accuracy is fourteen points below that. ~~**The loss is downstream of
retrieval there as well.**~~ That is the same shape as this project's own finding, arrived at
independently, and it is the single most useful thing in this document: the "does routing find the
evidence" question is close to the wrong question on more than one benchmark.

*The struck sentence is withdrawn, for the reason in the block above. The remainder of the section
stands as a description of what an any-gold measurement shows.*

## Where the effect lives, stated as a domain

Three benchmarks, three behaviours, and the differences are informative about the domain of the
claim rather than about its truth:

| benchmark | retrieval units competing | candidate/front-end behaviour |
|---|---|---|
| this project's synthetic generator | **2 to 6** logical contexts | **saturated at zero**: candidate miss 0.000 and selection loss 0.000 in every configuration, on both the corpus and the filler axis |
| LongMemEval_S | 38 to 62 sessions | measurable, but only at **depth 1**; depth 5 is 95-100% at every size |
| SCALE-QA, 1,000-3,000 dialogues | 1,175 to 3,561 windows | strong: candidate miss 0.131 to 0.246 as the corpus triples |

So the claim has a domain and it should be quoted with one: **the front-end/selection split is a
function of how many plausible competitors a fixed retrieval depth has to cover, and it is only
observable when that number is large relative to the depth.** Ten competitors against a depth of
five is not a regime where it shows. Two thousand against five is.

The synthetic row is worth its own sentence, because it is the project's own benchmark and it is
**useless for this question**: with 2 to 6 contexts the router gets every required context right in
every configuration, so there is nothing to decompose. Its corpus axis moved events from 1,260 to
5,640 with the catalog fixed at ~5.8 and the miss rate at zero throughout. A benchmark with no
failures cannot report a failure rate, however many configurations are swept.

## What this does and does not establish

**Establishes:** the corpus-size effect is real on an independent third-party benchmark, under a
paired controlled design, with no filler anywhere in the corpus. Direction and rough size agree with
SCALE-QA (roughly a tenth of a point at depth 1 for an eightfold corpus increase).

**Does not establish:** the mechanism beyond "a fixed retrieval depth against more competitors
covers less". The controlled axis here changes the number of sessions and nothing else, so it cannot
separate "more near-misses" from "more documents"; those are the same thing in a corpus where every
document is a near-miss.

**Does not establish:** that LongMemEval_S is a good benchmark for routing. At depth 5 it is
saturated for a lexical retriever, which is the opposite problem from SCALE-QA's -- and both are
problems the five instruments exist to surface.

```bash
# streamed, because the file is 277 MB and this machine has less RAM than that
python .scratch/longmemeval_density.py 20260420
```

*Grade: exact, deterministic, offline. Hits are `answer_session_ids` containment under BM25 ranking
over concatenated session text; a "hit" is any gold session in the top k, so a question with 1.90
gold sessions is scored the same as one with a single gold session. The controlled axis is paired
across sizes with n = 500 per bucket and three sampling seeds. No end-to-end accuracy is measured
here -- the 83.6% quoted is Engram's published number on this benchmark and is cited as such, not
reproduced.*
