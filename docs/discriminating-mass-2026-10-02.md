# Where the discriminating mass actually is, and a metric that was hiding it

2026-10-02. Offline, deterministic, no judge, no API. Three measurements, all on data this project
did not build: a ten-package sweep of SCALE-QA's own discriminating mass, a measurement of how much
of lexical recall is explained by query-to-evidence string overlap, and a re-reading of LongMemEval_S
under the metric that matches what its questions ask for.

`docs/termination-2026-10-02.md` closed this project as a router and left two open items. The second
was: *a benchmark whose discriminating part is larger than 16% and whose evidence can be reached by
paraphrase rather than verbatim.* This is the measurement of whether such a benchmark is available.

It found one. It also found that this project's own claim that **retrieval is very nearly solved on
LongMemEval_S** was an artefact of a generous retrieval metric, and that claim was published in
`docs/longmemeval-density-2026-10-02.md` and repeated in the termination statement. The correction is
in the section "The correction", and it is the most consequential thing in this document.

## Part 1: SCALE-QA cannot be scaled into discriminating

`SOLVED_SHARE_CEILING = 0.75` is the repo's own declared threshold: a package is usable only if more
than 25% of its questions defeat one cheap lookup. The package the project adopted fails it at 16.4%.
The question is whether that is a property of the *package* or of *SCALE-QA*.

Ten packages are on disk and they separate into two confound-free axes:

* **evidence-bearing dialogues**, fixed at ~512k tokens of stream: 50, 1,000, 1,500, 2,000, 3,000
* **filler**, fixed at 3,000 questions: 400k, 512k, 768k, 1m, 2m tokens of stream

Window 256, depth 5, `interleaved` arrangement, seed 20260420. "Hard" is a lexical lookup at depth 1
that does not return a unit holding all of the question's evidence -- the same rule
`difficulty_strata` uses.

| package | questions | evidence dialogues | windows | solved@1 | hard | hard share |
|---|---:|---:|---:|---:|---:|---:|
| pkg_128k_50 | 50 | 50 | 623 | 44 | 6 | 12.0% |
| pkg_168k_1000 | 1,000 | 1,000 | 1,175 | 832 | 168 | 16.8% |
| pkg_512k_1000 | 1,000 | 1,000 | 2,810 | 836 | 164 | 16.4% |
| pkg_252k_1500 | 1,500 | 1,500 | 1,759 | 1,243 | 257 | 17.1% |
| pkg_336k_2000 | 2,000 | 2,000 | 2,351 | 1,630 | 370 | 18.5% |
| pkg_400k_3000 | 3,000 | 3,000 | 3,033 | 2,402 | 598 | 19.9% |
| pkg_512k_3000 | 3,000 | 3,000 | 3,561 | 2,398 | 602 | 20.1% |
| pkg_768k_3000 | 3,000 | 3,000 | 4,785 | 2,400 | 600 | 20.0% |
| pkg_1m_3000 | 3,000 | 3,000 | 5,884 | 2,380 | 620 | 20.7% |
| pkg_2m_3000 | 3,000 | 3,000 | 8,195 | 2,387 | 613 | 20.4% |

**The filler axis is flat.** At 3,000 questions, growing the stream fivefold -- 3,033 windows to
8,195, the noise share going from 23% to 80% -- moves the hard share from 19.9% to 20.4%, with no
monotone trend across the five points. This reproduces the claim already recorded on
`ArrangementReachability.truth_blocks`, now for the discriminating share rather than for the
candidate-miss rate.

**The dialogue axis rises and saturates.** 12.0% at 50 dialogues, 16.4% at 1,000, 17.1% at 1,500,
18.5% at 2,000, 20.1% at 3,000. Each tripling of the corpus from 1,000 bought less than four points,
and the last two points cost 1,000 more dialogues and 750 more windows.

**No package clears the declared ceiling.** The best available is 20.7%, and it is not the largest
corpus -- it is one of five statistically indistinguishable points in the 19.9-20.7% band. So the
16% the project adopted was not a bad package. It is roughly what SCALE-QA is: **a benchmark whose
discriminating mass tops out near a fifth of its questions, and cannot be pushed past it by adding
either more evidence or more filler.**

That matters for the termination statement's arithmetic. Its "16% (164 of 1,000)" is correct for the
package quoted; the honest generalisation, now measured, is **20% at 3,000 questions, saturated**.

## Part 2: SCALE-QA's difficulty is string overlap

If the discriminating fifth of the benchmark is the part that decides everything, then what makes a
question discriminating is worth knowing. The evidence metric is verbatim containment, so the natural
suspect is that a question is hard exactly when its wording does not literally appear in the passage
holding the answer.

Measured on `pkg_512k_1000`, 999 scored (one question has no window holding all its evidence at this
window size). For each question the **stem** is tokenised -- everything before the answer options,
because a SCALE-QA `query` carries its four options in the same string and the options are the
distractors, not the question. Overlap is the fraction of stem tokens that also occur in the passage
holding the evidence. recall@1 is the dependent variable, because recall@5 is saturated at 0.941.

| stem overlap with the evidence | n | share | recall@1 | recall@5 | mean IDF-weighted overlap | unique shared terms | mean corpus df of shared terms |
|---|---:|---:|---:|---:|---:|---:|---:|
| (0, 0.2] | 67 | 6.7% | **0.463** | 0.642 | 0.089 | 0.24 | 1338.8 |
| (0.2, 0.4] | 709 | 71.0% | **0.828** | 0.951 | 0.226 | 0.55 | 1002.3 |
| (0.4, 0.6] | 205 | 20.5% | **0.976** | 1.000 | 0.413 | 1.20 | 777.4 |
| (0.6, 0.8] | 15 | 1.5% | 1.000 | 1.000 | 0.709 | 2.13 | 632.8 |
| (0.8, 1.0] | 3 | 0.3% | 1.000 | 1.000 | 0.949 | 0.33 | 579.7 |
| **all** | 999 | 100% | 0.837 | 0.941 | | | |

**Monotone in every bucket.** 0.463, 0.828, 0.976, 1.000. The 163 questions the baseline gets wrong
at depth 1 have a mean stem overlap of **0.256**; the 836 it gets right average **0.351**.

The benchmark's body is concentrated where the gradient is shallow: **94% of questions sit below 0.4
overlap and 71% sit in a single bucket**, and only 1.8% sit above 0.6, where recall is already 1.000.
So the two subsets the project has been calling "the easy 84%" and "the hard 16%" are close to the
same partition as "high string overlap" and "low string overlap" -- and the second one is
discriminating because it is measured by a metric that requires the answer's wording to appear.

The `mean corpus df` column is the counterweight, and it is why this is stated as a gradient rather
than as a mechanism. The shared tokens have a **mean document frequency of 579 to 1,338 out of 2,810
windows** -- they are mostly function words that every passage shares. The genuinely discriminating
signal sits in a small number of rare ones: the mean count of shared terms occurring in exactly one
window rises 0.24, 0.55, 1.20 as overlap rises. On this corpus **46.7% of the 30,368 distinct terms
appear in exactly one window**, so "find the window containing this unique name" is a large part of
what a lexical lookup does -- but it is not the whole of it, because recall still varies inside a
bucket.

**What this supports:** the discriminating subset of SCALE-QA is substantially a re-labelling of
string overlap, so a result measured on it is a statement about tokenisation before it is a statement
about memory. **What it does not support:** that overlap *determines* the outcome. Inside the 71%
bucket recall ranges over 0.828, so there is residual signal the overlap does not carry.

## Part 3: the correction -- LongMemEval_S under the metric its questions ask for

`docs/longmemeval-density-2026-10-02.md` reports that a plain BM25 puts a gold session in the top
five for **95% to 100%** of LongMemEval_S questions and concludes: *"Retrieval is very nearly solved
on LongMemEval_S too."* The termination statement repeats it as one of three findings that stand on
their own.

That is the **any-gold** metric: the question is scored a hit if *at least one* `answer_session_ids`
entry is retrieved. Its grade note says so explicitly -- *"a 'hit' is any gold session in the top k,
so a question with 1.90 gold sessions is scored the same as one with a single gold session"* -- but
the consequence was never followed through.

LongMemEval_S averages **1.90 gold sessions per question**, and its question types are not
interchangeable: `multi-session` (133 of 500) averages 2.59, `temporal-reasoning` (133) averages
2.20, and `knowledge-update` (78) averages 2.00. A question that needs three sessions to answer is
not answered by retrieving one of them. The metric that matches the question is **all gold sessions
in the top k**, with **coverage** -- the mean fraction of gold sessions retrieved -- as the version
that does not assume every listed session is necessary.

The same 500 instances, the same BM25 over concatenated session text, re-scored under all three.

**Natural haystack** (38 to 62 sessions, median 48):

| question type | n | gold sessions | any@1 | any@5 | **all@5** | coverage@1 | **coverage@5** | coverage@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single-session-assistant | 56 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| single-session-user | 70 | 1.00 | 0.843 | 0.986 | 0.986 | 0.843 | 0.986 | 1.000 |
| knowledge-update | 78 | 2.00 | 0.936 | 1.000 | 0.923 | 0.468 | 0.962 | 0.981 |
| temporal-reasoning | 133 | 2.20 | 0.744 | 0.940 | **0.714** | 0.388 | 0.834 | 0.897 |
| multi-session | 133 | 2.59 | 0.797 | 0.947 | **0.571** | 0.341 | **0.775** | 0.884 |
| single-session-preference | 30 | 1.00 | 0.267 | 0.633 | **0.633** | 0.267 | **0.633** | 0.833 |
| **all** | **500** | 1.90 | 0.802 | 0.946 | **0.774** | 0.513 | 0.866 | 0.929 |

**Controlled at 40 sessions** (every gold session kept, the rest sampled down, paired):

| question type | n | any@5 | all@5 | coverage@5 |
|---|---:|---:|---:|---:|
| temporal-reasoning | 133 | 0.940 | 0.737 | 0.848 |
| multi-session | 132 | 0.947 | 0.598 | 0.791 |
| single-session-preference | 30 | 0.667 | 0.667 | 0.667 |
| **all** | **496** | 0.946 | **0.790** | 0.875 |

The any-gold column reproduces the published numbers to within a point (0.946 here against 0.950 and
0.954 in the earlier three-seed run), so this is the same measurement, read differently.

**Read differently, it says the opposite.**

* Under **all-gold@5**, 22.6% of LongMemEval_S questions have at least one required session missing
  from the top five. Under any-gold@5, 5.3% do. The published claim quoted the second.
* It is concentrated: **42.9% of `multi-session`** and **28.6% of `temporal-reasoning`** questions
  fail all-gold@5, against 1.4% of `single-session-user` and 0% of `single-session-assistant`.
* `single-session-preference` fails the generous metric too -- **36.7% at any@5**, the worst type on
  the benchmark, and it is single-gold so all three metrics agree.
* The failure is not a corpus-size artefact: the controlled run at 40 sessions gives 0.598 for
  `multi-session` against 0.571 natural, and 0.737 against 0.714 for `temporal-reasoning`.
* The dataset card for `xiaowu0162/longmemeval-cleaned` states the version *"removes noisy history
  sessions that interfere with the answer correctness"*. The corpus has been cleaned of the
  sessions that most plausibly interfere, which should make retrieval **easier**; the deficit
  survives that.

So the finding that stood on its own -- retrieval is solved there as well, so the loss must be
downstream -- **does not survive the metric that matches the questions.** On the third-party
benchmark this project chose precisely because it has no filler and every competitor is a
near-miss, retrieval is the stage that fails for roughly a third of the questions that need more
than one session.

### What this does and does not establish

**Does not establish** that the earlier document was wrong about what it measured. It was explicit
about the metric, and its any-gold numbers reproduce exactly. What is withdrawn is the *conclusion*
drawn from them, which generalised an any-gold measurement to a claim about retrieval.

**Does not establish** that `answer_session_ids` is exactly the necessary set. Treating it as
necessary is the strong reading; `all_gold` depends on it and `coverage` does not. `coverage@5`
of 0.775 for `multi-session` means a quarter of the sessions LongMemEval names as holding the
answer are not retrieved, which is a retrieval deficit under any reading of the label.

**Does not establish** an end-to-end quality effect. Nothing here measures whether an answer got
better or worse; it measures whether the evidence was in front of the model. That distinction is the
one `docs/termination-2026-10-02.md` spent its length on, and it is not resolved here.

**Establishes** that the retrieval-stage deficit on LongMemEval_S is large, type-concentrated,
stable under corpus control, and invisible to the metric this project reported.

## The verdict on the open item

The termination statement's second open item asked for a benchmark whose discriminating part exceeds
16% and whose evidence is reachable by paraphrase rather than verbatim. **Both conditions are met by
the same data, and neither is met by SCALE-QA.**

| | SCALE-QA (best package) | LongMemEval_S, `multi-session` |
|---|---:|---:|
| discriminating share | 20.1%, saturated | **42.9%** (all-gold@5) |
| evidence reachable by paraphrase | no -- verbatim containment, and recall tracks overlap (0.463 to 1.000) | **yes** -- gold is a session-id set, so any passage in the session counts |
| third party | yes | yes |
| filler | 23-80% of the stream, inert | none |
| stable under corpus control | yes (filler axis flat) | yes (40-session paired run) |

Restricting LongMemEval_S to `multi-session` + `temporal-reasoning` + `single-session-preference`
gives 296 questions of which **35.8%** fail all-gold@5 -- above the repo's own declared 25% ceiling,
on a benchmark whose evidence labels are not strings.

That is where the next measurement should be taken, and it is the first time in this project that a
substrate has existed for the question "is the loss in retrieval or downstream" that can actually
distinguish the two.

```bash
# Part 1: the ten-package sweep (lexical phase only; no model load)
python .scratch/discriminating_sweep.py --phase hard

# Part 2: overlap-bucketed recall on one package
python .scratch/verbatim_dependence.py pkg_512k_1000

# Part 3: LongMemEval_S under any-gold, all-gold and coverage, by question type
python .scratch/longmemeval_strata.py
```

*Grade: exact, deterministic, offline, no model call and no judge in any part. Part 1 runs the
lexical arm only, which returns from BM25 before touching the embedder, so it is the placeholder
embedder and not a dense or router measurement; the hard rule is `difficulty_strata`'s, a baseline
lookup at depth 1. Part 2 scores 999 of 1,000 questions, one having no window holding all its
evidence at window 256, and its buckets are stated with their n because the two highest-overlap
buckets hold 18 questions between them. Part 3 re-scores the same 500 instances the earlier run
scored, with the any-gold column reproducing it, and the controlled arm keeps every gold session
while sampling the rest with a seeded RNG. `.scratch/` is local to this machine and is not part of
the published tree; the scripts are named so the runs can be reproduced, not because they ship.*
