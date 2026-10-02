# Termination statement, 2026-10-02

This project is being closed as a **router** and kept as a **record**. This is the statement that
says so, what it cost, what it produced, and what a future reader should do with it.

## The question it set out to answer

> In a long conversation interleaving several unrelated tasks, does routing each query to the right
> logical context **reduce the tokens the main model reads, without losing answer quality?**

A conjunction. One half has to hold for the other to matter.

## Where each half stands

### The token half: established, and stronger than the claim

`hybrid_router` reads **1,671 memory tokens** per turn against `full_history`'s median of
**370,102**. That is the 221x the project has been quoting, and it is true.

The stronger and more useful version is not the ratio. It is that **reading everything is often not
an option at all**: against the real input limits of the documented models (260,096 and 258,048
tokens, as reported by the gateway when it refuses), `full_history`'s median input is 485,966 and
**13 of 18 checkpoints exceed the limit**. On the full 24-checkpoint gate, **5 of 24 run**. So the
honest statement is not "routing is cheaper" but "on these conversations, full history does not
run", which is a claim about feasibility rather than about cost.

### The quality half: a null result, and the benchmark could not have shown otherwise

35 adjudicated pairs, **18-17, p = 1.00**. The project spent two and a half weeks and a large share
of its API budget getting to a coin flip, and then established why:

- Its own generator writes the hypothesis into the labels: `recency_reachability` is a **pure step
  function**, 100% of `continue` and `short_coreference` checkpoints reachable, 0% of
  `cross_context`, `return` and `switch`. A benchmark that decides the recency question in its
  generator cannot be evidence about recency.
- The paid judge is a tally at **n = 44 with 21 ties**, and on the project's only controlled
  ablation -- `indexed_router` against prose, same selection, 63% fewer tokens -- it **moved opposite
  to the deterministic metric in both runs**, 2 for 2.
- On the external benchmark the discriminating part is **16% of the questions** (164 of 1,000); the
  other 84% are answered by a single lexical lookup, where every arm scores at or above 0.990.
  Measured across all ten available packages this saturates: **20.1% at 3,000 questions**, with the
  filler axis flat to within a point over a fivefold increase and the dialogue axis rising less than
  four points per tripling. No package reaches the 25% the repo's own `SOLVED_SHARE_CEILING`
  requires, and the discriminating fifth is largely a re-labelling of query-to-evidence string
  overlap (`recall@1` by overlap bucket: 0.463, 0.828, 0.976, 1.000, with 94% of questions below 0.4).
  See `docs/discriminating-mass-2026-10-02.md`.

### The router as a method: it does not beat plain lexical retrieval

On SCALE-QA, a benchmark this project did not build, at window 256 on 3,000 questions:

| | discriminating questions (n = 602) |
|---|---:|
| plain BM25 | **0.556** |
| this project's router | 0.551 |
| paired difference | -0.007, 38 won against 42 lost, p = 0.738, [-0.035, +0.023] |

Statistically identical to BM25, and it was never shown to be better. On a rich candidate list,
where the gold is present for **98.2%** of discriminating questions and the list's own ceiling at
depth 5 is **0.982**, the best ordering this project could produce reaches **0.589**. Feature
engineering over the router's own signals claims about 0.04 of the 0.43 that is missing.

> **Amended 2026-10-02.** The heading is too strong, and the reason is the one
> `docs/discriminating-mass-2026-10-02.md` establishes: SCALE-QA's evidence label is a string, so
> BM25 is scoring against its own representation and no second channel can add to it. On LongMemEval's
> hard question types, where the label is a set of session ids, the same router **does** beat plain
> BM25 by an established margin -- `+0.078`, 31 won against 8 lost, p < 0.001, and **+0.135 on
> `multi-session` alone**, 21 against 3. What survives the amendment is narrower and still costs the
> router: the shipped configuration does not beat the fusion (`-0.017`, 15 against 20), the gain over
> BM25 belongs to equal-weight RRF and not to the calibration, and the ordering stage still leaves
> **0.247** of its own candidate list's ceiling unclaimed.
> `docs/longmemeval-decomposition-2026-10-02.md` has the paired tests.

## Every explanation this project proposed, and what happened to it

This is the part worth keeping. Each row was a plausible, publishable account of the deficit, and
each was measured rather than argued about.

| explanation | outcome |
|---|---|
| the two hard-coded builder constants (recent allowance, recent excluded from evidence) | measured; the preferred value is a **free improvement** (+0.069 reachability, +0 tokens), and it is now a recorded parameter |
| the router under-serves the recent window | **refuted**: it shows the model *more* of what the requirements ask about (0.636 against 0.398) |
| the router is behind its own best channel | **withdrawn**: -0.012, 9 won against 11 lost, p = 0.824; at 602 questions it is bounded at [-0.035, +0.023] and the point estimate flips sign on a random fold |
| equal-weight fusion of unequal channels costs the ordering | **established**: -0.085, 43 against 94, p < 0.001, [-0.123, -0.048] |
| the stage after the fusion recovers most of that | **established**: +0.078, 61 against 14, p < 0.001, [+0.051, +0.106] |
| repairing the fusion improves anything end to end | **refuted**: gating on measured channel recall leaves the held-out score at exactly the hand-set ranker's 0.712 |
| the front end's failure is a corpus-size effect | **refuted as stated**: the 0.115 corpus gap becomes 0.006 at a wide candidate budget |
| candidate generation is the bottleneck | **refuted**: the gold carries a positive score in some channel for 601 of 602 questions; the miss rate is a property of `top_per_retriever=10` and `max_candidates=20` |
| widening the candidate budget improves the answer | **refuted**: gold in the list 0.754 to 0.982, list 18 to 338 items, recall@5 0.550 to 0.528, paired -0.022, p = 0.198 |
| pair-local features fix the ranking | **downgraded twice**: one seed looked like a win over BM25 (p = 0.0195); six seeds showed the direction is consistent but only 1 in 6 is significant; the ablation showed the gain survives every subset, so it belongs to having pair information and to no named feature |
| more ranker work is the answer | **not attempted**: the measured headroom for the whole family is about 0.02 of a 0.43 gap |

The pattern in that table is the project's most transferable output: **three successive culprits --
the fusion weights, the corpus, the candidate budget -- were all hand-set constants or corpus
properties standing next to the stage that was actually failing, and each took one measurement to
rule out.** Sweep the constants around a stage before blaming the stage.

## What outlives the question

Five instruments, each existing because a claim was published and then withdrawn, or because a
benchmark answered a question it could not answer. They are now one step: `ctxlab audit`
(`docs/audit-2026-10-02.md`).

1. `recency_reachability` -- did the arrangement decide the recency question in advance
2. the difficulty split -- how much of a benchmark one cheap lookup already answers
3. the routing-miss decomposition -- never offered, or offered and dropped
4. `candidate_order_attribution` + paired intervals -- and `is_established`, which requires both
   tests to agree because a discrete bootstrap can exclude zero on three discordant pairs
5. the evidence-corpus count -- how many sessions compete, which is what the decomposition is
   actually a function of

And four findings that stand on their own:

- **The corpus-size effect replicates on an independent benchmark** (LongMemEval_S, paired
  truncation, three seeds: hit@1 0.920 to 0.810 and hit@5 1.000 to 0.950 as the corpus goes from 5
  to 40 sessions), with a stated domain: it is only observable when the number of competitors is
  large relative to the retrieval depth.
- **Retrieval is very nearly solved on more than one benchmark.** A plain BM25 over sessions puts a
  gold session in the top five for 95-100% of LongMemEval_S questions, while the published best
  end-to-end result on that benchmark is 83.6% accuracy.
  **Amended 2026-10-02: the 95-100% is an any-gold metric and the inference from it does not hold.**
  LongMemEval_S averages 1.90 gold sessions per question; under all-gold@5, which is what a
  multi-session question asks for, **42.9% of `multi-session` and 28.6% of `temporal-reasoning`
  questions have a required session outside the top five**, and 22.6% of the benchmark does. On that
  benchmark retrieval is the stage that fails. See `docs/discriminating-mass-2026-10-02.md`.
- **The router's features are structurally unable to rank.** Every one of its nine inputs is
  set-normalized, so none can express "this candidate contains the query's rare identifier". Its
  largest single weight sits on `entity`, a feature that is identically zero under the adapter's own
  document representation (0 of 3,561 contexts carry one).
- **On the substrate that discriminates, the front end is nearly free and the ranking is not.**
  LongMemEval_S's hard types, session-level labels, 296 questions: handing the router the whole
  haystack as its candidate list -- so a candidate miss is impossible -- raises the ceiling only from
  **0.922 to 0.990**, while the ordering stage reaches 0.676 of that 0.922 and leaves **0.247**. The
  stage that fails is the ordering stage, by about four to one in ceiling terms. And the sign of the
  fusion effect reverses with the label: RRF costs the ordering **-0.085** against BM25 on SCALE-QA's
  verbatim strings and gains **+0.051** on session ids. See
  `docs/longmemeval-decomposition-2026-10-02.md`.

## What will not be done

- **A cross-encoder or any further ranker work.** The measured headroom for the whole family is
  about 0.04, and the remaining 0.38 needs a model that sees query and candidate together -- a
  different architecture competing against standard RAG, entered from a standing start behind BM25.
  *Both numbers are SCALE-QA's. On LongMemEval's hard types the same quantities are 0.247 of a 0.990
  ceiling for the ordering stage, which is a larger share of a harder problem -- so if this decision
  is ever revisited, it should be revisited on those numbers rather than these. The decision above
  is unchanged: it was made on scope, not on the size of the gap.*
- **More judge rounds.** The instrument was shown to anti-correlate with the deterministic metric on
  the one controlled ablation.
- **More benchmark configurations on the project's own generator.** With 2 to 6 competing contexts
  it reports 0.000 candidate miss and 0.000 selection loss in every configuration on both axes. It
  has no failures to decompose.
- **Shipping the router.** It ties BM25 on a benchmark that is 84% solved and loses on part of the
  rest.

## What a future reader should do instead

If you arrived here wanting a memory system, this repository will not give you one. What it will
give you is a way to find out whether your own evaluation can answer the question you are asking it:

```bash
ctxlab audit <your-benchmark-package> --window-size 256 --depth 5
```

If you arrived here to continue the research, the two concrete open items are:

1. **A joint query-document model.** The candidate list holds the answer 98.2% of the time and no
   reordering of per-candidate scalars claims more than a tenth of that. A cross-encoder is the
   untried architecture, and the ceiling says the candidates justify trying it.
2. **A benchmark whose discriminating part is larger than 16% and whose evidence can be reached by
   paraphrase rather than verbatim.** Every measurement in this repository is conditioned on
   verbatim containment, which flatters lexical retrieval by construction.
   **Amended 2026-10-02: both conditions are now met by one dataset, and it is not SCALE-QA.**
   SCALE-QA saturates at 20.1% discriminating and its recall is monotone in string overlap
   (0.463 at overlap below 0.2 up to 1.000 above 0.6). LongMemEval_S restricted to `multi-session`
   plus `temporal-reasoning` plus `single-session-preference` gives 296 questions of which **35.8%
   fail all-gold@5**, and its evidence label is a session-id set rather than a string, so any passage
   in the session counts.
   **That substrate has now been used, and it moved the answer.** The front end costs 0.068 of the
   ceiling and the ordering stage leaves 0.247; the router beats plain BM25 there by an established
   +0.078 (on `multi-session`, +0.135) but does not beat an equal-weight fusion of the same channels.
   See `docs/longmemeval-decomposition-2026-10-02.md`.

## The honest bottom line

Two and a half weeks, **93 commits**, a null result on quality, and a router that ties the cheapest
baseline it was compared against. The feeling that produces is calibrated against the wrong target:
the target was set to "did I win" before the effect size was known, and the effect, on this evidence,
is small or absent.

What the work actually produced is a documented account of **how a benchmark can fail to be about
what it claims** -- five instruments, a corpus of withdrawn claims with their measurements, and a
finding about evaluation that generalises past this project. That is a scarcer output than another
router, and it is the reason this repository is being kept.

Nothing here was fabricated, and nothing was withdrawn quietly. That is the property worth
preserving.

*Grade: every number in this statement is reproduced, with its configuration and its grade, in the
document that owns it: `docs/recency-budget-2026-10-01.md`, `docs/discriminating-set-2026-10-02.md`,
`docs/fusion-weights-2026-10-02.md`, `docs/composition-sweep-2026-10-02.md`,
`docs/candidate-budget-2026-10-02.md`, `docs/ranker-ceiling-2026-10-02.md`,
`docs/pair-features-2026-10-02.md`, `docs/longmemeval-density-2026-10-02.md`,
`docs/discriminating-mass-2026-10-02.md`, `docs/longmemeval-decomposition-2026-10-02.md`,
`docs/session-report-2026-10-02.md`. The commit count originally given here was 93, carried forward
from the handover. Measured rather than carried forward, main carried **97** at `6829157c`, the
commit this note was written on -- the handover's 83 plus 14 added on 2026-10-02. The figure is
pinned to that commit and not to the branch tip, because the amendment that states a commit count is
itself a commit, so a count quoted against the tip is wrong the moment it is published
(`gh api repos/xxszyh/context-router-lab/commits?sha=main&per_page=1`, `Link` rel="last").
Five passages carry a dated amendment made after
first publication -- the discriminating share, the LongMemEval retrieval claim, the claim that the
router does not beat plain lexical retrieval, open item 2, and this note -- and in each case the
original text is left visible with the correction beside it. The last two amendments were made the
same day as the statement, after the measurements that forced them.*
