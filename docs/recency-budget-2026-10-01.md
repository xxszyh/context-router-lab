# The recency budget, and a hypothesis that did not survive being measured

2026-10-01. Two numbers were literals inside `ContextBuilder.assemble` until today: the share of
the token budget the immediately-preceding turns may claim (`0.15`), and whether those turns are
barred from the evidence channel (`True`). Neither appeared in any artifact. Every result this
project has published rests on both.

This records what happened when they were made measurable, and the measurement that refuted the
hypothesis they were made measurable to test.

## What changed, and what did not

`RECENT_BUDGET_FRACTION` and `EXCLUDE_RECENT_FROM_EVIDENCE` are now module constants with builder
parameters behind them, recorded by `describe_builder` on every arm result and pinned by
`tests/test_arms.py::test_default_assembly_profile_is_pinned`. The defaults are unchanged, and
that is checked rather than asserted: all nine rows of the README's synthetic table reproduce
**bit-identically** (`hybrid_router` 714.3 tokens at recall 1.000, `global_bm25` 749.7, oracle
440.2, and the rest to the decimal), as do the real-conversation memory-token counts
(`sea-ice` hybrid 1785.2 / recent 1004.1; `d22f2593` 1652.8 / 1108.0 on the 24 checkpoints that
ran). A change that moves a published number has to be a deliberate one.

Two instruments came with it.

**`requirement_reachability(requirements, memory)`** — the share of *substantive* requirements
whose checkable terms are present in the assembled memory. It is an upper bound on what any answer
drawn from that memory could score, and it separates "the model did not say it" from "the model was
never shown it". Refusal requirements are excluded rather than scored: a context containing
`无法回答` would satisfy one for the wrong reason. Unmeasured returns `None`, never `0.0`.

**`recency_reachability(cases)`** — per query type, whether the labelled evidence sits inside the
recency window, with a `definitional` flag for the types where every labelled checkpoint is
reachable or none is. This is the check whose absence let a self-confirming benchmark read as an
empirical result.

## The hypothesis, and its refutation

The motivating observation was real and is unchanged. On the two real conversations
`hybrid_router` shows far fewer of the immediately-preceding turns than `query_recent_only`:

| sea-ice, events of the last three turns shown | `query_recent_only` | `hybrid_router` |
|---|---:|---:|
| `continue` (14) | 2.93 | **1.86** |
| `return` (8) | 2.88 | **1.25** |
| `cross_context` (16) | 2.31 | 1.44 |

and on only 7 of 44 checkpoints does the router's memory contain all of what the baseline shows.
The mechanism was visible in the code: a 15% recent allowance (307 of 2048 tokens) plus an evidence
channel barred from the recent window means a recent turn that does not fit the allowance cannot be
recovered by any other channel.

**So the prediction was: on `continue`, the router is asking the model to answer from text it was
not given, and its requirement reachability should sit below the baseline's.**

**It does not. It sits above it, on every type but one:**

| requirement reachability, default profile | `query_recent_only` | `hybrid_router` |
|---|---:|---:|
| sea-ice, pooled | 0.398 | **0.636** |
| sea-ice, `continue` | 0.321 | **0.500** |
| d22f2593, pooled | 0.236 | **0.438** |
| d22f2593, `continue` | 0.333 | **0.500** |

The one cell where the baseline reaches more is sea-ice `short_coreference` (1.000 against 0.750).

A confound was checked before this was believed. The router inserts context descriptors
(`[Context: …] Goal: … Summary cue: …`) that could echo the requirement vocabulary without
containing any evidence. Descriptors are 5.0% (sea-ice) and 5.8% (d22f2593) of the router's memory
text; stripping every one of them moves sea-ice by **0.000** and d22f2593 by 0.014. The advantage
is in the evidence, not the labels.

**The under-serving hypothesis is therefore withdrawn.** The router shows the model *more* of what
the requirements ask about, not less. The reason it loses the `continue` cell is not that the
evidence is missing from its memory.

## Where the loss actually is

If the router's memory reaches more and its answers do not improve, the loss is between the memory
and the answer. The deterministic answer-side metric says exactly that:

| | `query_recent_only` | `hybrid_router` | judge (anchored, 3 repeats) |
|---|---:|---:|---|
| sea-ice, coverage of the answer | **0.333** | 0.288 | recent 14 – hybrid 9 |
| d22f2593, coverage of the answer | 0.285 | 0.285 | **hybrid 9** – recent 3 |

On sea-ice the router is shown more of what is needed and names less of it. On d22f2593 the two
answers are indistinguishable to the lexical metric and the judge separates them anyway. Neither
row is a routing result; both are consistent with the model not converting the extra material into
the answer.

*Grade: the reachability and coverage columns are exact and deterministic; the judge column is a
tally at n = 44 and n = 24 with 21 and 12 ties and is reported as a direction only.*

**One correction to `state-of-play` while here.** It says the coverage control and the paid judge
"disagree on direction on real data". Under the anchored judge they **agree** on sea-ice -- both
favour recency -- and differ only on d22f2593, where coverage ties exactly (0.285 against 0.285)
and the judge does not. The blanket statement is too strong.

## The frontier: reachability tracks tokens, not routing

| sea-ice | tokens | reachability |
|---|---:|---:|
| `query_recent_only` | 1004 | 0.398 |
| `indexed_router` | 790 | 0.542 |
| `hybrid_router` | 1785 | **0.636** |
| `global_bm25` | 1978 | **0.636** |

| d22f2593 | tokens | reachability |
|---|---:|---:|
| `query_recent_only` | 1117 | 0.236 |
| `indexed_router` | 630 | 0.306 |
| `hybrid_router` | 1681 | 0.438 |
| `global_bm25` | 2018 | **0.618** |

Two things in that table matter more than the ranking.

**On sea-ice the router earns its keep on cost, not on reach.** 1785 tokens at 0.636 against
`global_bm25`'s 1978 at 0.636: the same reach for 10% fewer tokens. That is the claim the project
makes on synthetic data, reproduced on a real conversation.

**On d22f2593 it does not, and is beaten outright.** `global_bm25` reaches 0.618 against the
router's 0.438. Narrowing the candidate set to the selected contexts costs reachability there. A
routing result that holds on one conversation and reverses on the other is not a result yet, and
this is the second independent axis on which these two conversations disagree in sign.

Raising the recent allowance helps the router and not the baseline: at `recent_fraction=0.50`,
sea-ice reachability rises 0.636 → **0.705** at 1783 tokens, i.e. no additional cost. That is a
free improvement and it is now a recorded configuration rather than a literal.

## The benchmark, on the real conversations

The diagnostic was run on real replay, and the answer is worse than the synthetic finding alone
suggested.

`sea-ice` carries **no `acceptable_evidence_sets` at all** (labelled = 0 for every query type), so
`evidence_set_recall` is unmeasured there and always was; the arm tables on that conversation are
token tables only.

`d22f2593` carries labels, and for four of its six types **every labelled checkpoint's evidence is
inside the last three turns**:

| d22f2593 | labelled | reachable | |
|---|---:|---:|---|
| `continue` | 2 | 2 | definitional |
| `cross_context` | 7 | 7 | definitional |
| `short_coreference` | 3 | 3 | definitional |
| `switch` | 5 | 5 | definitional |
| `return` | 7 | 5 | 0.71 — the only type that can separate the strategies |
| `new_context` | 0 | — | unlabelled, not a verdict |

On the committed synthetic dataset the same diagnostic is a pure step function, and it is now
pinned by a test:

| synthetic | labelled | reachable |
|---|---:|---:|
| `continue`, `short_coreference` | 240 | **100%** |
| `cross_context`, `return`, `switch` | 420 | **0%** |

**So "`query_recent_only` recovers only 46.2% of the labelled evidence" is a statement about the
generator.** It is 0.364 on the committed dataset, and it is exactly 240/660 because the generator
decides in advance which checkpoints recency can serve. A benchmark whose recency reachability is
0% or 100% within every query type cannot be evidence for or against routing, however many
checkpoints it has. `RecencyReachability.definitional` is that flag, and the test that pins the
synthetic numbers fails if the generator is ever changed to mix them -- which is the intended
behaviour, because the docs quoting 46.2% would have to move with it.

## The one controlled ablation the project already has, re-read

`indexed_router` is `hybrid_router` with the rendering changed and nothing else: same router, same
decision, same events, same order. That makes it the project's only experiment in which the
*selection is held fixed* and only the amount of text varies -- which is exactly the experiment the
dilution question needs. Two such runs exist, on the **identical 24 checkpoints**, same arms, same
model (`kimi-k2.5`), same budget, differing only in how much of each body survives
(`head=160` against `head=320`):

| run | `hybrid_router` tokens / coverage | `indexed_router` tokens / coverage | judge |
|---|---:|---:|---|
| `head=160` | 1653 / 0.299 | **611** / 0.292 | **indexed 5** – hybrid 2 (15 ties) |
| `head=320` | 1636 / 0.290 | **814** / **0.340** | hybrid 6 – **indexed 3** (12 ties) |

Two readings, and the second matters more than the first.

**Dilution is not confirmed.** Cutting 63% of the memory (1653 → 611 tokens) at a fixed selection
left coverage where it was (0.299 → 0.292). If the router's problem were that it shows the model
too much, this is the run that should have shown it, and it does not.

**The judge reverses with the rendering parameter, and moves opposite to the deterministic metric
both times.** At `head=160` the cheaper rendering has lower coverage and the judge prefers it; at
`head=320` it has higher coverage and the judge prefers the other one. That is two for two
anti-correlation on a controlled ablation -- not a noisy instrument, a *systematically* disagreeing
one, and the disagreement is with the metric that cannot see verbosity.

*Grade: the token and coverage columns are exact. Both judge runs are single-repeat and were
produced before `judge_style` was recorded, so by this project's own rule neither tally is
quotable; what is quotable is that they disagree with each other and with coverage. Re-running
these two ablations at `--repeats 3 --judge-style anchored` is 2 x 24 x 6 = 288 calls, cheaper than
the 264-call sea-ice round, and it tests the instrument the whole quality claim rests on.*

## The self-confirmation is not this benchmark's flaw. It is the whole class's.

This project's generator writes its hypothesis into its labels, and the first reading of that is
"our generator is bad". Adopting an external benchmark tests that reading, and it does not survive.

[SCALE-QA](https://github.com/LordTARN1SHED/SCALE-QA) (arXiv:2608.25655, EMNLP 2026; data CC BY 4.0,
TSIM code Apache-2.0) is the closest published setting to this project's question: 3,000 four-way
multiple-choice questions over ten domains, each with an item-level source dialogue, and a runtime
builder that **flattens those dialogues into one mixed-topic turn stream with the episode metadata
removed**. Each question carries `expected_doc`, the exact evidence copied out of its own dialogue --
the per-query evidence ground truth this project had to delete as circular -- and grading is
deterministic multiple choice, so no judge is needed at all.

Its builder was run here (`--context-budget 128k --num-questions 50`): 50 dialogues totalling 6,855
tokens, filled with 121,238 tokens of bundled UltraChat noise, one 128,439-token stream of which
**5.3% is truth**.

**And a recency baseline scores zero on it.** Concatenating truth then noise -- the arrangement a
plain "combine `GROUND_TRUTH_HISTORY` and `NOISE`" reading produces -- puts every question's
evidence a median of **125,350 tokens** from the end of the stream. Within the last 128, 256, 320,
1,000, 5,000 and 20,000 tokens: **0 of 50**. Not one query.

The mirror image of this project's own benchmark, which puts the evidence inside the recency window
100% of the time for some query types. Both are degenerate. **The arrangement is chosen by the
evaluator, not by the benchmark**, so the same benchmark can be made to prove recency right or
recency wrong without changing a single question.

That is the finding, and it is bigger than this project. *The step function is not a defect of a
generator; it is a property of the whole class of benchmarks that flatten dialogues into a stream,
and it follows the question to any external benchmark unless the arrangement is controlled and its
recency reachability is reported.* `recency_reachability` is not a patch for this repo's generator.
It is the instrument that tells you whether any such benchmark can answer the question you are
asking it, and it runs unchanged on SCALE-QA.

What SCALE-QA does give, and this project's own benchmark cannot, is the ability to *fix* it:
exact per-query evidence, deterministic grading, a documented construction, and a published
reference implementation to sit beside. Adopting it does not make the question answerable by
itself. It makes it answerable *deliberately*.

## The external measurement

`ctxlab scale-qa` runs this project against SCALE-QA offline and free. The adapter is
`src/context_router/external/scale_qa.py`; the arrangement is a named, seeded parameter, and the
report prints its recency reachability before any recall number so a degenerate arrangement cannot
be read as a method result.

On the 1,000-question, 512k package (445,373-token stream, 26.7% truth, `interleaved`):

| window, all 1,000 questions | units | `lexical` | `dense` | `hybrid` | `router` |
|---:|---:|---:|---:|---:|---:|
| 256 | 2,810 | **0.941** | 0.816 | 0.939 | 0.922 |
| 320 | 2,404 | **0.942** | 0.837 | 0.937 | 0.918 |
| 256, matched 200 questions | 2,810 | **0.950** | — | — | **0.950** |

`lexical` leads at both window sizes. `router` is 0.019 and 0.024 behind it, `dense` is 0.125 and
0.105 behind, and `hybrid` -- the fusion -- lands between them without closing the gap to lexical.

**Exact-evidence recall at depth 5, deterministic, no model and no judge.**

`ContextRouter.route` rebuilt its BM25 index, re-embedded every context and re-serialised the whole
catalog on every call, so the router arm originally ran capped at 200 questions and the baseline
was re-run on the same 200, because a 1,000-question baseline against a 200-question router is two
numbers about two different samples. Matched on those 200, they were **identical to three
decimals**. `index_cache_size` was then added, keyed by the caller and carrying the catalog's
version hash, and the uncapped run gives the table above.

**On this external benchmark the project's router does not beat plain lexical retrieval.** It ties
on one sample and loses on another, and on the 50-question package it loses at all three window
sizes -- 0.880 / 0.980 / 0.980 against 0.960 / 1.000 / 1.000. `hybrid` lands just under `lexical`
too, so RRF fusion does not recover what the dense channel loses.

One part of that gap is this project's own document representation, and it was measured rather
than guessed. `FlatContext.searchable_text()` concatenates name, goal, summary, entities and
lexical terms into the indexed document; padding a window with identifiers and top terms it
already contains lengthens the document, flattens the BM25 weights and costs **eight points of
recall** (0.900 with the derived fields, 0.980 without). `_window_context` now leaves them empty
by default and keeps the knob so the effect stays re-measurable.

### The dense channel, with a real neural encoder

The question README limitation 2 leaves open is whether the dense channel helps once the embedder
carries semantics. It has now been measured with a pinned static sentence encoder
(`minishlab/potion-base-8M` through `model2vec`, CPU-only, no API; the model hub is unreachable
from this machine but a mirror serves it). Window 256 on the 1,000-question package:

| embedder | `lexical` | `dense` | `hybrid` | `router` |
|---|---:|---:|---:|---:|
| `hash` (placeholder) | 1.000* | 0.300* | 0.940* | — |
| **`neural` (potion-base-8M)** | **0.941** | **0.711** | 0.928 | **0.931** |
| `lsa` (corpus-fitted) | 0.941 | 0.816 | 0.939 | 0.922 |

*the `hash` row is the 50-question package, where the placeholder scores 0.300 against 0.900 for
the encoder and 0.960 for LSA -- the first direct measurement of what limitation 2 was pointing
at, and the reason no dense number from this project was quotable before.*

**The dense channel does not help, and the placeholder was not the whole reason.** A real neural
encoder reaches 0.711 where lexical reaches 0.941. LSA reaches 0.816, *above* the neural encoder,
which is explicable rather than paradoxical: LSA is fitted on the very windows it retrieves over,
so it learns that corpus's term structure, while the encoder is zero-shot.

Three limits on this, and the first is the one that keeps the question partly open. potion-base-8M
is small, English-centric and **static** -- no attention, no word order -- so this settles "a small
static encoder does not help here", not "no neural encoder would". A transformer encoder with a
larger parameter count remains untested, and remains a one-argument swap. Second, the metric is
verbatim evidence containment, which is inherently lexical-friendly; a semantic retriever's
advantage is paraphrase matching, and containment cannot credit a passage fetched in other words
(the evidence strings are copied verbatim out of the dialogue, so a retriever that finds the right
dialogue does score). Third, and this is the honest reading of all three rows together: **`hybrid`
never recovers what the dense channel loses** -- 0.928 against lexical's 0.941 -- so RRF fusion is
buying nothing on this benchmark either.

One row of that table is worth its own sentence. The router scores **0.931** with the neural
encoder against **0.922** with LSA -- a better dense channel moves it *up* by nine points of a
point, and it still lands below plain lexical at 0.941. So the dense channel is not inert for the
router: it is a real input to candidate generation and to the ranker's features. It is simply not
enough to close a gap that, at this point, four independent measurements agree exists.

### A corpus-fitted embedder, and what it does to one of the README's explanations

`dense` and `hybrid` were not reported above because the default embedder is a hashing placeholder
and would have measured the placeholder. `ctxlab scale-qa --embedder lsa` fixes that offline:
`LsaEmbeddingProvider` fits TF-IDF plus a truncated SVD on the window corpus, which is a genuine
distributional embedding -- texts sharing context land near each other, which hashing cannot do --
and is deterministic. It is not a neural encoder, and it is fitted on the corpus it retrieves over,
and both of those are stated in its docstring. A pinned neural encoder remains a one-argument swap
through `OpenAICompatibleEmbeddingProvider`; the model hub is unreachable from this machine.

On the 1,000-question package at window 256:

| arm | all-evidence recall@5 |
|---|---:|
| `lexical` (BM25) | **0.941** |
| `dense` (LSA) | 0.816 |
| `hybrid` (RRF of the two) | 0.939 |

**The README explains `global_dense` scoring below `global_bm25` by the placeholder embedder
(limitation 2), and this weakens that explanation.** With an embedder that does carry semantics,
dense retrieval is *further* below lexical, not closer to it -- 0.816 against 0.941, where the
placeholder's own numbers were 0.939 against 1.000. The placeholder is still not a real dense
retriever, and this is still not a neural encoder, so the limitation stands as written. But
"dense loses because the embedder is a placeholder" is no longer the only reading available, and
the honest statement is the weaker one: on this task and these two embedders, dense retrieval has
not yet been shown to help, and the fusion in `hybrid` does not recover what the dense channel
loses (0.939 against lexical's 0.941).

Two limits on these numbers, both stated in the module. The dense channel is the hashing
placeholder, so `dense` and `hybrid` measure the placeholder and are not reported, and the
`lexical` baseline is therefore **not numerically comparable** to SCALE-QA's own dense
fixed-window figures (0.719 / 0.647 / 0.577). And windows never span a dialogue boundary, which is
easier than the paper's control.

## Why lexical is strong, and what it says about the benchmark

Four arms were measured and lexical won every time. That is either a fact about the task or a fact
about the benchmark, and the difference decides whether any of those numbers is evidence about a
retriever. Three hypotheses were written down and tested; two are refuted.

**H1/H3 -- the task is a string match.** Refuted. The query contains a median of only **31%** of
its gold evidence's terms, and exactly **1 of 1,000** questions has the query containing all of
them. This is not lookup.

**H2 -- the queries carry rare surface tokens (`Aegis-7`, `PEG-20`, `Patient #402`) that BM25
matches by IDF and a zero-shot encoder cannot.** Half right, and not the explanation. 681 of 1,000
questions share at least one rare token (document frequency <= 5 over 2,810 windows) with their
gold evidence, and having one is worth **+0.134** to lexical:

| subgroup | n | `lexical` | `dense` | `hybrid` | `router` |
|---|---:|---:|---:|---:|---:|
| shares a rare token | 681 | **0.984** | 0.775 | 0.975 | 0.972 |
| shares no rare token | 319 | **0.850** | 0.574 | 0.828 | 0.843 |

But the gap it was meant to explain is still there without them, and is *wider*: 0.276 against
0.209. Whatever the dense encoder is losing, it is not the rare tokens.

**The answer is the benchmark's difficulty distribution.** A question is *solved* when a single
unit returned by the lexical baseline at depth 1 already contains all of its evidence: **836 of
1,000**, median rank 1, recall@5 94.1%. `ctxlab scale-qa` now reports this split beside the pooled
score (`difficulty`, schema 1.1), so the two numbers cannot drift apart.

It is defined on `expected_doc` alone, which is why it runs on any benchmark that carries evidence
labels -- and it differs from the earlier dialogue-level phrasing, "BM25's first hit is already the
gold window", 837, on **exactly one question**. That difference is real rather than rounding.
Question 645 carries two evidence items that a 256-token window cuts in half: no unit holds both,
so no arm can answer it, while the dialogue-level test counts it as solved because its top hit
lands in the right dialogue and finds one of the two items. **837 = 836 + 1 question that is
unanswerable at this window size**, which also puts a ceiling of 0.999 rather than 1.000 on every
arm here.

| subset | n | `lexical` | `dense` | `hybrid` | `router` |
|---|---:|---:|---:|---:|---:|
| one lookup already holds all the evidence | 836 | **1.000** | 0.776 | **1.000** | 0.990 |
| the rest | 164 | **0.640** | 0.378 | 0.561 | 0.628 |

**Eighty-four percent of this benchmark is not a retrieval problem.** One BM25 lookup answers it,
and every arm that contains a lexical channel scores at or above 0.990 there. The entire question
lives in the remaining 164, where lexical still leads at 0.640 against the router's 0.628 and
`hybrid`'s 0.561. Pooled, the same four arms span 0.941 to 0.928; stratified, they span 1.000 to
0.990 on the part that measures the benchmark and 0.640 to 0.378 on the part that measures them.

That is the same finding this project made about its own synthetic benchmark, in a new place.
There, the generator decided the recency question in its labels, and the diagnostic that surfaced
it was `recency_reachability`. Here, a single BM25 lookup decides 84% of the retrieval question,
and the diagnostic that surfaces it is this split. **In both cases the benchmark's difficulty is
not located where the method's claimed advantage is**, and in both cases the pooled score hides it.

Two consequences, and the second is a change to how results on this benchmark should be read.

1. **Any arm that contains lexical retrieval is being scored mostly on a solved problem.** The
   0.941 / 0.931 / 0.928 band at the top of every table in this document is 836 questions where all
   of them are near-perfect plus 164 where they are all mediocre. Reporting the pooled number as
   the comparison is what makes the arms look close; reporting the split shows the arms differ by
   0.010 on the part that discriminates and by 0.010 on the part that does not.
2. **The router's deficit is not where the design claims its advantage.** It loses 0.010 on the
   solved 836 and 0.012 on the unsolved 164 -- and the second of those is smaller than that sample
   can resolve (the paired test below puts it at 9 questions won against 11 lost), so only the first
   is a measured difference. A relation-aware, calibrated ranker was supposed to earn its keep on
   the hard cases, and the decomposition below shows where the distance actually is.

The generalisable instrument is the split itself: **before quoting a pooled score on any retrieval
benchmark, measure how much of it one cheap baseline already answers.** That is `recency_reachability`
generalised from "can a recency window reach the evidence" to "can a single lookup already answer
this", and it runs on any benchmark that carries evidence labels. It is no longer a script: it is
`difficulty_strata` in the adapter, printed by `ctxlab scale-qa` for every window size and recorded
in the report, because the pooled score it qualifies was the thing being quoted.

## The hard subset: the loss is in the ranking, not in the retrieval

The 164 questions the split isolates are the only ones that can decide anything, so the router's
behaviour there was decomposed the way this project already decomposes a routing miss on its own
benchmark -- was the evidence never offered (a candidate miss), or was it offered and then dropped
(a selection loss)?

| on the 164 hard questions | n |
|---|---:|
| the gold window was in the router's candidate list and was returned | 103 |
| selection loss -- in the candidate list, not returned | 37 |
| candidate miss -- never in the candidate list | 24 |

and the router's own candidate ranking, exact-evidence recall at increasing depth:

| depth | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| router candidate list | 0.195 | 0.390 | 0.463 | 0.549 | **0.622** | 0.762 | **0.848** |
| plain BM25 | — | — | — | — | **0.640** | — | — |

Three readings, and the third is the one that moves the next step.

1. **Candidate generation is not the bottleneck.** The router's own list holds the gold for **139 of
   the 164 (85%)**, at a median rank of 3 and a maximum of 16. Whatever the dense/lexical/entity RRF
   front end is doing, it is finding the evidence.
2. **The ranker is where the remaining distance lives.** The router's ordering reaches 0.622 against
   plain BM25's 0.640 -- a difference the paired test in the next subsection cannot resolve at this
   sample size -- and the 0.226 between what it returns at depth 5 and the 0.848 its own list already
   contains is lost entirely between "candidate" and "returned". That subsection says which stage of
   the pipeline the distance belongs to, and how much of it is real.
3. **The depth policy consults the ranking last.** The router selects the maximum of three contexts
   on **156 of 164** queries, and the returned set is those three followed by candidates up to depth
   5 -- so only two slots are left to the ranking. Of the 139 questions whose gold is in the list,
   **64** sit within those two slots (rank <= 2), **65** sit at rank >= 3 and are not among the
   selected three, and 10 sit at rank >= 3 but were selected anyway. This is the same over-selection
   the real conversations showed (68% and 42% of checkpoints selecting the cap of three); here its
   cost is visible, because the material displaced is a ranking that holds the answer.

The honest limit on the third reading: returning the candidate list's top five *instead* scores
0.622, below the **0.628** the current selection-plus-fill returns, and the three selected contexts
alone already answer 74 of the 164. The selection is not pure overhead, and widening the depth does
not by itself make the router win -- it moves the router to about lexical's level, not past it.

### Which stage loses the ordering

Every candidate carries the ranks of the three channels that produced it and the calibrated
probability that reordered them, so the loss can be attributed to a named stage instead of to "the
ranker". Ordering the **same** candidate list by each field in turn, on the 164 hard questions:

| order the candidate list by | recall@5 |
|---|---:|
| `lexical_rank` -- the lexical channel alone | **0.640** |
| `dense_rank` -- the dense channel alone | 0.378 |
| `entity_rank` -- the entity channel alone | 0.421 |
| `rrf_score` -- the fusion, before calibration | **0.567** |
| `calibrated_probability` -- the fusion after calibration | **0.628** |
| any five of the same list (its ceiling) | **0.848** |

The candidate list has a median of 17 members, and plain BM25 over all 2,810 windows scores 0.640.

**Those five numbers are one draw of 164 questions, and the differences among the top three are
smaller than that sample can resolve.** The comparison is paired -- the same questions, the same
candidate lists, only the ordering changes -- so it can be tested directly:

| paired, same 164 questions | difference | win / lose | exact McNemar p | bootstrap 95% |
|---|---:|---|---:|---|
| `calibrated_probability` - `lexical_rank` | **-0.012** | 9 / 11 | **0.824** | [-0.067, +0.043] |
| `rrf_score` - `lexical_rank` | **-0.073** | 13 / 25 | 0.073 | [-0.146, +0.000] |
| `calibrated_probability` - `rrf_score` | **+0.061** | 16 / 6 | 0.052 | [+0.006, +0.116] |

**None of the three clears p < 0.05**, and the first is nowhere near it: 9 questions won against 11
lost. It was quoted to three decimals because the rest of the table is quoted to three decimals, not
because 164 questions can resolve a thousandth. This project already applies that standard to the
paid judge -- a tally at n = 44, *reported as a direction only* -- and had not applied it here.

Three things follow, and the third is a correction to an earlier draft of this section.

1. **The front end loses nothing.** The lexical channel, restricted to the router's own ~17
   candidates, recalls **0.640** -- the same as BM25 over all 2,810 windows, to three decimals. If
   the RRF front end were dropping lexical hits that equality could not hold. Candidate generation
   is not what costs the router.
2. **The equal-weight fusion is the weakest ordering of the three.** Ordering the identical
   candidates by `rrf_score` drops to **0.567**, 0.073 below the lexical channel alone and 25
   questions lost to 13 won. RRF admits a channel whose own recall on this subset is 0.378 at the
   same weight as one whose recall is 0.640, and the sum comes out worse than the better channel by
   itself. This is the one comparison here whose bootstrap interval does not cross zero -- it stops
   exactly at it -- and it is the only row worth acting on.
3. **The combination is not demonstrably behind its own best channel.** `calibrated_probability`
   sits 0.012 below `lexical_rank`, at 9 questions won against 11 lost -- a coin. An earlier draft of
   this section read that as "calibration recovers the damage and not all of it", which is a finding
   the sample does not contain. **It is withdrawn.** The router's deficit on this benchmark is
   produced by its fusion step, and the calibration stage recovers it to the point where what is
   left cannot be measured here.

The correction matters more than its size. What this table supports is that **equal-weight fusion of
channels of unequal quality is a measurable defect** -- and `docs/fusion-weights-2026-10-02.md` on
the `channel-weighted-fusion` branch shows that repairing it changes nothing end to end: gating the
fusion on each channel's measured recall moves the router's own held-out score by zero. What the
table does not support is that the router returns an order worse than its best single channel, and
the earlier draft said that it did.

*Grade: exact, deterministic, offline -- one router pass over the 164 questions with the pinned
static encoder, no judge, no API. The recalls are exact-evidence containment and are not estimates;
**the differences between them are estimates, and the paired table above is their grade** -- a
20,000-resample bootstrap over questions and an exact McNemar on the discordant pairs. The
attribution is additionally **conditional on this benchmark's channel quality**: the dense channel is
far weaker than lexical here (0.378 against 0.640 over the same subset), which is a property of a
verbatim-evidence benchmark as much as of the encoder. "Equal-weight RRF costs the ordering" is
therefore established where that gap exists and is not claimed where it does not.*

Both halves are now first-class and reproducible from the CLI rather than from a one-off script.
`WindowRetriever.retrieve_with_decision` keeps the `RouteDecision` that `retrieve` discards, and
`routing_miss_decomposition` and `candidate_order_attribution` turn it into the two tables above.
`ctxlab scale-qa` prints both, under `router_decomposition` and `router_ordering` (schema 1.1), and
the hard subset they run on is the one `difficulty_strata` reports -- taken from a shared
`baseline_solved_mask`, so the split and the decomposition cannot disagree about which questions are
hard.

## Two incidental defects

**`oracle_router` is not a routing arm on real replay.** It selects `case.required_context_ids`,
which schema 2.0 deliberately leaves empty because the field is circular. So on a real conversation
it selects nothing, the builder falls back to the recent window, and the arm becomes a recency
baseline with a smaller budget -- reachability 0.080 at the default profile, rising with the recent
allowance exactly as `query_recent_only` does. Any arms table read on real data will show it as a
routing result. It is not one.

**`datasets/sanitized/d22f2593/benchmark.jsonl` is stale.** It is schema 1.0 and still carries
`required_context_ids`, the field the project removed as unlabelable. The loader refuses it
loudly, which is the designed behaviour, but the file is dead weight in the repository and should
be regenerated or removed.

## What this leaves

The null result is unchanged in size and changed in explanation. It is not that the router fails to
find the evidence; it finds more of it than the baseline does. It is not that the benchmark is
merely noisy; the benchmark cannot decide the question, and on the real conversation it labels
mostly the checkpoints recency already reaches.

Four things follow, and the last two are decisions rather than tasks.

1. `recent_fraction=0.50` is a free improvement on sea-ice (+0.069 reachability, +0 tokens), and it
   is also a **clean single-variable experiment**: the routing decision is identical on **44 of 44**
   sea-ice checkpoints and **26 of 26** d22f2593 checkpoints, at -0.1% and +1.2% tokens. Nothing
   about the router moves; only what the builder shows. A paid comparison of that profile against
   the default is therefore an ablation of *assembly* rather than a second routing strategy, and it
   is the cheapest quality experiment this project can run.
2. The frontier says reachability is bought with tokens, not with routing, on one conversation and
   lost by routing on the other. Whatever the next paid run compares, it should compare the
   *cheapest profile that holds reachability* against recency -- not the default profile.
3. The project's claim needs restating. "Routing cuts tokens by 221x and loses no quality" is
   supported. "Routing finds the evidence better" is supported on one conversation and reversed on
   the other. "Routing improves answers" is not supported and the benchmark cannot currently test
   it.
4. On the external benchmark the loss is now attributed rather than merely located, and bounded. The
   front end reaches the evidence for 85% of the questions that discriminate; the equal-weight fusion
   orders those same candidates 0.073 below the lexical channel it is fusing (13 won against 25
   lost); calibration buys back 0.061 and lands within noise of that channel (9 won against 11 lost,
   p = 0.82). The measurement is therefore complete enough to say **what the defect is**, and not
   large enough to rank the methods that repair it: **164 discriminating questions cannot separate
   these combinations**, so what can move this is more questions rather than a better fusion.
   `docs/fusion-weights-2026-10-02.md` records a fusion repair that measures as zero benefit for
   exactly that reason. Both diagnostics are permanent CLI output (`router_decomposition`,
   `router_ordering`), so the larger sample can be judged against the same two tables instead of
   against a pooled score.

```bash
# reproduce every number above, offline, no API key
ctxlab sweep-assembly run/events.sqlite run/dataset/benchmark.jsonl run/sweep.json \
    --include-recent-in-evidence

# the external benchmark, with its arrangement's reachability and its difficulty split
ctxlab scale-qa external/pkg_512k_1000 scaleqa.json --window-size 256 \
    --mode lexical --mode dense --mode hybrid --mode router --embedder neural
```
