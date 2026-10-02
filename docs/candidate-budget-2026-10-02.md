# The candidate budget was never the constraint, and neither was the front end

2026-10-02. Offline, deterministic, no judge, no API. `pkg_512k_3000` (3,000 questions, 3,561
windows at 256) and `pkg_168k_1000` (1,000 questions, 1,175 windows), window 256, depth 5,
`embedder=neural`.

The thread so far: the router's hard-subset miss rate was 0.246 on the 3,000-dialogue corpus and
0.131 on the 1,000-dialogue one, and `docs/composition-sweep-2026-10-02.md` showed that this tracks
the number of evidence-bearing dialogues rather than the amount of filler. That left "the front end
is the bottleneck" standing as the explanation. It is not, and this closes the thread in three
steps: the front end's failure is a budget artifact, the budget is cheap to widen, and widening it
**does not improve the answer at all**. The ranking stage was the constraint the whole time.

## Step 1: the miss is a property of two hand-set constants

`RoutingPolicy.top_per_retriever = 10` and `max_candidates = 20` are hand-set, unreported, and
decide how much of each channel's ranking reaches the router. The candidate generation was
reproduced analytically from the same building blocks `ContextRouter.route` uses, which is exact and
costs one pass over the grid instead of one per cell. **It reproduces the measured miss rate at the
defaults (0.246), which is the check that the replication is faithful.**

Candidate miss rate on the 602 hard questions, as both constants move:

| `top_per_retriever` | `cap=20` | `cap=50` | `cap=100` | `cap=1000` |
|---:|---:|---:|---:|---:|
| **10 (default)** | **0.246** | 0.246 | 0.246 | 0.246 |
| 20 | 0.213 | 0.143 | 0.143 | 0.143 |
| 50 | 0.233 | 0.095 | 0.068 | 0.068 |
| 100 | 0.261 | 0.098 | 0.051 | 0.030 |
| 200 | 0.239 | 0.088 | 0.045 | **0.018** |

The same list before the RRF cut -- is the gold in the union of the channels' top-N at all:

| `top_per_retriever` | 10 | 20 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|
| gold in the union | 0.754 | 0.857 | 0.932 | 0.970 | 0.982 |

And the retrievers themselves are almost never the limit: **the gold window carries a positive score
in at least one channel, at any depth, for 601 of the 602 hard questions (0.998)** -- 0.994 on the
1,000-dialogue package. The evidence is in the index. The router discards it before ranking.

Two details in that grid are worth naming. At `cap=20` raising `top_per_retriever` does not help, and
above 50 it makes things *worse* (0.213, 0.233, 0.261, 0.239): a bigger union means more gold
dropped by the same 20-slot cut. And at `top=10` raising the cap does nothing at all, because the
union is at most 20 items and the cut never binds. The two constants bind in different regimes and
the default sits where `top_per_retriever` binds.

## Step 2: the corpus effect disappears at a wider budget

| package | evidence dialogues | hard | miss at (10, 20) | miss at (200, 1000) |
|---|---:|---:|---:|---:|
| `pkg_168k_1000` | 1,000 | 168 | 0.131 | **0.012** |
| `pkg_512k_3000` | 3,000 | 602 | 0.246 | **0.018** |

The 0.115 gap that `docs/composition-sweep-2026-10-02.md` attributed to corpus size is **0.006** at
a wide budget. Corpus size mattered because a fixed budget has to cover more near-misses, not
because the retrievers find less. That is a correction to the previous round's mechanism, not to its
measurement.

## Step 3: and it buys nothing

So the list contains the gold for 98.2% of the hard questions instead of 75.4%. The end-to-end test
is whether the ranker returns it:

| on the same 602 questions | default (10, 20) | wide (200, 1000) |
|---|---:|---:|
| gold in the candidate list (the ceiling) | 0.754 | **0.982** |
| candidate list, median size | 18 | **338** |
| `rrf_score` ordering, recall@5 | 0.472 | 0.473 |
| `lexical_rank` ordering, recall@5 | 0.556 | 0.556 |
| **`calibrated_probability` -- what the router returns** | **0.550** | **0.528** |

Paired, wide against default: **-0.022** -- 37 questions won against 50 lost, exact McNemar
**p = 0.198**, bootstrap **[-0.051, +0.008]**.

**Giving the ranker 338 candidates, including the right one 98.2% of the time, does not improve its
top five and may slightly degrade it.** The interval crosses zero, so the honest statement is "no
improvement, with a small degradation not excluded" rather than "it hurts". Either way it is not the
fix, and the two constants should stay where they are.

That is the answer to the question the whole thread was circling. The front end was never the
constraint on quality. It was measured as one because the *only* thing that had been measured was
how often the gold appeared in the list -- and a list metric improves when you widen the list, while
the metric that matters does not.

## What this leaves

1. **The ranking stage is the bottleneck and it is now the only candidate left.** Three rounds
   proposed a different culprit -- the fusion, the corpus, the candidate budget -- and each was a
   hand-set constant or a corpus property standing next to the ranker. With all three measured, the
   ranker is what remains: the calibrated order sits below the best single channel (0.528 against
   0.556 wide, 0.550 against 0.556 narrow), and it is the only stage whose output gets worse when it
   is given strictly more information.
2. **The two constants are not worth changing, and that is a result.** `top_per_retriever` and
   `max_candidates` do cost candidate coverage, measurably and a lot. They do not cost quality.
   A reader who skips the end-to-end test would "fix" the front end and ship a regression risk for
   nothing.
3. **The generalisable move, third time in two days: sweep the hand-set constants around a stage
   before blaming the stage.** `rrf_weights`, the corpus, and now the candidate budget each looked
   like the explanation until it was varied. The ones that survived are the ones that are not
   constants.

### Incidental: the entity channel is inert under this adapter

`_entity_scores` scores a context by the entities it carries, and the adapter's `_window_context`
leaves `entities` empty -- 0 of 3,561 contexts carry one -- because filling them was measured to cost
eight points of recall (`derived_fields`). So the third channel contributes no candidates, and
`HeuristicContextRanker` still spends 1.70 of its linear form -- its largest single weight -- on
`entity`, a feature that is identically zero on this benchmark.

`candidate_order_attribution` reports it: ordering by `entity_rank` scores 0.252 with 18 candidates
and **0.003** with 338. The first number was an artifact of a short, id-ordered list; the second is
what a dead feature looks like. A fitted ranker would learn that weight is zero, which is an argument
for the fitted ranker that has nothing to do with its accuracy.

```bash
ctxlab scale-qa external/pkg_512k_3000 scaleqa.json --window-size 256 \
    --mode lexical --mode router --embedder neural
```

*Grade: the candidate-budget grid is an analytic reproduction of `ContextRouter`'s candidate
generation, validated by matching the router's own miss rate at the defaults (0.246) before any
cell of it was read. The end-to-end numbers are two router passes over the same 602 questions with
the pinned static encoder, and their difference is graded by exact McNemar plus a 20,000-resample
bootstrap. "Gold carries a positive score" means a BM25 or cosine score strictly above zero, so it
is a statement about visibility and not about retrievability at a useful magnitude; it is the
weakest form of the claim, and the claim holds in that form.*
