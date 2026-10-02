# The features are the limit, not the model form

2026-10-02. Offline, deterministic, no judge, no API. `pkg_512k_3000`, window 256, depth 5,
`embedder=neural`, `RoutingPolicy(top_per_retriever=200, max_candidates=1000)`.

`docs/candidate-budget-2026-10-02.md` left one stage standing and one number on the table. Over a
rich candidate list the gold is present for 98.2% of the hard questions, the list's own ceiling at
depth 5 is 0.982, and the router returns 0.528. The ranking stage leaves about 0.43 unclaimed, and
that gap can be closed on either side of the model:

- the **form** is wrong -- a fixed linear combination over nine normalized features, whose largest
  weight (1.70) sits on `entity`, a feature that is identically zero under this adapter; or
- the **features** do not contain what an order would need.

A fitted ranker over the *same* features separates the two, and this project already has one.

## Design

One wide-policy pass over the 602 hard questions, storing every candidate's feature vector, label
and ranks -- **202,490 rows** -- so that every ordering below is evaluated over the identical rows
rather than over four separate retrievals. Questions are split 40/20/40 into trainer, calibrator and
evaluation, and `PlattContextRanker.fit` sees the first two only.

| fold | questions | rows | positives |
|---|---:|---:|---:|
| train | 240 | 80,834 | 239 |
| calibration | 120 | 40,252 | 122 |
| holdout | 242 | 81,404 | -- |

## Result

Held out, on 242 questions whose candidate list has a median of 336 members:

| ordering of the same 336 candidates | recall@5 |
|---|---:|
| `rrf_score` -- the fusion, before any ranker | 0.459 |
| `heuristic` -- `HeuristicContextRanker`, what ships | 0.517 |
| `lexical_rank` -- plain BM25 order | 0.521 |
| **`fitted` -- `PlattContextRanker` over the same nine features** | **0.541** |
| a perfect pick of any five from the list | **0.971** |

Paired, on the same questions:

| comparison | difference | win / lose | exact McNemar p | bootstrap 95% |
|---|---:|---|---:|---|
| fitted - lexical_rank | +0.021 | 26 / 21 | 0.560 | [-0.033, +0.074] |
| fitted - heuristic | +0.025 | 9 / 3 | 0.146 | [+0.000, +0.054] |
| heuristic - lexical_rank | -0.004 | 29 / 30 | 1.000 | [-0.066, +0.058] |

## What that says

**Fitting moves the ordering in the right direction and does not establish anything at n = 242.**
The fitted ranker is the best of the four and beats the shipped one by 9 questions to 3, but twelve
discordant questions is not a measurement; the interval stops exactly at zero.

**And it does not matter, because the ceiling is 0.971.** Even the best ordering available reaches
**0.541** against a list that contains the answer 97% of the time. Fitting the model form -- the
change that would fix "the form is wrong" -- is worth about 0.02 of the 0.43 that is missing. The
remaining 0.41 is not reachable by any recombination of these nine features, by hand or by fit.

The precise statement, because the loose one is wrong: **the evidence is present in the candidate
list but not separable from it.** Being in the list means some channel scored it above zero, which
is a very weak property. It does not mean anything in the router's feature vector distinguishes that
window from the three hundred others.

Two supporting readings:

1. **The shipped ranker is no better than trusting BM25's order.** 0.517 against 0.521, 29 wins
   against 30 losses, p = 1.00. Nine features, a hand-set linear form and a sigmoid reproduce what
   sorting by the lexical channel alone already did. This is the same conclusion
   `docs/discriminating-set-2026-10-02.md` reached on a narrow candidate list, now on a rich one
   where the ranker has far more to work with.
2. **The fusion is the worst of the four orderings** again (0.459), consistent with every previous
   measurement: equal-weight RRF of channels of unequal quality costs the order, and the stage after
   it recovers most of that.

## What this rules out, and what it leaves

**Ruled out:** another ranker over these features. Re-weighting them, fitting them, adding a third
channel, or calibrating the output differently are all changes to a function whose input does not
contain the answer. The measured headroom for that entire family is about 0.02.

**Left open, and it is an architecture question rather than a tuning one:** a model that sees the
query and the candidate *text* together -- a cross-encoder, or features computed jointly over the
pair rather than each side summarized separately. Nothing in this project has ever tried that, and
the ceiling of 0.971 says the candidates are good enough to justify trying.

**Also left open, and cheaper:** the two channels that produce the candidates are a 2,810-document
BM25 index and a small static encoder. The dense channel's own order scores 0.297 to 0.378 against
lexical's 0.556, so the candidate list is overwhelmingly lexical. Whether a real transformer encoder
changes the *separability* of the list -- not the presence of the gold in it -- has not been
measured.

*Grade: exact, deterministic and offline. Every row is a feature vector the router actually
computed, evaluated by sorting the identical stored rows, so the four orderings differ only in their
comparison function. The paired tests are exact McNemar on the discordant pairs and a 20,000-resample
bootstrap over questions. With 242 held-out questions and 12 discordant pairs on the headline
comparison, no difference below about 0.05 should be read as a difference -- which is why the finding
here is the ceiling, and not the ordering of the four rows.*
