# Pair-local features: a gain that survives every ablation, and coefficients that mean nothing

2026-10-02. Offline, deterministic, no judge, no API. `pkg_512k_3000` (3,000 questions, 3,561
windows) and `pkg_168k_1000`, window 256, depth 5, `embedder=neural`,
`RoutingPolicy(top_per_retriever=200, max_candidates=1000)`.

`docs/ranker-ceiling-2026-10-02.md` left one architecture question: the features are the limit, not
the model form. There is a structural reason to expect that, and it is worth stating before the
numbers, because it is what the experiment was built to test.

**Every one of the router's nine features is set-normalized.** `_normalize` divides by the maximum
over whatever candidate list it is given, so none of them can express *"this candidate contains the
query's rare identifier"* -- a fact that does not depend on what else is in the list. A feature that
cannot represent that cannot rank by it.

## Design

Eight pair-local features, computed jointly over the query and the candidate text and independent of
the candidate set by construction:

| feature | what it is |
|---|---|
| `query_coverage` | share of the query's distinct terms present in the candidate |
| `idf_coverage` | the same, weighted by inverse document frequency |
| `rare_hits` | query terms with document frequency <= 5 that are present |
| `max_idf_present` | the rarest query term the candidate contains |
| `bigram_hit` | an adjacent query pair appears adjacently in the candidate |
| `identifier_overlap` | identifiers (`Aegis-7`, `#402`) shared with the query |
| `candidate_tokens` | length prior |
| `max_query_tf` | how often the candidate repeats the query's most frequent term |

One wide-policy pass stores **202,490 candidate rows** -- every candidate's nine router features, its
eight pair features and its label -- so both models see identical rows. Questions split 40/20/40 into
trainer, calibrator and evaluation. **The same estimator is used for both feature sets**
(`StandardScaler` + `LogisticRegression(class_weight="balanced")`), so a difference between them
belongs to the features and not to the fitting.

## Six seeds

Held out, 242 questions each, median candidate list 336:

| seed | lexical order | shipped heuristic | logreg, 9 router features | logreg, 9 + 8 pair | any five of the list |
|---|---:|---:|---:|---:|---:|
| 20260420 | 0.521 | 0.517 | 0.554 | **0.587** | 0.971 |
| 1 | 0.554 | 0.521 | 0.545 | **0.583** | 0.992 |
| 7 | 0.533 | 0.508 | 0.558 | **0.587** | 0.983 |
| 2026 | 0.554 | 0.508 | 0.533 | **0.562** | 0.983 |
| 12345 | 0.579 | 0.525 | 0.570 | **0.620** | 0.975 |
| 999983 | 0.570 | 0.504 | 0.533 | **0.595** | 0.988 |
| **mean** | **0.552** | **0.514** | **0.549** | **0.589** | **0.982** |

Paired, per seed: the pair model beats the shipped heuristic in **6 of 6**, every one of them
established (p between 0.0000 and 0.0072). It beats plain lexical order in **6 of 6** on the point
estimate, but only **1 of 6** individually clears p < 0.05 -- the first seed tried, at p = 0.0195,
which is what a mean difference of +0.037 looks like at n = 242.

**That first seed is the reason this document has six.** A single split would have read as a
demonstrated win over BM25's own order. It is not one; it is a consistent direction.

## The ablation, which is the actual finding

Re-fitting with named pair features removed, same seeds and same folds:

| variant | features | recall@5, mean |
|---|---:|---:|
| the router's nine | 9 | 0.549 |
| all eight pair features added | 17 | 0.589 |
| keep only `candidate_tokens`, `max_idf_present`, `idf_coverage` | 11 | 0.592 |
| drop `candidate_tokens` (5 seeds) | 16 | 0.582 |
| drop `max_idf_present`, `idf_coverage` (5 seeds) | 15 | 0.594 |
| drop all eight (6 seeds) | 9 | 0.549 |

Three things, and the first is the check that the rest is real.

1. **Dropping all eight reproduces the nine-feature model exactly** (0.554, 0.545, 0.558, 0.533,
   0.570, 0.533), which is what says the column filtering is doing what it claims.
2. **The gain survives every subset tried.** +0.040 for all eight, +0.043 for the three that
   sound like a length prior and a rarity prior, +0.033 for the five that sound like nothing in
   particular, +0.045 with the length prior removed. Different subsets, same place. So the effect
   belongs to *having set-independent pair information in the model* and **not to any named
   feature** -- which is a weaker and more durable claim than the one the coefficients suggested.
3. **The coefficients were actively misleading.** Across seeds the largest interpretable
   coefficients sat on `max_idf_present` (+0.38 to +0.50) and `idf_coverage` (+0.10 to +0.41), in a
   pattern stable enough to look like a mechanism -- and removing both changes the mean by
   **+0.005**, in the wrong direction. `candidate_tokens` was the other large one (-0.61 to -0.72)
   and removing it costs 0.007. In a correlated feature set, coefficient magnitude is not
   importance, and six seeds of stability do not make it importance.

The one interpretable thing that survives is a sign pattern rather than a magnitude:
`query_coverage` carries a **negative** coefficient (-0.19 to -0.42) while `idf_coverage` is
positive, consistently. Matching *more* of a query is not the same as matching the *informative*
part of it, and a plain coverage feature rewards long generic candidates. That is worth knowing
whether or not it is doing any work here -- and on this evidence it is not.

## Cross-corpus check

`pkg_168k_1000`, 168 hard questions, 54,181 rows, one seed: lexical order 0.691, shipped heuristic
0.647, logreg over nine features 0.721, with the pair features **0.750**, and every question's gold
in the candidate list (**1.000**). The same ordering of the four, on a corpus a third the size; at
n = 168 no comparison is individually established (8 to 13 discordant pairs).

## What this says

**The ranker's features can be improved, by about +0.04, and the improvement is not attributable to
anything in particular.** That is a real result and a modest one. Two readings follow, and the
second is the one the project should act on.

1. **A fitted ranker beats a hand-set one**, and that is the only comparison in this document that
   is established everywhere it was measured: 0.549 against 0.514 over six seeds, and 6 of 6
   against the shipped heuristic for the augmented model. Nine features and a sigmoid are not
   obviously better than sorting by one channel; a fit is.
2. **The ceiling is still 0.982.** Every ordering tried lands between 0.51 and 0.59 against a list
   that contains the answer 98% of the time. The feature engineering above claims about 0.04 of the
   0.43 that is missing, and the ablation says the exact form of it does not matter. What remains is
   a model that sees the query and the candidate *together* in a way no sum of per-pair scalars
   reproduces -- which is an architecture, not a feature list.

```bash
# build the rows once, then re-fit per split seed in seconds
python .scratch/pair_features.py --package external/pkg_512k_3000 --seed 20260420
python .scratch/pair_features.py --package external/pkg_512k_3000 --seed 1
python .scratch/pair_features.py --package external/pkg_512k_3000 --seed 20260420 --drop-pair candidate_tokens
```

*Grade: exact, deterministic and offline. Every row is a feature vector the router actually
computed, plus eight scalars computed from the same query and candidate text; the four orderings
differ only in their comparison function. The paired tests are exact McNemar on the discordant pairs
and a 20,000-resample bootstrap over questions. **The estimator is one fit per fold per seed, with no
tuning and no nested cross-validation; 361 positives over 121,086 rows is a regime where a 17-feature
linear model is expected to be unstable, and the ablation shows that instability directly.** The
six seeds and the second package bound the direction; they do not bound the size, and the honest
summary of the size is "between +0.03 and +0.05, on this benchmark, for this estimator".*
