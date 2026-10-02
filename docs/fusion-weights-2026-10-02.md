# Channel weights, and a premise that did not survive being tested

2026-10-02, branch `channel-weighted-fusion`. Offline, deterministic, no judge, no API. Package
`external/pkg_512k_1000`, `arrangement=interleaved seed=20260420`, window 256 (2,810 units), depth
5, `embedder=neural` (`minishlab/potion-base-8M`, CPU only).

This branch was opened to test one repair. The repair is small and the test refuted the reason it
was proposed, so most of what follows is a correction.

**The code this describes lives on the `channel-weighted-fusion` branch and not on `main`** --
`RoutingPolicy.rrf_weights`, `WindowRetriever(policy, ranker)` and `held_out_fusion`. This file is
the record of what that code measured, kept on `main` because `docs/recency-budget-2026-10-01.md`
and `docs/discriminating-set-2026-10-02.md` both cite it.

## What the previous round claimed, and what is actually true

`docs/recency-budget-2026-10-01.md` closed with an attribution table over the 164 questions a cheap
lookup does not already answer, ordering the router's own candidate list by each field in turn:

| ordering of the same candidates | recall@5 |
|---|---:|
| `lexical_rank` -- the lexical channel alone | 0.640 |
| `dense_rank` | 0.378 |
| `entity_rank` | 0.421 |
| `rrf_score` -- the fusion, before calibration | 0.567 |
| `calibrated_probability` -- what the router returns | 0.628 |

and read the last row as "calibration recovers most of the damage, and not all of it -- 0.012 below
simply using the lexical channel over the same list". That reading is now known to be too strong.
The comparison is **paired**: the same 164 questions, the same candidate lists, only the ordering
changes. Paired, the difference is 9 questions won against 11 lost:

| paired comparison | difference | win / lose | exact McNemar p | bootstrap 95% |
|---|---:|---|---:|---|
| `calibrated_probability` - `lexical_rank` | **-0.012** | 9 / 11 | **0.824** | [-0.067, +0.043] |
| `rrf_score` - `lexical_rank` | **-0.073** | 13 / 25 | 0.073 | [-0.146, +0.000] |
| `calibrated_probability` - `rrf_score` | **+0.061** | 16 / 6 | 0.052 | [+0.006, +0.116] |

**None of the three clears p < 0.05 at n = 164, and the first one is nowhere near it.** The router's
calibrated ordering is not demonstrably behind its own strongest channel; it is not demonstrably
ahead either. The 0.012 was reported to three decimals because that is how the rest of the table is
reported, not because 164 questions can resolve it -- and the same standard this project already
applies to the paid judge ("a tally at n = 44, reported as a direction only") had not been applied
here.

The middle row is the one that survives, and only just: the **equal-weight fusion is the worst of
the three orderings**, 25 questions lost to 13 won against a single channel, at p = 0.073. That is
the defect worth repairing. The claim that the *combination* falls below its best member is
withdrawn.

## The repair, and the rule it was supposed to satisfy

The premise was: *a fusion that cannot beat its own strongest member should not be fusing.* The
repair is to stop hand-setting the fusion's weights and set them from a **measurement of each
channel taken on data the evaluation does not use**.

`RoutingPolicy.rrf_weights` makes the fusion weights expressible. `None` -- the default -- is
`1.0 / (k + rank)` summed in the same order as before, **bit-identical**, which is what makes the
knob safe to add to a pipeline whose numbers are already published;
`tests/test_router.py::test_naming_every_channel_at_weight_one_is_the_default_fusion_exactly` pins
that, and all 255 previously passing tests still pass. A channel left out of the mapping
contributes nothing, so switching one off is spelled by omission rather than by a zero.

`WindowRetriever(policy=..., ranker=...)` lets a caller ask what a *different* fusion or a *fitted*
ranker would have done over the identical unit set. `held_out_fusion` fits every lever on a training
fold -- channel weights by measured recall, the ranker by `PlattContextRanker.fit` -- and reports
only on a held-out fold.

The gate is the only free number in it, and it is registered in advance rather than tuned: **a
channel is kept if its own recall on the training fold is within 0.05 of the best channel's.** The
holdout fold is never consulted to set it.

### What was predicted before the numbers

- **P1** measured channel weights beat equal-weight RRF on held-out questions.
- **P2** measured channel weights beat the hand-set ranker on held-out questions.
- **P3** a ranker fitted on the training folds beats the best single channel on held-out questions.

P3 is the one that decides whether the fix is *gating* or *learning*.

## The result

Split: 400 train / 200 calibration / 400 holdout, of which **66** are questions the lexical baseline
does not already answer at depth 1.

Channel recall measured on the **training** fold, by ordering the router's own candidates:

| channel | train recall@5 |
|---|---:|
| lexical | 0.927 |
| dense | 0.703 |
| entity | 0.430 |

At tolerance 0.05 only `lexical` survives the gate: `{'lexical': 1.0, 'dense': 0.0, 'entity':
0.0}`. Held out:

| ordering | holdout hard, n = 66 |
|---|---:|
| `lexical_rank` -- the best single member | 0.682 |
| `rrf_score` -- equal weight | **0.621** |
| hand-set ranker -- the current default | **0.712** |
| gated fusion + hand-set ranker | **0.712** |
| fitted ranker (`platt-e63075cd383f`) | 0.697 |

**P1 holds.** Gating recovers the fusion's loss and more: 0.712 against the equal-weight fusion's
0.621, and the gate was set from the training fold alone.

**P2 fails.** Gating changes the held-out number by nothing at all: 0.712 either way. The repair
does not improve on what the router already does.

**P3 holds, weakly.** A ranker fitted on 600 questions it never saw beats the best single channel
on the 66 it is judged on, 0.697 against 0.682. So on this benchmark the combination *is* worth
trusting -- which is the opposite of the premise the branch was opened on. A 0.015 difference at
n = 66 is not something this document claims either.

And the sign of the head-to-head flips between the full set and this fold: on all 164 hard
questions the hand-set ranker is 0.012 behind the lexical channel, on 40% of them it is 0.030
ahead. That is what the paired test above already said: the difference is smaller than the sample.

## What this leaves

Three statements, in descending strength.

1. **Equal-weight fusion of channels of unequal quality is a real defect on this benchmark and it is
   measurable.** Raw fusion 0.567 against the best member's 0.640, and on a training fold where the
   best channel reaches 0.927 the other two reach 0.703 and 0.430 -- a spread that equal weight
   cannot express by construction. This is the finding, and it does not depend on the holdout.
2. **Repairing that defect changes nothing end to end.** The router's calibrated order already
   matched the best single channel, and gating leaves its held-out score unchanged. A
   reasonable-sounding repair with no measured benefit is a result, and it is cheaper to record
   than to ship.
3. **The premise behind the branch is withdrawn.** "A fusion that cannot beat its best member should
   not fuse" is not triggered here once calibration is included. It was triggered by the raw
   fusion, which is not what the router returns.

So `rrf_weights` stays in the branch as a knob with a tested default, and the finding is a
correction rather than a feature. The next thing worth measuring is not a better fusion; every
combination tried here lands between 0.62 and 0.71 on 66 questions, and the interval on a
comparison of that size is wider than the spread between them. It is the **size of the hard
subset**: 164 questions cannot separate these methods, and the honest next step is either more
questions or a benchmark whose discriminating part is larger than 16%.

```bash
# the design and its holdout, offline, no API key
ctxlab scale-qa external/pkg_512k_1000 scaleqa.json --window-size 256 \
    --mode lexical --mode router --embedder neural
```

*Grade: every recall number is exact-evidence containment, deterministic and offline. The paired
tests are exact McNemar on the discordant pairs plus a 20,000-resample bootstrap over questions;
the holdout comparison is a single seeded split (seed 20260420) and is therefore one draw, not a
distribution. The holdout's 66 hard questions carry a standard error of roughly 0.06, so no
difference below about 0.12 in that table should be read as a difference.*
