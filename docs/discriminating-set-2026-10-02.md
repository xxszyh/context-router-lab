# A bigger discriminating set, and what it settled

2026-10-02. Offline, deterministic, no judge, no API. Package `external/pkg_512k_3000`,
`arrangement=interleaved seed=20260420`, window 256, depth 5, `embedder=neural`.

`docs/recency-budget-2026-10-01.md` was corrected on the same day for quoting three ordering
differences without their intervals: at 164 questions the differences among the lexical channel, the
equal-weight fusion and the calibrated order were all smaller than the sample. Its own closing line
said what to do about that -- *what can move this is more questions rather than a better fusion*.
This is that, and it works better than expected.

## The package

Built with SCALE-QA's own builder, which is offline and bundles its noise. Its default
`--num-questions` is 3000; the 1,000-question package this project had been using was a deliberate
subset of it, and the 1,000 selected ids are a strict subset of the 3,000.

```bash
python build_eval_dataset.py --context-budget 512k --num-questions 3000 \
    --output-dir external/pkg_512k_3000
```

**It is a different difficulty regime, not only a larger sample**, and that is worth stating before
any number is compared across the two:

| | 1,000-question package | 3,000-question package |
|---|---:|---:|
| questions | 1,000 | 3,000 |
| stream | 445,373 tokens (26.7% truth) | 458,234 tokens (**77.9% truth**) |
| windows at 256 | 2,810 | 3,561 |
| lexical@1 already answers | 836 (83.6%) | 2,398 (79.9%) |
| **discriminating questions** | **164** | **602** |
| lexical recall@5, pooled | 0.941 | 0.911 |
| router recall@5, pooled | 0.931 | 0.903 |
| gold inside the router's candidate list | 0.848 | 0.754 |

The budget is the same 512,000 tokens, so tripling the truth leaves far less room for noise: the
3,000-question package is mostly evidence, and the absolute recalls fall accordingly. A result that
holds in both is a result about the method rather than about one corpus mix.

## What it settled

Paired, on the same questions and the same candidate lists, ordered five ways:

| ordering | 1,000-package, n = 164 | 3,000-package, n = 602 |
|---|---:|---:|
| `lexical_rank` | 0.640 | 0.556 |
| `rrf_score` | 0.567 | 0.472 |
| `calibrated_probability` | 0.628 | 0.550 |

| paired comparison | difference | win / lose | exact McNemar p | bootstrap 95% |
|---|---:|---|---:|---|
| `calibrated_probability` - `lexical_rank` | **-0.007** | 38 / 42 | **0.738** | **[-0.035, +0.023]** |
| `rrf_score` - `lexical_rank` | **-0.085** | 43 / 94 | **0.000** | **[-0.123, -0.048]** |
| `calibrated_probability` - `rrf_score` | **+0.078** | 61 / 14 | **0.000** | **[+0.051, +0.106]** |

Three things, and the third is why this was worth doing.

1. **The equal-weight fusion defect is established, not suggested.** At 164 questions the 0.073 gap
   to the lexical channel sat at p = 0.073 with an interval that stopped exactly at zero. At 602 it
   is 43 questions won against 94 lost, p < 0.001, interval [-0.123, -0.048]. Fusing three channels
   of unequal quality at equal weight costs the ordering, and that is now a measured fact about this
   benchmark rather than a direction.
2. **So is the calibration stage's recovery.** 61 won against 14 lost, +0.078, interval
   [+0.051, +0.106]. Whatever the fusion loses, the learned-looking stage that follows it gets most
   of it back, and that too is now established rather than asserted.
3. **The combination's distance from its best single channel is bounded, and it is small.** At 164
   questions this was -0.012 with an interval of width 0.110 -- a number that could not be
   distinguished from anything. At 602 it is -0.007, 38 against 42, p = 0.738, interval
   **[-0.035, +0.023]**. The calibrated order is within about three points of the best single
   channel in either direction, and the honest reading is the bound, not the point estimate: *the
   router's ordering is not worse than its strongest channel by more than ~0.035, and not better by
   more than ~0.023, on this benchmark.*

So the earlier correction was right to withdraw the 0.012, and it was wrong to imply the question
was unanswerable. It was answerable; it just needed four times the questions, and the answer is a
bound rather than a winner.

Both regimes agree on the direction of all three, which is the replication the single package could
not provide.

## One thing that did not replicate, and it was load-bearing

The misdecomposition itself changes character between the two packages. On the 1,000-question
package, 103 of the router's 164 hard questions were recovered, 37 were **selection losses** and
only 24 were **candidate misses** -- which is what supported the sentence *candidate generation is
not the bottleneck*. On the 3,000-question package the same decomposition is:

| on the discriminating subset | 1,000-package, n = 164 | 3,000-package, n = 602 |
|---|---:|---:|
| recovered | 103 (62.8%) | 331 (55.0%) |
| selection loss -- offered and not returned | 37 (22.6%) | 124 (20.6%) |
| **candidate miss -- never offered** | **24 (14.6%)** | **147 (24.4%)** |

**The candidate miss rate nearly doubles.** With 77.9% of the stream being evidence rather than
noise, the front end finds the gold in the candidate list for 75.4% of the hard questions instead of
84.8%, and failing to offer the evidence at all is now as large a share of the loss as dropping it.

So "candidate generation is not the bottleneck" is true of the package it was measured on and not of
the other one, and the honest version is narrower: *on a stream that is mostly noise, the front end
offers the gold and the loss is downstream of it; on a stream that is mostly evidence, the front end
is also a bottleneck.* Which one a reader should care about depends on which mix their own traffic
resembles -- and the 26.7%-truth package, having been built to fill a budget with noise, is not
obviously the realistic one.

**That reading is wrong, and it was corrected the same day.** The obvious suspect for the difference
was the evidence-to-filler ratio, and `docs/composition-sweep-2026-10-02.md` sweeps it: from 27% to
98% truth at a fixed question set the candidate-miss rate does not move (0.237 to 0.248), while from
1,000 to 3,000 evidence dialogues at a fixed truth share it goes 0.131 to 0.246. **The corpus decides
and the filler does not** -- a noise block is UltraChat and is easy to reject, while another
SCALE-QA dialogue is a near-miss. "Candidate generation is not the bottleneck" is a statement about
corpus size, and this document wrote it without one.

## The instrument, made first-class

The reason the correction was needed is that the report could print a difference without the test
that qualifies it. That is now impossible:

- `ordering_hits` is the primitive -- per-question exact-evidence hits for each ordering, plus the
  candidate list's ceiling, from **one** routing pass whatever the number of fields.
- `ordering_report` averages it into the table; `paired_ordering_comparison` pairs it into exact
  McNemar plus a bootstrap, over the same hits.
- `ORDERING_COMPARISONS` names the three head-to-head tests in the module rather than letting a
  report choose a flattering pair.
- `ctxlab scale-qa` prints the interval under every comparison and labels it: a difference whose
  interval crosses zero is reported as **"a direction, not a result"**, and one whose interval does
  not is reported as **"established"**. Both go into the JSON, so a report cannot quote the point
  estimate without carrying its test.

```bash
ctxlab scale-qa external/pkg_512k_3000 scaleqa.json --window-size 256 \
    --mode lexical --mode router --embedder neural
```

*Grade: recalls are exact-evidence containment, deterministic and offline. The paired tests are an
exact McNemar on the discordant pairs and a 20,000-resample bootstrap over questions; the bootstrap
is over the questions measured, so it bounds sampling error and not the choice of package. The two
packages differ in composition as well as size (26.7% against 77.9% truth), so the agreement between
them is evidence about the direction and not about the magnitudes. One question in the 3,000 still
has no single window holding all its evidence, which puts a ceiling of 0.9997 rather than 1.0 on
every arm.*
