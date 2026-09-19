# The real-conversation answer gate, run and read

v0.2 work package E, run on 2026-09-18 and analysed the next morning. The run completed; the
analysis did not, so the numbers sat in `.local/` until now. Everything below is measured.

The result is **two-sided and should not be summarised as a win or a loss**:

- the **cost** half of the claim holds, and by a margin much larger than the synthetic result;
- the **quality** half is not measurable at this sample size, and the one instrument that
  produced a direction put `full_history` ahead.

## What ran

| | |
|---|---|
| model | `deepseek-v4.1-flash` — a private proxy alias, **not reproducible**; see the caveat below |
| arms | `query_recent_only`, `global_hybrid`, `hybrid_router`, `full_history` |
| checkpoints | 20 of the 26, the ones with an answer to compare against |
| calls | 80 planned, **78 completed** |
| output budget | 4096, chosen after the first round at 1024 failed as described below |
| memory budget | 2048 tokens for the selective arms |

## The first round was not a result

At `max_tokens=1024`, **72.2% of the 18 `full_history` replies hit the output cap and 61.1%
came back with no prose at all** — this model's thinking block shares the output budget and
runs long when the input is large. The selective arms were unaffected, because they had less to
think about. That asymmetry means the round measured the budget, not the arms, and it is kept
only as an audit of that failure.

At 4096 the problem largely goes away: one `full_history` reply still truncates and one still
comes back empty, both 1 in 18. **This is the third time this repository has paid for the same
lesson** — the thinking budget is not the answer budget — and the first time it was caught
before being written up.

## Cost: the claim holds, by about 220x

| arm | memory tokens | input tokens | output tokens |
|---|---:|---:|---:|
| `query_recent_only` | 1,204 | 1,461 | 339 |
| `hybrid_router` | **1,671** | 2,260 | 335 |
| `global_hybrid` | 2,004 | 2,795 | 334 |
| `full_history` | **370,102** (median) | 486,600 | 1,517 |

`full_history` memory tokens run 53,381 to 836,459 across the 18 checkpoints it completed.
Against `hybrid_router`'s 1,671 the median ratio is **221x**, and the input-token ratio is
**215x**. *Grade: exact — tokens are counted, not estimated.*

Output tokens are also 4.5x higher for `full_history`, because a larger context makes the model
reason longer. That is a cost the memory-token number does not capture and it should be
reported alongside it.

## `full_history` is not always executable

**Two of the twenty checkpoints were rejected by the server**: `q-2572` and `q-3070`, both
`400 InvalidParameter`, `Input exceeds the context limit (1048566 tokens)`. The three selective
arms succeeded on both.

This is not a truncation or a quality difference — the strategy could not be run. It is the
strongest single argument for routing in this dataset, and it is invisible in a quality tally
because there is no answer to judge. It belongs in the write-up as its own row:
**`full_history` has a non-zero failure-to-run rate at 10% here, and the router has 0%.**

## Quality: the two instruments disagree, and one of them is measuring wording

Deterministic coverage, all four arms:

| arm | coverage | refusal rate |
|---|---:|---:|
| `global_hybrid` | 0.050 | 0.45 |
| `query_recent_only` | 0.042 | 0.30 |
| `hybrid_router` | 0.042 | 0.30 |
| `full_history` | **0.037** | 0.17 |

**Every arm scores between 3.7% and 5.0%.** The metric has no signal at all on real replay —
and it ranks `full_history` *last*, which is the opposite of what the blinded judge found.

The reason is known and was measured before this run: the requirements quote the reference
answer verbatim, and **43% of them are anchored on prose rather than on a number or a filename**
(`docs/state-of-play.md`, step 1). An arm that answers correctly in different words scores near
zero. So this metric cannot separate the arms here, and the near-tie across all four is an
artefact of its construction rather than a finding about the arms.

**On synthetic data both instruments agreed on direction. On real replay they disagree.** That
is worth stating plainly: the agreement in `answer-quality-experiment.md` §4 was a property of
a dataset where the evidence ids are constructed, and it does not transfer.

## The blinded judge

18 pairs, `hybrid_router` vs `full_history`, A/B order swapped, refusal-gated pairs decided
deterministically without a call. 17 pairs reached the judge.

| | |
|---|---:|
| order agreement | **14 / 17 = 82%** |
| ties | 13 of 16 answerable pairs |
| tally, answerable only | `full_history` **3**, `hybrid_router` **0** |
| tally, must-refuse stratum | `full_history` 1, tie 1 |
| parse failures | 0 |
| judge cost | 23,114 in / 38,042 out tokens |

`full_history` won every decided pair. **Three decisions is not a result.** At the observed 82%
order agreement roughly one decision in five is positional noise, so the honest reading is that
the judge could not separate the arms on 13 of 16 pairs and, where it could, preferred
`full_history`.

### The confound that has to be checked before this is believed

The judge scores against requirements **quoted from the reference answer**. `full_history` is
the arm that sees the most of the conversation those requirements came from, so it is the arm
most likely to reproduce their wording and their facts. A 3-0 in that direction is therefore
consistent with a real quality difference *and* with the metric favouring the arm that saw more
of the source.

Reading the judge's own reasoning, it is scoring content rather than phrasing — "names both
files and run order", "τ₃ 变化仅数十秒，远小于 1090 s" — which argues against pure wording
bias. But that is an argument, not a measurement, and it is the first thing to test if this
line is continued.

## What this establishes, and what it does not

**Establishes.** On 20 real checkpoints, routing to selected contexts cuts memory tokens by
about **221x** at a median, and `full_history` fails to run on **10%** of them while the
selective arms do not.

**Does not establish.** Any quality difference in either direction. The lexical metric has no
signal; the judge has three decisions and 18% positional noise; n = 20; and the judge's target
is quoted from the arm most likely to resemble it.

**Unpublishable regardless.** The model is a private proxy alias. Everything above is a
measurement of *this* endpoint on *this* day, and the numbers would move under a documented
model. The cost ratio is the part most likely to survive that change, because it is a token
count rather than a judgement.

## If this line continues

1. **Re-run under a documented model** — the one thing blocking publication.
2. **Re-judge with requirements the judge cannot have been written from.** Either paraphrase the
   requirements away from the reference wording, or have the judge score against a rubric
   instead of the quoted spans. This is the confound above, and it is testable.
3. **Report the executability failure as its own metric.** A 10% failure-to-run rate is a
   routing argument that no quality metric can express.
4. **Raise n before reading the tally.** Three decided pairs at 82% order agreement cannot
   support a direction claim.
