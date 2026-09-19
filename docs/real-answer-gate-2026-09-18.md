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

### The wording confound: tested, and refuted

The judge scores against requirements **quoted from the reference answer**, and `full_history`
is the arm that sees the most of the conversation those quotes came from. So the 3-0 could have
been a phrase-matching advantage rather than a quality difference. It was tested two ways.

**Offline first, on the run already in hand.** If the judge rewards wording, the arm it picks
should carry more of the requirement's quoted spans *literally*. It does — the four decided
pairs are almost exactly the pairs where `full_history` carries more of them (3 of 4, against 1
of 14 among the ties). But that is not evidence of a wording effect on its own, because a
quoted span can be a filename or a number, and an answer naming the right file *is* more
correct. Splitting the anchors settles which it is:

| anchor class | count | can a paraphrase remove it? |
|---|---:|---|
| content only (digit or ASCII identifier) | 20 (42%) | no |
| both kinds | 14 (29%) | no |
| **prose only** | **13 (27%)** | **yes — the only class a wording effect acts on** |

**Three of the four decisions carry at least one content anchor**; only `q-1370` rests on prose
alone.

**Then causally, on 16 calls.** The four decided pairs were re-judged twice: once with the
original requirements, and once with the prose anchors rewritten and the content anchors
(filenames, numbers, symbols, technical terms) left verbatim.

| run | result |
|---|---|
| control — original requirements | 4/4 → `full_history`, order agreement 4/4 |
| intervention — prose anchors reworded | **4/4 → `full_history`, order agreement 4/4** |

**Nothing moved.** The control reproduces the original run exactly, so the judge is test-retest
stable on these pairs and the null result is not noise. The paraphrased judge reasons about
claims and cites the numbers — "COMSOL auxiliary/verification, not main solver", "gives rounding
preference and 0.30×SE", "tens of seconds, far below 1090.5 s" — rather than matching the
wording it was handed.

**The confound is refuted. The 3-0 is not a phrase-matching artefact.**

## What the run is actually measuring, and why it is at the floor

The per-requirement credits are the number that matters most and the one nobody had counted.

| | |
|---|---:|
| requirement instances judged | 48 requirements × 17 pairs |
| credits the judge gave `hybrid_router` | **1** |
| credits the judge gave `full_history` | **9** |

**Both arms fail almost everything.** The 3-0 rests on a difference of eight credits out of
forty-eight, and the honest reading is not "full history is better" but **"neither arm
reproduces what the original answer contained, and full history reproduces slightly more of
it"**.

The reason is structural rather than about either arm. **The requirements were written from the
original assistant's answer**, which had the entire preceding conversation *and the user's
follow-up turns*. Neither a 1.7k-token selection nor a 370k-token history is that answer, and
the requirements encode its specific conclusions. That is what caps the ceiling here, and no
re-run of the judge can lift it.

### One more trap, for whoever runs this next

`ANTHROPIC_MODEL` in this environment is `deepseek-v4.1-flash[1M]`, where `[1M]` is Claude
Code's context shorthand and not part of any model id. Calling the endpoint with it returns
`400 模型不存在: deepseek-v4.1-flash[1M]`. The CLI needs `--model deepseek-v4.1-flash`
explicitly. This one fails loudly, so it costs two calls rather than a table of zeros — but it
is the same family as the budget trap and it is worth knowing before debugging a 400.

## What this establishes, and what it does not

**Establishes.** On 20 real checkpoints, routing to selected contexts cuts memory tokens by
about **221x** at a median, and `full_history` fails to run on **10%** of them while the
selective arms do not.

**Does not establish.** Any quality difference in either direction, and the reason is now
sharper than "underpowered". The judge credited **1** requirement to `hybrid_router` and **9**
to `full_history` across 48 requirement instances and 17 pairs — both arms fail almost
everything, so the four decisions are a difference at the floor, not a separation. The
requirements were written from the original answer, which had the full conversation *and* the
user's follow-ups; neither arm is that answer. **Re-judging cannot fix this** — the ceiling is
in the label set, not in the judge.

The wording confound, by contrast, **is** settled: refuted, causally, on 16 calls.

**Unpublishable regardless.** The model is a private proxy alias. Everything above is a
measurement of *this* endpoint on *this* day, and the numbers would move under a documented
model. The cost ratio is the part most likely to survive that change, because it is a token
count rather than a judgement.

## If this line continues

1. **Write the requirements from the query, not from the answer.** This is the binding
   constraint. "What must a correct answer to this query contain" is a different question from
   "what did this answer happen to contain", and only the first one has a ceiling above the
   floor. It is a labelling job, and it is the same shape as `answer_requirements` was — done by
   reading, no retrieval, no counterfactual — so the protocol in
   `docs/annotation-protocol.md` mostly carries over.
2. **Re-run under a documented model** — still the thing blocking publication.
3. **Report the executability failure as its own metric.** A 10% failure-to-run rate is a
   routing argument that no quality metric can express, and it is the part of this run that is
   already solid.
4. ~~Test the wording confound~~ — **done, refuted**; see above. Do not spend calls on it again.
