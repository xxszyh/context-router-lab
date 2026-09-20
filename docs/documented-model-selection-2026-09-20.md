# Choosing a documented model, and what the choice costs the gate

2026-09-20. The 2026-09-19 judge run settled that the query-derived labels work and left the
direction favouring `full_history` 5-2 at n = 18 -- under `deepseek-v4.1-flash`, a private proxy
alias that the gateway resolves to `deepseek-v4-1-flash-260910`. Nothing publishable can rest on
an alias, so the next step named in `docs/answer-gate-judge-2026-09-19.md` was to re-run the
whole gate under a model with a public, stable name.

Selecting that model turned out to be a measurement, not a lookup, and the measurement changed
the experiment's design. This records both.

## The open question this inherited

A probe of the documented candidates left one result unexplained: `qwen3-max` answered a
~50,000-token prompt reporting `input_tokens=13`, and a ~130,000-token prompt reporting
`input_tokens=100`, where `kimi-k2.5` reported 66,676 and 106,784 for the same inputs.

Two explanations fit, and they have opposite consequences:

- the model reads everything and only the usage report is broken -- costs are mis-accounted,
  answers are fine;
- the prompt is silently truncated before the model sees it -- every answer is generated from a
  fragment, which would look in the results exactly like "this arm is cheap and bad".

Usage numbers cannot separate these. Only what the model can *see* can.

## The needle test

A unique marker is planted in a long prompt and asked for back. A model that read the whole
prompt returns it; a model that kept only part of it cannot. Marker at the **start** failing
while marker at the **end** passes would mean head truncation -- but both passing is also what
head+tail retention looks like, so the marker is also placed at depth.

| model | 50k start/end | 130k start/end | 130k at 25/50/75% depth |
|---|---|---|---|
| `qwen3-max` | found / found | found / found | found / found / found |
| `kimi-k2.5` | found / found | found / found | found / found / found |

**No truncation, at any depth, in either model.** The `input_tokens=13` reading was a broken
usage report on the gateway -- a real defect, but an accounting one. It means token counts from
these models must not be used for cost claims; it does not disqualify them as answerers.

## The ceilings, read off the gateway's own refusals

| model | documented name | ceiling | protocol |
|---|---|---|---|
| `qwen3-max` | Qwen3 Max | 258,048 | `anthropic:messages` |
| `kimi-k2.5` | Kimi K2.5 | 260,096 | `anthropic:messages` |
| `minimax-m3` | MiniMax M3 | ~600k-900k (refused at 900k) | `anthropic:messages` |
| `deepseek-v4.1-flash` | *none -- proxy alias* | 1,048,566 | `anthropic:messages` |

## `minimax-m3`, the one candidate that advertises enough room

`minimax-m3` is the only documented candidate whose advertised context reaches the range the
`full_history` arm needs, so it was tested the same way rather than taken at its word.

The first pass used `max_tokens=64` and returned five empty answers. That is not a model verdict:
this project's own error log records that a thinking block shares the output budget, so a small
cap leaves nothing for the text -- and the two models that passed at 64 tokens are non-reasoning,
which is why the cap was harmless for them. Re-run at `max_tokens=2048`:

| prompt | depth | result |
|---|---|---|
| 20,000 | 50% | **found** -- the task is within its ability |
| 500,000 | 5% | lost -- returned a *garbled* 11-character near-miss of the 14-character marker |
| 500,000 | 50% | lost -- returned an empty content block, `stop_reason=end_turn` |
| 500,000 | 95% | lost -- returned raw filler text |

At 500k it half-sees, empties, or echoes filler. A model that garbles a planted marker will
garble a conversation, so its advertised 1M is not usable for the arm that needs it, and it is
not a candidate.

## What this costs the gate, and why that is the result rather than a problem

`full_history`'s input tokens in the 2026-09-18 run: median **485,966**, max **1,042,473**, over
18 records. Against the documented ceilings:

| ceiling | `full_history` can run |
|---|---:|
| 1,048,566 (the proxy alias) | 18/18 |
| ~600k (`minimax-m3`, and it fails fidelity there) | 12/18 |
| 258,048-260,096 (`qwen3-max`, `kimi-k2.5`) | **5/18** |

Under any documented model that can actually be trusted, the arm that served as the quality
reference runs on **5 of 18** checkpoints. The quality comparison does not merely lose power --
on 13 of 18 conversations there is nothing to compare, because one arm cannot be run at all.

This is the same finding the cost half of the project already rests on, arriving from a different
direction. The 2026-09-18 run reported it as "`full_history` fails to run 10% of the time, the
selective arms 0%", measured against a 1M model that was itself generous. Measured against the
context windows that publicly available models actually serve, the number is **72%**. The claim
is not that routing is cheaper than reading everything; it is that at a documented model's
ceiling, reading everything is not an available strategy.

## The design that follows

The comparison the gate should be powered on is not `hybrid_router` vs `full_history`, which
cannot be run. It is:

- **`hybrid_router` vs `query_recent_only`** -- relevance-selection against recency-selection at
  the same budget. Both run on every checkpoint, so this pair is fully powered at n = 24, and it
  is the comparison a deployed router actually faces: not "should I select" but "should I select
  by relevance or by recency".
- **`full_history`** reported on the checkpoints where it survived, with its n stated, as an
  anecdote. It is not evidence at 5 pairs and must not be presented as such.

Arms and `max_tokens` are unchanged from 2026-09-18 so the two runs are comparable; the only
variable is the model.

## Still not established

Whether relevance-selection preserves answer quality. This document removes the model-alias
blocker and reshapes the comparison; it does not produce a quality verdict. The blinded judge on
the documented-model run is the measurement that would, and it has not been run.
