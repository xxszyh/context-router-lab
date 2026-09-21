# The answer gate under a documented model

2026-09-20. The first answer-quality measurement on real conversations that rests on a
publicly documented model rather than a private proxy alias, and the first time a quality
measure has come out in routing's favour. Both of those need qualifying, and the qualifiers
are the substance of this document.

## What ran

| | |
|---|---|
| model | `kimi-k2.5` (MoonshotAI Kimi K2.5), served on `anthropic:messages`, ceiling 260,096 |
| arms | `query_recent_only`, `global_hybrid`, `hybrid_router`, `full_history` |
| checkpoints | 24 -- the whole labelled set, not the gate's 20 |
| answers | `max_tokens=8192`, 0 truncations, 0 failures on the three selective arms |
| judge | same documented model, swapped-pair protocol, `--limit 24` |

`full_history` is excluded from the judged comparison: it ran on **5 of 24** checkpoints and
failed the other 19 with HTTP 400 context refusals. See
`docs/documented-model-selection-2026-09-20.md` for why that is a result rather than a defect.

The judged pair is therefore `hybrid_router` vs `query_recent_only` -- relevance-selection
against recency-selection at comparable budget, the only pair where both arms run everywhere.

## The result

| | wins | losses | ties | order agreement |
|---|---:|---:|---:|---:|
| `hybrid_router` | **6** | 2 | 14 | 0.71 |
| `query_recent_only` | 2 | **6** | 14 | 0.71 |

23 pairs judged of 24 (one pair was tied by the refusal gate before any call). 0 parse failures.

By stratum: on the 21 answerable pairs `hybrid_router` takes 6-2; on the 2 `must_refuse` pairs it
takes 0-1. That second number is a single pair and carries nothing.

## What this does and does not say

**It says the direction now favours routing on a quality measure.** Every previous quality
measurement either put both arms at the floor (2026-09-18, labels written from the answer) or
favoured `full_history` (2026-09-19, proxy alias, and an arm that a documented model cannot run
anyway). This is the first one where the selective arm comes out ahead.

**It does not say routing preserves quality, and it does not give a magnitude.** Three reasons,
all visible above:

1. **14 of 23 pairs are ties.** The judge could not separate the arms in 61% of pairs. The
   comparison is not "routing answers better"; it is "routing answers no worse often enough that
   a judge finds a difference in a minority of cases".
2. **The judge disagrees with itself more than it did.** Order agreement fell to **0.71** from
   0.82 on 2026-09-19 -- the same protocol, a different model. On 8 decided pairs, 6-2 is not
   distinguishable from a coin (two-sided p ~ 0.29). The instrument is noisier than the effect.
3. **The strata are thin.** The `must_refuse` stratum, where a router's failure mode would show
   most clearly, is 2 pairs.

The honest statement is: **routing cuts tokens by roughly 221x, never fails to run where
`full_history` fails 79% of the time, and on the one comparison both arms can run, a blinded
judge finds routing ahead 6-2 with 14 ties -- a direction that the sample cannot turn into a
magnitude.**

## A defect in the instrument, found on the way

The judge is shown both arms' answers and reasons about them, so **its transcript quotes them
and inherits whatever they carry.** One answer in this run quotes an absolute path from the
original conversation, and the user name came through into one judge call.

Consequences, both now recorded rather than assumed:

- **A judge output is not automatically committable.** The 2026-09-19 judge output was recorded
  as "clean, committable"; it happens to have zero occurrences of the name, but that was luck
  rather than a property, and this run's output has one. Gate every derived artifact by value.
- **The 2026-09-20 judge output stays in `.local/`.** Consistent with the repository's rule that
  real records are never committed; only these numbers are.

The export gate also had a false-positive mode on serialised JSON that this run surfaced and
which is now fixed -- see the commit that added `assert_clean_artifact`.

## What would move this

The measurement is now possible under a documented model, which was the blocker. What it needs
is sample, not another re-run: the 6 labelled checkpoints that never entered a gate, and more
real conversations, raise the 8 decided pairs toward a number that can carry a direction.
