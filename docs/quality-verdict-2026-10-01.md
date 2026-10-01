# The quality claim, after both conversations and a better instrument

> **Superseded, same day, by [`judge-anchoring-2026-10-01.md`](judge-anchoring-2026-10-01.md).**
> Every tally below was produced by the `terse` judge prompt. Anchoring the judge in the text
> it is judging took `mean_consistency` from 0.797 to 0.891 on the published conversation and
> 0.853 to 0.864 on sea ice, and independent hand reads agree with the anchored verdicts on
> 8 of 8 pairs against 2 of 8 for `terse`. Under it the published conversation is **9-3 rather
> than 7-5** and the pool is **18-17, p = 1.00, rather than 16-22, p = 0.42**.
>
> **The null result below is therefore stronger than this document could state, not weaker.**
> The rest of the page is kept as written, because the instrument comparison it makes — one
> repeat against three, and why the one-repeat numbers were the ones being reported — is
> unaffected and is the reason the anchored run was possible at all.

2026-10-01. Every judge run in this project has now been repeated, both conversations have been
measured with the same instrument, and the result is not the one the project has been reporting.

## The four runs

| conversation | repeats | `hybrid_router` | `query_recent_only` | ties | decided | p | consistency |
|---|---:|---:|---:|---:|---:|---:|---:|
| published `d22f2593` (24 ckpt) | 1 | **6** | 3 | 15 | 9 | 1.00 | -- |
| published `d22f2593` (24 ckpt) | **3** | **7** | **5** | 12 | 12 | **1.00** | 0.797 |
| sea ice `139471e7` (44 ckpt) | 1 | 8 | 11 | 25 | 19 | 0.65 | -- |
| sea ice `139471e7` (44 ckpt) | **3** | 9 | **17** | 18 | 26 | 0.17 | 0.853 |
| **pooled, 3 repeats** | | **16** | **22** | 30 | 38 | **0.42** | |

Only the three-repeat rows are comparable to each other. The one-repeat rows are the numbers this
project has been reporting, and they were produced by an instrument that turns order
disagreement into ties -- 42% of those ties were the judge contradicting itself.

## What changed, and why it is not a change of method

**The published conversation's lead did not survive.** 6-3 became 7-5, and at twelve decided pairs
that is as even as an even number allows: p = 1.00. The result reported on 2026-09-20 as "the first
quality measure to favour routing" is, under the better instrument, indistinguishable from a coin.

**The second conversation's lean toward recency sharpened.** 8-11 became 9-17, p = 0.17.

**Pooled, `query_recent_only` leads 22-16**, which is p = 0.42 -- not significant, and leaning the
way the routing claim does not want.

The repetition was validated separately before any of this was read: it recovered seven collapsed
ties and **flipped no decision** in the pairs both runs decided. So the movement is the removal of
noise, not a different question being asked.

## The honest statement

Not "routing cuts tokens by 221x and a blinded judge finds it ahead". That was true of one run of
one conversation at one repeat, and it is not true now.

What holds:

- **routing cuts the tokens the main model reads by roughly 221x** on real conversations, counted;
- **`full_history` fails to run on 79% of checkpoints** against a documented model's context
  ceiling, where the selective arms never fail;
- **on answer quality, a blinded judge cannot distinguish relevance-selection from recency-selection**
  -- 38 decided pairs across two conversations at p = 0.42, and the lean, such as it is, is toward
  recency.

The first two are unaffected by any of this and remain the project's solid ground. The third is
now a null result rather than a positive one, and it should be reported as such.

## What would move it, and what would not

Not more checkpoints from these two conversations. The pooled 38 decided pairs at p = 0.42 would
need roughly three times the sample to resolve a lean this size, and the two conversations
disagree in magnitude if not in direction.

What might: a judge whose `mean_consistency` is above 0.80 on every conversation rather than on
one of them. The published conversation scored 0.797 against the second's 0.853, and the
conversation with the weaker instrument is the one whose result moved most under repetition. The
instrument is still the binding constraint, and it is now measurable per run.
