# The routing advantage did not replicate, and pooled it is exactly even

2026-10-01. The 2026-09-20 result -- a blinded judge putting `hybrid_router` ahead of
`query_recent_only` 6-3, the first quality measure in this project to favour routing -- was run
again on a second conversation with the same model, the same protocol and the same pair. It
reversed. Pooled across both, the two arms are level.

## The two runs

| | `hybrid_router` | `query_recent_only` | ties | order agreement | pairs |
|---|---:|---:|---:|---:|---:|
| published `claude-d22f2593` (24 ckpt) | **6** | 3 | 15 | 0.70 | 24 |
| sea ice `139471e7` (44 ckpt) | 8 | **11** | 25 | 0.65 | 44 |
| **pooled** | **14** | **14** | 40 | -- | 68 |

28 decided pairs, 14 each, two-sided p = 1.00.

## What this settles, and what it does not

**It settles that the 2026-09-20 direction was a property of that conversation, not of routing.**
That result was reported as a direction and explicitly not a magnitude, because 9 decided pairs
could not carry one. A second conversation was the test that statement implied, and it fails:
the direction did not survive, and it did not merely weaken, it reversed.

**It does not establish that routing is worse.** 8-11 on the second conversation is 19 decided
pairs and also not a magnitude. What the two runs together show is that this instrument, on these
conversations, cannot tell the two arms apart -- 40 of 68 pairs tied, and the two decided sets
point opposite ways.

**The honest statement is now narrower than it was yesterday.** Not "routing cuts tokens 221x and
a judge finds it ahead", but: routing cuts tokens by roughly 221x, never fails to run where
`full_history` fails on 79% of checkpoints, and on answer quality a blinded judge cannot
distinguish it from taking the most recent turns instead.

## The instrument is the limiting factor, and it is getting worse

Order agreement across every judge run in this project:

| run | order agreement |
|---|---:|
| 2026-09-18, `hybrid_router` vs `full_history`, old labels | 0.82 |
| 2026-09-19, same pair, query-derived labels | 0.82 |
| 2026-09-20, `hybrid_router` vs `query_recent_only` | 0.71 |
| 2026-09-21, `indexed_router` vs prose, head 160 | 0.77 |
| 2026-09-21, `indexed_router` vs prose, head 320 | 0.75 |
| **2026-10-01, `hybrid_router` vs `query_recent_only`, sea ice** | **0.65** |

A judge that contradicts itself on a third of pairs cannot resolve a difference that produces 28
decisions out of 68. Every quality claim this project has made has been bounded by that, and the
bound is tightening rather than loosening as the conversations get less like the first one.

## What the lexical measure did this time

It agreed with the judge -- `query_recent_only` 0.333 against `hybrid_router` 0.288. That is worth
noting because coverage and the judge have disagreed in both directions before, and it is a
reminder that agreement on one run is not evidence that coverage is a quality measure: on the
published conversation the same coverage measure put `query_recent_only` *last*.

## What would change the picture

Not another judge run at this sample size, and not another conversation labelled the same way.
The instrument needs to be more stable before more pairs are worth buying -- either by judging
each pair more than twice and taking a majority, or by using a judge whose order agreement is
measurably higher. Until one of those happens, further replications will produce more of this:
two directions, neither one a magnitude.
