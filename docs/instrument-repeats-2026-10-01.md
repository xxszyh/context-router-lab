# Asking each pair three times: what repetition bought

2026-10-01. The judge was the binding constraint on every quality claim here -- order agreement
across its runs had drifted from 0.82 to 0.65, and the last run decided 19 of 44 pairs while
tying 25. So each pair was asked three times, in both orders each time, six calls per pair, and a
strict majority decided. Same answers, same model, same prompt; the only variable was how many
times each pair was asked.

## The result

| | `hybrid_router` | `query_recent_only` | ties | decided | p |
|---|---:|---:|---:|---:|---:|
| one repeat | 8 | 11 | **25** | 19 | 0.65 |
| **three repeats** | 9 | **17** | **18** | 26 | **0.17** |

**Repetition removed seven ties and flipped no decisions.** Of the 25 pairs the first run tied,
seven resolved to `query_recent_only`, one to `hybrid_router`, and seventeen stayed ties -- and
of the 18 pairs both runs decided, **none changed sides**. That is the behaviour the design
predicted: the collapsed ties were noise and came out, the genuine ties were information and
stayed, and the decisions that were already made did not move.

Of the 26 pairs the three-repeat run decided, 18 were unanimous across all six calls, three had
one dissent and five had two.

## What it changes about the claim

**The instrument was part of the problem, and it is now measurably better.** Ties fell by 28%,
`mean_consistency` is 0.853, and the two runs agree on every pair they both decided. The fix is
not a change of result dressed up as a change of method.

**The direction on this conversation is now clearer and still not significant.** `query_recent_only`
leads 17-9 where the single-repeat run had it 11-8; the two-sided p is 0.17, down from 0.65 and
still above any conventional threshold. So the honest reading of the second conversation is
unchanged in kind -- recency is ahead -- and stronger in degree.

**The seven recovered ties split 7-1 toward `query_recent_only`.** Worth noting and not worth
reading: eight coin flips landing 7-1 is p ~ 0.07, and the more interesting question is whether
the collapsed ties were systematically ones where the recency answer was better, which one run
cannot answer.

## What this does not fix

**The published conversation has not been re-judged.** Its 6-3 for `hybrid_router` came from a
single repeat, and comparing it against a three-repeat run would be comparing two instruments.
The pooled figure from `docs/replication-2026-10-01.md` -- 14-14 -- is therefore provisional, and
the next step is to re-judge the published conversation the same way before either number is
reported.

**A judge that contradicts itself a third of the time is still a weak instrument.** Majority-of-six
raises the effective reliability; it does not make the judge a good one. `mean_consistency` of
0.853 means the average pair had one call in seven disagreeing with the winner, and that is the
ceiling on how fine a distinction this instrument can carry.

## The two kinds of tie, now measured separately

The reason repetition is worth its cost, and the reason it must take a strict majority rather than
a plurality:

- **collapsed** -- the orders disagreed and the protocol turned that into a tie. 38 of the 90
  ties across this project's six judge runs. Repetition removes these.
- **genuine** -- both orders said tie because the judge found the answers equal. 47 of 90.
  Repetition must not remove these, and does not: seventeen survived three repeats.

That split is why a rule that always breaks ties would be wrong, and why `most * 2 > len(verdicts)`
is a strict majority rather than `Counter.most_common` -- with six calls a three-three split would
otherwise be decided by whichever verdict happened to be counted first.
