# Session report, 2026-10-02: three withdrawn explanations and one stage left standing

A consolidated account of one working session against the handover state of 2026-10-01. Six commits
on `main`, one experiment branch, and three explanations the session proposed and then refuted
itself on. Written because the project's most expensive asset is the record of what it got wrong and
why, and because a reader arriving at the earlier documents one at a time would find three
successive corrections without a map of them.

Everything below is offline, deterministic, and reproducible without an API key.

## Where it started

The handover's own summary: **the token half of the proposition is established and stronger than
originally claimed, and the quality half is a null result** -- 35 adjudicated pairs, 18-17, p = 1.00,
with the root cause not in sample size but in two implementation choices nobody had measured. No
external benchmark had been touched, and the project's own generator was known to write its
hypothesis into its labels.

## The six commits

| commit | what it did | decisive number |
|---|---|---|
| `b3215ab` | adopted SCALE-QA as an external benchmark; made the difficulty split, the miss decomposition and the ordering attribution first-class | 836/1000 questions already answered by one lookup |
| `f990d311` | **withdrew** the claim that the router's ordering sits behind its best channel | 9 won against 11 lost, p = 0.824 |
| `8420ecc0` | four times the discriminating questions, each difference reported with its paired interval | 602 questions; two of three differences established |
| `5e830023` | swept the filler share and the corpus size to find what moved the decomposition | corpus 0.131 -> 0.246; filler nothing |
| `1ea79a42` | put the fusion record on `main`, where it was already cited from | no dangling references |
| `f216a290` | swept the candidate budget, and measured whether widening it helps | oracle 0.754 -> 0.982; recall@5 0.550 -> 0.528 |

The experiment branch `channel-weighted-fusion` holds the fusion-weights work and its negative
result; `docs/fusion-weights-2026-10-02.md` records it on `main`.

## The claims, now

| claim | status | evidence |
|---|---|---|
| Routing reads far less memory than full history | **supported** | 1,671 tokens against a 370,102 median; and full history often does not fit a real context window at all |
| Routing loses no answer quality | **null, unchanged** | 18-17 at p = 1.00; the benchmark cannot currently test it |
| The router beats plain lexical retrieval | **refuted** | 0.640 against 0.556 on the discriminating questions of two packages; it ties a subset and loses the rest |
| The router's ordering is behind its own best channel | **withdrawn** | -0.012 at 9 won against 11 lost (p = 0.824); at 602 questions -0.007, interval [-0.035, +0.023]. It is *bounded*, not negative |
| Equal-weight fusion of unequal channels costs the ordering | **established** | -0.085, 43 won against 94 lost, p < 0.001, [-0.123, -0.048] |
| The stage after the fusion recovers most of it | **established** | +0.078, 61 against 14, p < 0.001, [+0.051, +0.106] |
| Repairing the fusion improves anything end to end | **refuted** | gating on measured channel recall leaves the held-out score at 0.712, exactly where the hand-set ranker already was |
| Candidate generation is the bottleneck | **refuted** | it is a property of `top_per_retriever=10` and `max_candidates=20`; the gold is scored by some channel for 601 of 602 hard questions |
| The front end's failure is a corpus-size effect | **refuted** | the 0.115 corpus gap becomes 0.006 at a wide candidate budget |
| Widening the candidate budget improves the answer | **refuted** | gold in the list 0.754 -> 0.982, list 18 -> 338 items, recall@5 0.550 -> 0.528 (paired -0.022, p = 0.198) |
| Dense retrieval helps on this benchmark | **not shown** | 0.297 to 0.378 against lexical's 0.556; a small static encoder, so a transformer encoder stays untested |
| The entity channel contributes anything here | **refuted** | 0 of 3,561 contexts carry an entity; its ordering recall falls from 0.252 to 0.003 when the list grows and id order stops coinciding |
| The ranking stage is the bottleneck | **supported, by elimination** | it is the only stage whose output gets *worse* when it is given strictly more information |

## The methodological finding

Three successive explanations for the same deficit, all of them plausible, all of them alive in the
literature, and all of them **hand-set constants or corpus properties standing next to the stage that
was actually failing**:

1. **the fusion weights** -- equal weight for channels of unequal quality. Real defect, established.
   Repairing it measures as zero benefit, because the calibration stage downstream already absorbed
   it.
2. **the corpus** -- miss rate 0.131 at 1,000 evidence dialogues and 0.246 at 3,000. Real effect,
   and it disappears (0.115 -> 0.006) the moment the candidate budget is widened.
3. **the candidate budget** -- the front end offers the gold for 75.4% of hard questions at the
   defaults and 98.2% at a wide budget. Widening it buys no recall at all.

The move that separated them each time was the same: **sweep the hand-set constants around a stage
before blaming the stage.** The things that survived are the ones that are not constants. This is
worth stating as a rule for the next reader, because all three refutations took one measurement each
and each of them would otherwise have shipped.

## The instruments, and what each prevents

| instrument | the number whose absence lets a benchmark mislead |
|---|---|
| `recency_reachability` | whether the arrangement decided the recency question in advance (0% or 100% = it did) |
| `difficulty_strata` | how much of a benchmark one cheap lookup already answers (84% here) |
| `routing_miss_decomposition` | whether a miss was never offered or offered and dropped |
| `candidate_order_attribution` + `paired_ordering_comparison` | the interval on every difference, so a 0.012 at n = 164 cannot be quoted as a finding |
| `ArrangementReachability.truth_blocks` | how many evidence-bearing sessions are in the index |

Five instruments, one shape: **measure the property of the benchmark that would make the headline
number meaningless, before quoting the headline number.** Two of them exist because this session
published something it later had to withdraw.

## What is open

1. **The ranker.** It is the last stage standing and the only one that degrades under more
   information. Its form is a fixed linear combination over normalized features whose largest weight
   sits on a feature that is identically zero here; its "calibrated" scores are set-dependent,
   because `_normalize` divides by the maximum of whatever candidate set it is given, which makes the
   `t_low` threshold mean different things at different list sizes. A fitted ranker exists and did
   *worse* on a held-out fold (0.697 against 0.712), so "fit it" is not obviously the answer either.
   The open question is whether the router's per-candidate signals contain anything that could beat
   plain lexical order (0.556) over a rich candidate list whose ceiling is 0.982.
2. **The quality half of the proposition.** Unchanged and still untested by any instrument this
   session added. It needs a benchmark whose discriminating part is larger than 16% and whose
   evidence can be reached by paraphrase, not only verbatim.
3. **The real conversations.** Every measurement in this session is on an external benchmark. On
   `sea-ice` the router earns its keep on cost, not on reach; on `d22f2593` it is beaten outright by
   `global_bm25`. That sign disagreement has not been revisited since the handover.

## Reproduce

```bash
# the external benchmark, with its arrangement's reachability and its difficulty split
ctxlab scale-qa external/pkg_512k_3000 scaleqa.json --window-size 256 \
    --mode lexical --mode router --embedder neural

# the internal benchmark
ctxlab sweep-assembly run/events.sqlite run/dataset/benchmark.jsonl run/sweep.json \
    --include-recent-in-evidence
```

The package builder is SCALE-QA's own and is offline:

```bash
python build_eval_dataset.py --context-budget 512k --num-questions 3000 --output-dir pkg_512k_3000
```

## What was deliberately not published

Four files carrying a concurrent writer's in-flight edits were left untouched
(`docs/annotation-protocol.md`, `docs/state-of-play.md`,
`src/context_router/evaluation/metrics.py`, `docs/enlargement-sea-ice-2026-09-22.md`), and the
measurement was rebuilt from `main` plus an explicit file list rather than from the working tree,
which carries both lines at once. Nothing from `artefacts/` or `annotation/` was touched or
uploaded: those hold unscrubbed real conversations. Two files that *are* published,
`src/context_router/evaluation/arms.py` and `tests/test_arms.py`, carry both parties' edits and
could not be split -- the published tree was verified by rebuilding it and running the suite there.

*Grade: every number in the tables above is reproduced in the document that owns it, with its
configuration and its grade. Nothing in this report is measured for the first time here.*
