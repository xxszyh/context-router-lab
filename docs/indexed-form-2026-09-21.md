# Does the form matter? The indexed rendering against the prose it replaces

2026-09-21. `indexed_router` is `hybrid_router` with one thing changed: the same events, in the
same order, selected by the same router, rendered as one compact row each -- identity kept,
body truncated to a 160-character head -- instead of as full prose. That costs **67% fewer
memory tokens** on the real checkpoints (median 660 against 1,980).

This is the measurement of whether that 67% is free.

## The result

| | wins | losses | ties | order agreement |
|---|---:|---:|---:|---:|
| `indexed_router` | **5** | 4 | 15 | 0.77 |
| `hybrid_router` | 4 | **5** | 15 | 0.77 |

24 pairs, all judged, 1 parse failure. Offline, the lexical coverage moved from 0.299 to 0.278
-- up on 3 checkpoints, down on 6, unchanged on 15.

**The compressed form is ahead 5-4 across all 24 pairs.** Nine decided pairs at 77% order
agreement is not a magnitude, and 15 of 24 pairs tied, so the honest reading is that the form
does not cost quality and may buy a little -- at two thirds fewer tokens. That is the outcome
that makes the arm worth having. (An earlier version of this table read 5-2: it held two
checkpoints out of the tally that the corrected label below puts back in.)

## The refusal stratum went the other way -- and it turned out to have no members

The 2-0 split described in this section was first read as compression costing the ability to
abstain. **That reading is wrong twice over**: the answers do not support it, and the stratum it
was read from should not exist. The correction is below; the section is kept because the wrong
reasoning is the instructive part.

Both of the checkpoints filed as `must_refuse` went to the **prose** form, order-consistently, and
it looked corroborated: the 2026-09-19 run, judging `hybrid_router` against `full_history`, also
gave both of them to the arm with more context.

**That reading is wrong, and the answers say so.** Checking the refusal flags on the four
answers behind the earlier result:

| checkpoint | `hybrid_router` | `full_history` | judged winner |
|---|---|---|---|
| `q-1227` | declines in substance, no cue | **refuses explicitly** | `full_history` |
| `q-1370` | declines in substance | **engages** | `full_history` |

On `q-1227` *both* arms declined and `full_history` still won; on `q-1370` *neither* refused and
`full_history` still won. Refusal does not track the verdict in either direction. The same holds
in this run: on `q-1227` the index arm is the one whose answer trips the refusal detector, and it
lost.

### What is actually going on: the label means something else

The two checkpoints carry **contradictory labels**:

| checkpoint | `must_abstain` | `query_type` | first `answer_requirement` |
|---|---|---|---|
| `q-1227` | **True** | `new_context` | 必须明确回答「pinn」能否用于该数模题，并给出适用性判断 |
| `q-1370` | **True** | `new_context` | 必须明确回答能否用「comsol」辅助做题 |

`must_abstain` says the correct behaviour is to decline; the requirements say the correct answer
must clearly state whether the method can be used and must recommend it as an auxiliary check.
The judge scores against the **requirements**, so it rewards the answer that engages, and the
abstention label is unreachable by construction.

**The contradiction is in the reader, not in the labels.** The annotation protocol sets
`must_abstain` by two rules: rule 1, where the reference reply drew on no context *and declined*
(`unanswerable`), and rule 2, where it drew on no context *and did not decline* (`new_context`).
Both checkpoints are rule 2. `must_abstain` there is a descriptive fact about the reference reply
-- it answered from method knowledge rather than from the conversation -- and the requirements
are consistent with it. The evaluation was reading a provenance label as a behavioural
requirement, so it scored a good answer as a failed refusal.

**Fixed** (`04531fc`): scoring and the judge's strata now key on `query_type == "unanswerable"`,
which is the protocol's rule 1 and the only case where declining is correct. No checkpoint in
this dataset is `unanswerable`, so **the refusal stratum has zero members** and every judged pair
belongs in the main tally.

Re-counted from the saved outcomes, at no cost:

| run | as reported | corrected |
|---|---|---|
| 9-20 `hybrid_router` vs `query_recent_only` | 6-2, 14 ties (22 pairs) | **6-3**, 15 ties (24) |
| 9-21 `indexed_router` vs prose, head 160 | 5-2, 15 ties (22) | **5-4**, 15 ties (24) |
| 9-21 `indexed_router` vs prose, head 320 | 3-6, 12 ties (21) | **4-7**, 12 ties (23) |

Small, all the same direction, and **no conclusion in this document changes**. That is the useful
thing about this class of defect: it is worth fixing precisely because it was not load-bearing
enough to have been caught from the results.

The head-length sweep that was launched to test the abstention hypothesis was therefore testing
something that does not exist as stated. Its coverage curve is still worth having -- it answers
"how short can the head be before answers degrade" -- and its refusal counts are not reported.

## The head length is a real knob, and the default sits at the bottom of its curve

The sweep ran the same selection at five head lengths. The refusal counts it was built for say
nothing -- see above, the stratum is a broken label -- but the coverage curve is a separate and
useful result.

| head (chars) | mean memory tokens | mean strict coverage, 22 answerable |
|---:|---:|---:|
| 40 | 370 | 0.273 |
| 80 | 452 | 0.222 |
| **160 (the default)** | **660** | **0.242** |
| 320 | 789 | **0.341** |
| 640 | 1,097 | 0.341 |
| prose (no truncation) | 1,625 | 0.295 |

Two things are worth reading off it, and one caution.

**The default is the worst point on the curve.** `INDEX_HEAD_CHARS = 160` was chosen when the arm
was built, not measured, and it scores below both its shorter and its longer neighbours. A
160-character head cuts a dense tool result -- where the numbers a requirement asks for actually
live -- somewhere in the middle.

**At 320 the compressed form beats the prose form on coverage at half the tokens** (0.341 against
0.295, 789 against 1,625). If that survives a judge it is the strongest form of the result this
arm exists to test.

**The caution is the same one that applies to every coverage number in this repository.** The
curve is not monotone -- 40 scores above 80 and 160, then coverage jumps between 160 and 320 --
which is what a noisy lexical measure looks like, and 22 checkpoints cannot separate differences
of this size. Coverage rewards an answer that names more of the right nouns, and a longer answer
names more of them; that is exactly the failure caught on `q-0801` below.

### The judged comparison was run, and it refutes the curve

| | head 160 | head 320 |
|---|---|---|
| judge: `indexed_router` vs prose | **5-4** | **4-7** |
| coverage: index vs prose | 0.278 vs 0.299 | **0.326** vs 0.270 |
| mean memory tokens | 660 vs 1,625 | 808 vs 1,605 |

**Coverage points the opposite way from the judge in both runs.** At 160 the compressed form
scored *lower* coverage and *won*; at 320 it scored *higher* coverage and *lost*. The coverage
gain the sweep found was verbosity, exactly as the `q-0801` case predicted, and the sweep's
reading of 320 as the best point is refuted.

Across the two judged runs the forms are **9 to 11 on 20 decided pairs, with 27 ties** -- the
prose form marginally ahead, and 9 of 20 is as close to even as an even number allows. The honest
conclusion is that the form is not measurably load-bearing at either head length, and the 51-67%
token saving is therefore not being paid for in answer quality.

The caveat is the instrument. Order agreement was 0.77 and 0.75, and 27 of 43 pairs tied -- a
real difference smaller than roughly a fifth of pairs would be invisible to a judge that
disagrees with itself this often. "Not measurable" is the claim; "no difference" is not.

**The default stays at 160.** The judge cannot separate the head lengths, and 320 costs 22% more
memory tokens for no measurable gain, so the cheaper point is the one to keep. Changing it on the
strength of the coverage curve would have been changing it on the strength of the measure this
comparison just refuted.

The `must_refuse` stratum split 1-1 at head 320, having gone 0-2 and 2-0 in the two earlier runs.
Three runs, three different answers -- from a stratum that has since turned out to have **no
members at all**. Those three results were noise being read as a pattern, which is what a stratum
built on a misread label produces.

## What the coverage number would have said

The lexical measure put the two forms 0.021 apart and would have supported "the compression is
nearly free". It also scored `q-0801` **0.00 for prose and 1.00 for the index** at an 80% token
saving -- and reading the two answers shows why that is not a win for the index: the prose answer
is one sentence naming one file, the index answer is a paragraph naming three, and the
requirement asks for *the* file name. The longer answer hits more terms. This is the
coverage-rewards-verbosity failure the README warns about, caught in the act on a checkpoint
where it would otherwise have been read as a strong result for the compressed form.

## Artifacts

The judge transcript is **not committable**: it quotes both arms' answers, and 2 of its 49 calls
carry the local user name inherited from an answer that quotes an absolute path. It stays in
`.local/` with only its numbers published, consistent with the repository's rule that real
records are never committed. The answers file is likewise local.

## What is not established

Whether the compressed form preserves quality, in magnitude. 7 decided pairs of 24 cannot carry
it, and the refusal stratum -- the one place a real difference showed up, twice now -- has two
checkpoints in it.
