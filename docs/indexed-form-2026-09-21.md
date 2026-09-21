# Does the form matter? The indexed rendering against the prose it replaces

2026-09-21. `indexed_router` is `hybrid_router` with one thing changed: the same events, in the
same order, selected by the same router, rendered as one compact row each -- identity kept,
body truncated to a 160-character head -- instead of as full prose. That costs **67% fewer
memory tokens** on the real checkpoints (median 660 against 1,980).

This is the measurement of whether that 67% is free.

## The result

| | wins | losses | ties | order agreement |
|---|---:|---:|---:|---:|
| `indexed_router` | **5** | 2 | 15 | 0.77 |
| `hybrid_router` | 2 | **5** | 15 | 0.77 |

24 pairs, all judged, 1 parse failure. Offline, the lexical coverage moved from 0.299 to 0.278
-- up on 3 checkpoints, down on 6, unchanged on 15.

**On the 22 answerable checkpoints the compressed form is ahead 5-2.** Seven decided pairs at
77% order agreement is not a magnitude, and 15 of 24 pairs tied, so the honest reading is that
the form does not cost quality and may buy some -- at two thirds fewer tokens. That is the
outcome that makes the arm worth having.

## The refusal stratum went the other way -- and the reason is a broken label, not a mechanism

| stratum | pairs | index | prose | order agreement |
|---|---:|---:|---:|---:|
| answerable | 22 | **5** | 2 | 0.77 |
| `must_refuse` | 2 | 0 | **2** | **1.00** |

Both refusal checkpoints went to the **prose** form, order-consistently. A first reading of this
was that compression costs the ability to abstain, and it looked corroborated: the 2026-09-19
run, judging `hybrid_router` against `full_history`, also gave both of these checkpoints to the
arm with more context.

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

### What is actually going on

The two checkpoints carry **contradictory labels**:

| checkpoint | `must_abstain` | first `answer_requirement` |
|---|---|---|
| `q-1227` | **True** | 必须明确回答「pinn」能否用于该数模题，并给出适用性判断 |
| `q-1370` | **True** | 必须明确回答能否用「comsol」辅助做题 |

`must_abstain` says the correct behaviour is to decline. The requirements say the correct answer
must clearly state whether the method can be used and must recommend it as an auxiliary check.
The judge scores against the **requirements**, so it rewards the answer that engages, and the
abstention label is unreachable by construction. The 2-0 is a judge preferring the more decisive
answer, and the stratum is not measuring refusal at all.

This is a real defect in the benchmark rather than a result about compression, and it is the
second label problem found the same way -- by reading the answers instead of the tally. Which of
the two labels is right is a judgement call that needs making: `q-1227` asks whether PINN is
applicable, which is arguably answerable from method knowledge rather than from the conversation,
in which case `must_abstain` is the wrong label; `q-1370` asks the same about COMSOL. Until that
is settled the stratum should not be reported, and no conclusion about abstention should be drawn
from it.

The head-length sweep that was launched to test the abstention hypothesis is therefore testing
something that does not exist as stated. Its coverage curve is still worth having -- it answers
"how short can the head be before answers degrade" -- but its refusal counts are reported with
this caveat or not at all.

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
