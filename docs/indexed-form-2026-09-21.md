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

## The refusal stratum went the other way, and it is the second time

| stratum | pairs | index | prose | order agreement |
|---|---:|---:|---:|---:|
| answerable | 22 | **5** | 2 | 0.77 |
| `must_refuse` | 2 | 0 | **2** | **1.00** |

Both refusal checkpoints went to the **prose** form, and the judge agreed with itself on both
orders for both of them. Two pairs is nothing on its own. It is worth writing down because it is
the second time this has happened from a different direction: the 2026-09-19 run, judging
`hybrid_router` against `full_history`, also gave **both** `must_refuse` checkpoints to the arm
with more context.

A mechanism that would explain both: a truncated context still *looks* like there is material
here, so the model engages and answers; a fuller context makes the insufficiency of the grounds
visible, so it declines. If that is what is happening, then compression does not merely trade
tokens for detail -- it specifically degrades the ability to say "I cannot tell from this".

**That is a hypothesis with two supporting observations of n = 2, not a finding.** The test that
would settle it is cheap and named: take the checkpoints whose correct behaviour is to refuse,
render the same selection at several head lengths, and see whether the refusal rate rises
monotonically with the head. If it does, the head length is a calibration knob for abstention
rather than a compression parameter, and that is a result in its own right.

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
