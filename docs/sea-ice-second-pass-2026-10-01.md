# The second conversation, second labelling pass: off the floor

2026-10-01. The requirements were re-collected with the assistant's reply withheld, which is the
fix `docs/sea-ice-gate-2026-10-01.md` named after diagnosing the first pass at 17.1% reply-only
anchors. It worked, and the number it moved is large enough to be worth recording separately from
the result it enables.

## What changed

Same 44 checkpoints, same contexts, same routing labels. Only `answer_requirements` were rewritten
-- 89 of them, against 125 in the first pass -- from the query and the conversation strictly
before it.

Five of the 89 could not be scored at all, because their anchor is written as plain text the
scorer cannot see: `03_数据` and `0.411` begin with a digit, `附录` is Chinese, `B` is a single
character. Each already named its anchor in the `围绕 X 进行核验` tail, so the edit was to the
punctuation around that anchor and to nothing else. That is the same fix the published dataset's
four unscoreable requirements got.

## The floor is gone

| arm | first pass | **second pass** | published conversation |
|---|---:|---:|---:|
| `query_recent_only` | 0.070 | **0.333** | 0.292 |
| `global_hybrid` | 0.076 | **0.288** | 0.333 |
| `hybrid_router` | 0.083 | **0.288** | 0.312 |
| `indexed_router` | 0.061 | **0.261** | -- |

Four times the coverage, and inside the published conversation's range. The 4x gap between the two
conversations was entirely the reply-derived anchors.

## A prediction that the measurement refuted

The pass left 18 of 89 requirements whose anchors all appear in the query, and this document's
author expected those to inflate every arm equally and compress the differences. Measured
directly -- how many of the 44 checkpoints does each requirement hold on -- that is wrong:

| requirement holds on | count |
|---|---:|
| every checkpoint (no signal) | **0 (0.0%)** |
| no checkpoint | 20-23 (22.5-25.8%) |
| 1 to 43 (discriminates) | **66-69** |

**Not one requirement is satisfied everywhere.** Having an anchor inside the query is a necessary
condition for an answer to satisfy a requirement by echoing it, and it is not a sufficient one --
the answers do not reproduce the query's wording. The theoretical objection was worth stating and
the measurement is what settles it; this is the third time in this project that a coverage
argument has been made and then refuted by reading the actual numbers.

## What the numbers now say, and what they do not

**The lexical measure puts `query_recent_only` first** -- 0.333 against `hybrid_router`'s 0.288.
On the published conversation the same measure put `global_hybrid` first and `query_recent_only`
last. So the ordering is not stable across conversations, which is the expected behaviour of a
measure this project has already documented as rewarding length over correctness.

It must not be read as routing getting worse on a second conversation. The blinded judge is the
instrument for that, and it has disagreed with coverage on real data twice.

**22.5% of the requirements hold on no checkpoint at all.** They are not a defect in the same way
the reply-derived anchors were -- they are answerable in principle, and the arms simply do not
reach them. Worth a look before the set is used for anything that reports a per-requirement
breakdown, and not worth blocking on.
