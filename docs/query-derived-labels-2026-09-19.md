# Query-derived answer requirements, and the reopened gate

2026-09-19. The 2026-09-18 answer gate's quality half failed at the floor: its
requirements were written by reading the assistant's reply, so they encoded what
that one answer -- which had the full conversation *and* the user's follow-ups --
happened to say. Across 48 requirement instances the judge credited
`hybrid_router` 1 and `full_history` 9; both arms at the floor, and the ceiling
inside the label set. No re-run of the judge could lift that, so the labels were
re-derived instead.

## What changed

The 20 checkpoints the answer gate ran now carry requirements written from the
**query**, with the original reply withheld. The annotator saw the query and the
conversation strictly before it -- the same causal envelope a router gets -- and
wrote what a correct answer must contain, one to three per checkpoint. This is
the labelling step `docs/real-answer-gate-2026-09-18.md` named as the binding
constraint, done.

The four requirements that could not be scored at all (no `「」` span and no
ASCII identifier, so `requirement_satisfied` is False outright) were given the
checkable anchor their wording already implied; they would otherwise have capped
their checkpoints below 1.0 forever, which is the failure the validator now
refuses.

## Gates the new labels passed before any judge call was spent

All measured against the 2026-09-18 answer texts, scripts in the usual scratch
directory (`check_query_labels.py`, `span_provenance.py`).

**Scoreability.** All 59 requirements carry a matchable term; the validator
accepts the rewritten dataset.

**Contamination.** The annotator had read all 26 original answers in an earlier
pass, so a verbatim-recalled anchor would masquerade as a checkable fact. Of 112
quoted content anchors, 7 match an old answer-derived label exactly and 2
near-match; of those 9, **8 appear in the conversation strictly before the
query**, so they were read off the worksheet rather than recalled from the
withheld answer, and the ninth is the user's own query wording. No anchor is
answer-only.

**Discrimination.** Off-diagonal coverage stays low (mean **0.058**, 318 of 380
pairs at exactly 0.0, none at 1.0), so the labels still tell checkpoints apart.

The diagonal is the number that changed meaning. Against the original answers it
drops from **1.000 to 0.342** -- and that drop is the point, not a defect. The
old set scored perfectly against the answers it was copied from; a label that
does that is measuring itself. A query-derived label is not supposed to be
satisfiable by the particular reply that happened to exist, because that reply
had context the arms under test do not.

## What this reopens

A free control runs the identical swapped-pair protocol with the deterministic
coverage scorer and no model. On the old labels it could not separate the arms.
On the new ones it does, and the direction flips:

| | wins | losses | ties |
|---|---:|---:|---:|
| `hybrid_router` | **9** | 2 | 6 |
| `full_history` | 2 | **9** | 6 |

18 pairs, `hybrid_router` vs `full_history`, order-swapped. This is a lexical
control, not a verdict -- it cannot tell a correct explanation from an answer
that lists the right nouns, and it must not be reported as answer quality. What
it establishes is that the ceiling the old labels imposed is **gone**: there is
now separation for a judge to measure.

## What has not happened yet

The paid blinded judge has not been re-run. That is the one remaining measurement
needed before the quality half of the real-conversation claim can be written.
It needs the endpoint credentials (`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`)
and the explicit `--model deepseek-v4.1-flash` -- the environment's
`ANTHROPIC_MODEL` carries Claude Code's `[1M]` suffix, which the endpoint rejects
as `400 模型不存在`. The command is:

```bash
ctxlab judge-answers \
  .local/real-answer-run-2026-09-19-query-labels.json \
  .local/real-answer-judge-hybrid-vs-full-querylabels-2026-09-19.json \
  --arm-a hybrid_router --arm-b full_history --limit 20 \
  --model deepseek-v4.1-flash --max-tokens 2048
```

The answer run file it reads is `.local/real-answer-run-2026-09-19-query-labels.json`:
the 2026-09-18 answers and token counts untouched, `answer_requirements` replaced,
coverage summary recomputed. It stays in `.local/` -- the raw answer text still
contains absolute paths from the original conversation and fails the PII gate,
so it is not committable. The dataset and the judge output are the committable
artefacts.

Still blocking publication, unchanged: the model is a private proxy alias.
