# State of play

One page to pick this project up. Everything below is measured, and each claim names the
evidence grade, because several of them are weaker than their numbers look.

## The question

> In a long agent conversation that interleaves several unrelated tasks, can routing the query
> to the right logical contexts cut the tokens the main model reads, without losing answer
> quality?

## Established

**1. The architecture has headroom, on synthetic data.** Oracle routing saves **85.2%** of
memory tokens against full history at unchanged labelled-evidence recall. Gate one passes on
substance, not on a technicality. *Grade: exact — the synthetic labels are constructed, so the
evidence ids are ground truth.*

**2. The router beats the strongest baseline on both axes, out of sample.** On sessions it was
never trained or tuned on: evidence recall **1.000 at 642 memory tokens**, against
`global_bm25` at 1.000 and 750. *Grade: exact for recall and tokens; the claim is about
evidence recall, not answer quality.*

**3. Cheap is not correct, and the router used to be worse than cheap.** The three cheapest
arms save 89–95% of tokens and recover 46–54% of the evidence. Before the seven defects were
fixed the router scored 0.744 against BM25's 1.000 while spending more — the defects are
recorded in the README, and finding them required widening the dataset first. *Grade: exact.*

**4. Evidence recall is a directionally valid proxy for answer quality.** The arm with 0.55
recall **refused 60% of the time** rather than fabricating, so the model does not paper over
missing memory with parametric knowledge. Both instruments agree on direction: the judge 10–2,
strict coverage 0.600 against 0.250. *Grade: **direction only**. Magnitudes are not usable —
see below.*

**5. The synthetic router's measured failure is over-selection, not misses.** 780 checkpoints:
candidate required-context recall 1.000, candidate misses 0, selection losses 0, but selection
precision 0.591 with 540 excess contexts. *Grade: exact. Measured by Codex's stage diagnostics.*

**6. Real replay has a working context-level label, if a weaker one.** "Which contexts does the
answer draw on" is measurable lexically on the corrected reply text: over 23 checkpoints,
**micro recall 0.880, precision 0.710, F1 0.786**, scoring the reply against each context's
member events. Deterministic, no model, no cost, re-derivable by anyone. It is a property of the
answer text, not a counterfactual, so it is **not** `required_context_ids` and does not rescue
the routing gates. What it does is replace an assumption about label quality with a measurement:
65% exact-set agreement with careful reading, and the disagreements are enumerated in v0.3.
*Grade: measured against a 23-checkpoint sample; the descriptors were the wrong signal and had
scored 21-29%.*

## Not established

**Answer quality, in magnitude.** The judge agrees with itself across answer order only **85%**
as of its last run, about a third of its calls produced no readable verdict at some point, and
half the ties are refusal checkpoints. n = 20. **Nothing here is publishable as a quality
result.**

**Anything about a documented model.** Every call-based number used `deepseek-v4.1-flash`
through a private Anthropic-compatible proxy. The alias is not reproducible by anyone else and
the operator can change what it resolves to. Marked as such in the README and in
`answer-quality-experiment.md`.

**The real-replay routing gates.** See blockers.


## Blockers

**The required-context label is circular.** `required_context_ids` means "without this context
the query cannot be answered" — a counterfactual. To label it one must already know which
context contains the answer, which is the capability the benchmark exists to measure. Five
attempts failed; two ran on correct inputs and one of those isolated the reason.
→ `docs/v0.3-necessity-is-circular.md`. The synthetic benchmark is not circular because the
generator constructs the evidence.

**No independent second annotator.** The plan's 20% double-annotation floor is the only thing
blocking scale-up, and it is at 0%. A same-annotator recheck was run and is recorded in its own
fields; it cannot substitute, and the attempt is evidence for why.

**Re-running anything call-based needs ≥1024 max_tokens.** This model's thinking block shares
the output budget and exceeds 512 on its own. At 16 it produces empty replies that read as
results.

## Next step, in order

1. **Write `answer_requirements` for the 26 real checkpoints.** Read `answer_after`'s output and
   state what a correct answer must contain. No counterfactual, no retrieval. Already checked on
   five: requirements are writable by reading and the existing scorer discriminates 1.00 against
   the checkpoint's own answer and 0.00 against another's. *Offline.*
2. ~~Re-open the lexical question on the corrected reply text~~ — **done, and it works**:
   micro recall 0.880 / F1 0.786 for "which contexts does the answer draw on", scoring against
   member events rather than descriptors. The context-level label for real replay exists, is
   deterministic, and costs nothing. Wired into the schema as the weaker label it is.
3. **The answer-quality gate on real conversations** (v0.2 work package E). Same arms, one
   pinned model, blinded pairwise judging, token cost reported. *Needs a documented model.*

Routing gates are answered on synthetic data, where the labels are exact. Real replay answers
the quality and cost question. That split is v0.3's main consequence.

## Where things are

| | |
|---|---|
| synthetic arms + gate | `evaluation/arms.py`, CLI `compare-baselines` |
| answer experiment | `evaluation/answers.py`, CLI `answer-experiment` |
| blinded judge | `evaluation/judge.py`, CLI `judge-answers [--judge coverage]` |
| necessity by ablation | `evaluation/necessity.py`, CLI `annotate-necessity` — **its result was negative; read v0.3 before reusing** |
| real-replay format + PII gate | `datasets/real_replay.py`; labels in `datasets/real-replay/`; scrubbed export gitignored |
| Claude history importer | `importers/claude_code.py`, CLI `ingest-claude` |
| reports | `docs/answer-quality-experiment.md`, `docs/v0.3-necessity-is-circular.md`, `docs/annotation-protocol.md` |

159 tests, `ruff check`, `ruff format --check`, `mypy src tests` all clean.

## Error log, for whoever continues

Recorded because each recurred after being understood once, and each costs calls:

1. **The thinking budget.** Set `max_tokens` for the answer, forget that thinking shares it. Hit
   twice; the second time wasted 117 calls and produced a table of zeros that read like a
   finding.
2. **Reading the wrong events.** Judged "what the reply says" on the first assistant event after
   a query — normally a tool call, median 72 characters against the answer's 1,520. Invalidated
   three of six attempts and misled the write-up, which claimed six failures when three never
   ran on the right input.
3. **Watching one metric per iteration.** Four judge rounds each watched a different number —
   consistency, then no-verdicts, then cost — so run 5's perfect 100% consistency was reported
   without noticing the judge was wrong on the only pair that could be checked by hand.
4. **Scripted edits that report success without matching.** Several applied nothing while
   printing success, leaving a half-edited file. Two of the three ablation crashes trace to it.
   Verify by reading the diff, not the script's output.
