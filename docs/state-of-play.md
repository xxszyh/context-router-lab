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

**7. On real conversations the token cut is 221x, and full history sometimes cannot run at
all.** 20 checkpoints: `hybrid_router` reads **1,671** memory tokens, `full_history` a median of
**370,102** (range 53,381–836,459), input-token ratio **215x**. Two of the twenty were rejected
by the server at `>1,048,566 tokens` for `full_history` and succeeded on all three selective
arms — a **10% failure-to-run rate against 0%**, which no quality metric can express.

That 10% was measured against a 1M proxy alias, which was itself generous. Measured against the
context windows that **publicly documented** models actually serve — 258,048 for `qwen3-max`,
260,096 for `kimi-k2.5`, both read off the gateway's own refusal — `full_history`'s median input
of **485,966** (max 1,042,473) puts it over the limit on **13 of 18** checkpoints: a **72%
failure-to-run rate**. The claim is not that routing is cheaper than reading everything; it is
that at a documented model's ceiling, reading everything is not an available strategy at all.
→ `docs/documented-model-selection-2026-09-20.md`.
*Grade: exact — tokens and status codes are counted; the ceilings are the gateway's own.*

**8. The answer gate's ceiling is lifted, and a control separates the arms.** The 2026-09-18
quality failure was in the labels, not the arms: requirements written from the original answer
put both arms at the floor. On 2026-09-19 the 20 gate checkpoints were re-labelled **from the
query, with the original reply withheld**. The new set passes scoreability (59/59 matchable),
contamination (8 of 9 recalled anchors were visible before the query; the ninth is the user's
own wording) and discrimination (off-diagonal 0.058). A free coverage-judge control on the new
labels splits `hybrid_router` from `full_history` **9–2, 6 ties** — the separation the old set
could not produce. The diagonal's drop from 1.000 to 0.342 is the point: the old set was scoring
itself. **The paid judge has now run on the new labels: the arms separate, but the direction
favours `full_history` 5–2 with 9 ties, and at n=18 / 82% judge agreement that is a signal, not
a conclusion. The coverage control and the paid judge now disagree on direction on real data --
the concrete case for never reporting coverage as quality.** *Grade: exact tally; the model is
still a private proxy alias.* → `docs/answer-gate-judge-2026-09-19.md`.*

## Not established

**Answer quality, in magnitude.** Still not publishable, but the reason moved. The 2026-09-18
verdict ("judge 82% consistent, decided 3 of 16, nothing publishable") was a property of labels
written from the original answer; on 2026-09-19 those labels were replaced with a query-derived
set and a free control now separates the arms (see **8**). What remains unmeasured is the
**paid blinded judge on the new labels** — the one number the quality half needs. n = 20.
→ `docs/real-answer-gate-2026-09-18.md` (the old result) and
`docs/query-derived-labels-2026-09-19.md` (the reopened gate).

**Anything about a documented model.** Every *existing* call-based number used
`deepseek-v4.1-flash` through a private Anthropic-compatible proxy. The alias is not reproducible
by anyone else and the operator can change what it resolves to. Marked as such in the README and
in `answer-quality-experiment.md`.

The selection half of this is now done — `kimi-k2.5` and `qwen3-max` are publicly documented
names served on `anthropic:messages`, and both were verified to read a long prompt faithfully
(a planted marker recovered at 25%, 50% and 75% depth of a 130k-token prompt, so the earlier
`input_tokens=13` reading was a broken usage report rather than silent truncation). What is not
established is a **result** under one: the re-run is measured, not yet judged.
→ `docs/documented-model-selection-2026-09-20.md`.

**The real-replay routing gates.** See blockers.


## Blockers

**The required-context label is circular — resolved by removing the field, not by labelling
it.** `required_context_ids` means "without this context the query cannot be answered" — a
counterfactual. To label it one must already know which context contains the answer, which is
the capability the benchmark exists to measure. Five attempts failed; two ran on correct
inputs and one of those isolated the reason. **Schema 2.0 drops the field from the real-replay
annotation entirely**, which is what v0.3 recommended: a field nobody can label is worse than
no field, because it invites a gate. `answer_requirements` replaces it. The synthetic
benchmark is unaffected and not circular — the generator constructs the evidence, so its labels
are exact by construction and the routing gates are answered there.
→ `docs/v0.3-necessity-is-circular.md`.

**The independent-human annotation and six-checkpoint adjudication are complete.**
All 6 rows are complete and all 13 final requirements match their reference replies. Every
final set also exactly matches the existing primary gold label, so no gold text changed. The
user confirmed that `annotator-2` is human and that B06's intended decision was “use primary”;
the import records the correction without rewriting the source workbook. The blinded workflow
is recorded as user-reported independence. Coverage is 6/26 = 23.1%, so the plan's 20%
independent-human floor is met.
See `docs/secondary-annotation-2026-09-18.md`.

**Re-running anything call-based needs ≥1024 max_tokens.** This model's thinking block shares
the output budget and exceeds 512 on its own. At 16 it produces empty replies that read as
results.

## Next step, in order

1. ~~Write `answer_requirements` for the 26 real checkpoints~~ — **done, and they discriminate.
   Superseded on 2026-09-19: this set was written from the answer and capped the gate at the
   floor; the 20 gate checkpoints were re-labelled from the query (see 5).** The numbers below
   describe that first, answer-derived set, kept because its 1.000 diagonal is exactly the
   self-measurement the rewrite removed.
   The labels live on the checkpoints in `datasets/real-replay/claude-d22f2593.json`, behind the
   same PII gate the export path uses; 65 requirements over 24 checkpoints (two have none: their
   reference turn was interrupted
   before any prose existed, so correct behaviour there is `must_abstain`, not a requirement
   string). Each checkpoint's requirements score **1.000 against their own answer**, and
   against all 552 other-answer pairs the mean is **0.0088** with 541 pairs at exactly 0.00,
   **no pair at 1.00**, and **60 of 65 requirements satisfied by no other answer at all**.
   *Grade: exact, and offline — this is a property of the label set, not a model result.*
   The eight requirements that first failed to match their own answer were **all** a quoted
   span broken by markdown; see the error log.

   **How much wording matters, bounded without a second annotator.** A differently-worded
   requirement is usually one with a different number of checkable terms, so the proxy is to
   vary how strictly the labels are read and watch the off-diagonal. The diagonal holds at
   **1.000 under every rule**, and the off-diagonal degrades slowly: **0.0094** when every
   term must match, **0.044** at half, **0.146** at any (three leaked checkpoints, thirteen at
   the loosest). *Grade: measured, with a stated substitution — strictness stands in for
   wording, which is an approximation.* Two limits come with it: **43%** of requirements are
   anchored on prose rather than on a number, so they are the ones a re-wording could move;
   and the labels do not extend to the two interrupted checkpoints, so the usable set is
   **24 of 26**.
2. ~~Re-open the lexical question on the corrected reply text~~ — **done, and it works**:
   micro recall 0.880 / F1 0.786 for "which contexts does the answer draw on", scoring against
   member events rather than descriptors. The context-level label for real replay exists, is
   deterministic, and costs nothing. Wired into the schema as the weaker label it is.
3. ~~Adjudicate the six independently labelled checkpoints~~ — **done.** Both raw sets and the
   adjudication artifact are preserved. The human annotator and B06 correction are confirmed;
   23.1% now counts toward and passes the independent-human coverage floor.
4. ~~The answer-quality gate on real conversations~~ — **run, and two-sided.** On 20 real
   checkpoints `hybrid_router` reads **1,671** memory tokens against `full_history`'s median
   **370,102** — a **221x** cut — and `full_history` **failed to run on 2 of 20** with
   `Input exceeds the context limit (1048566 tokens)` where all three selective arms succeeded.
   Quality did **not** separate: the lexical metric puts every arm between 3.7% and 5.0% and
   ranks `full_history` *last*, while the blinded judge put `full_history` ahead 3–0 with 13 of
   16 pairs tied. The wording confound — the requirements quote the reference answer, and
   `full_history` is the arm most likely to resemble it — was **tested and refuted**: re-judging
   the four decided pairs with the prose anchors rewritten moved **nothing** (4/4 either way,
   control reproducing the original exactly). **The real problem is the floor**: across 48
   requirement instances the judge credited `hybrid_router` **1** and `full_history` **9**, so
   the decisions separate almost nothing. The requirements were written *from the original
   answer*, which had the full conversation **and the user's follow-ups**; neither arm is that
   answer, and re-judging cannot lift the ceiling. → `docs/real-answer-gate-2026-09-18.md`.
   *Still needs a documented model before any of it is publishable.*
5. **The answer gate's labels re-derived from the query** — **done, and the ceiling is gone.**
   The 2026-09-18 gate failed at the floor because its requirements were written from the
   original answer. On 2026-09-19 the 20 gate checkpoints were re-labelled with the original
   reply withheld. The new 59-requirement set passes scoreability, contamination (no anchor is
   answer-only) and discrimination (off-diagonal 0.058). A free coverage control separates
   `hybrid_router` from `full_history` 9–2-6; the **paid judge reverses that to 5–2 for
   `full_history`** with 9 ties -- the arms now separate, but in the direction that does not
   flatter routing, and still underpowered. → `docs/answer-gate-judge-2026-09-19.md`.
   **Outstanding: re-run under a documented fixed model (the gateway exposes many), then enlarge
   the sample with the 6 labelled-but-unrun checkpoints.**
6. **Choosing the documented model, and what it costs the comparison** — **done, and it
   reshapes the gate.** `minimax-m3` was the only documented candidate advertising the 1M the
   `full_history` arm needs; it reads a 20k prompt fine but at 500k returns a *garbled* marker,
   an empty content block and raw filler, so its advertised window is not usable. Under the two
   candidates that are trustworthy the arm runs on 5 of 18 checkpoints, so the powered comparison
   is **`hybrid_router` vs `query_recent_only`** — relevance against recency at equal budget,
   both running everywhere, and the comparison a deployed router actually faces. `full_history`
   is reported where it survived, with its n. → `docs/documented-model-selection-2026-09-20.md`.

Routing gates are answered on synthetic data, where the labels are exact. Real replay answers
the quality and cost question. That split is v0.3's main consequence.

## Where things are

| | |
|---|---|
| synthetic arms + gate | `evaluation/arms.py`, CLI `compare-baselines` |
| answer experiment | `evaluation/answers.py`, CLI `answer-experiment` |
| blinded judge | `evaluation/judge.py`, CLI `judge-answers [--judge coverage]` |
| necessity by ablation | `evaluation/necessity.py`, CLI `annotate-necessity` — **its result was negative; read v0.3 before reusing** |
| shallow answer scorer | `evaluation/scoring.py` — `deterministic_coverage`; markdown-normalised, **never report it as answer quality on its own** |
| real-replay format + PII gate | `datasets/real_replay.py`; labels in `datasets/real-replay/`; scrubbed export gitignored |
| Claude history importer | `importers/claude_code.py`, CLI `ingest-claude` |
| reports | `docs/answer-quality-experiment.md`, `docs/v0.3-necessity-is-circular.md`, `docs/annotation-protocol.md`, `docs/real-answer-gate-2026-09-18.md`, `docs/secondary-annotation-2026-09-18.md` |
| run artefacts | `.local/` — gitignored; the run and judge JSON live there, never the raw store |

Validation status is refreshed after each imported annotation artifact; see the latest commit
or working-tree test output for the exact count.

183 tests, `ruff check`, `ruff format --check`, `mypy src tests` all clean.

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
5. **The metric measuring formatting.** `requirement_satisfied` compared raw text, so a quoted
   span with `**`, a backtick or an escaped `\*` inside it counted as a miss. On the real label
   set that silently deflated **8 of 65 requirements**, every one a verbatim quote from the very
   answer it was supposed to match — `「内部运动规律对 τ₄ 完全无影响」` against
   `… τ₄ **完全无影响**`. It never surfaced as an error, only as a plausible-looking low score.
   This is the dangerous version of defect 2: not reading the wrong input, but reading the right
   input through a filter that only fails on one arm's formatting. `normalize_for_matching` now
   strips `*`, `` ` `` and `\` and all whitespace from **both** sides; underscores are kept
   deliberately, because they are identifier characters here and not emphasis. The check that
   caught it generalises: **make each label score 1.00 against its own answer before trusting
   any score against another's.**
