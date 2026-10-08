# Maintenance and the next quality experiment, 2026-10-07

## Where the project is stuck

The original real-conversation answer experiment remains a null result. The
2026-10-02 termination statement is retained as historical evidence, including
its later amendments. Neither synthetic evidence coverage nor session recall
is a measurement of generated-answer quality.

The most promising measured opening is the LongMemEval experiment: on its three
hard question types, ranking left 0.247 of the default candidate list's all-gold
ceiling unclaimed. Scalar features cannot read a question and a passage
together. Expanding the candidate list by itself did not solve ordering.

There was also an engineering blocker: the LongMemEval code existed only in
machine-specific, unpublished scratch scripts. A reader of the public repository
could not execute the experiment described in the documentation.

## What now ships

The longmemeval command streams the external dataset and reports:

- BM25, dense retrieval, equal-weight RRF, and the existing router on the same questions.
- Any-gold, all-gold, and fractional gold-session coverage, with all-gold primary.
- Candidate misses, selection losses, and a depth-aware oracle ceiling.
- Paired all-gold differences, exact McNemar tests and seeded bootstrap intervals.
- Dataset SHA-256, model identifiers, candidate policy, question types and limit.
- A separate, opt-in joint query-passage reranking arm.
- A flushed per-question checkpoint; interrupted runs can resume only with matching
  data and retrieval configuration. An uncommitted, incomplete final line is
  discarded on resume; completed records remain intact. A final report is replaced atomically.

Labels are only scored after retrieval. The answer text and has_answer fields
never enter a retriever or a reranker. Missing gold sessions are validation errors;
they are never silently removed from the denominator. Abstention cases are
counted separately and excluded from the retrieval-success metric.

The official cleaned S file repeats identical sessions in seven of its 296 hard
questions. Identical visible transcripts under the same session id are collapsed;
conflicting transcripts under one id are rejected. The report records collapsed
copies. Gold sessions remain intact.

The cross-encoder is pinned to an immutable revision, runs locally on CPU, and
does not change the default router or pretend its raw scores are calibrated
probabilities. It takes the same candidates as the fusion/router arms, picks two
180-word passages per candidate with BM25 (40-word overlap), then max-pools the
pair scores per session. This bounds model input and avoids silently truncating a
long session to its opening paragraph.

## Reproduce

Obtain the cleaned LongMemEval_S JSON from the benchmark's official release:
[LongMemEval](https://github.com/xiaowu0162/LongMemEval).
The dataset is external and is not copied into this repository.

~~~sh
python -m pip install -e ".[dev,neural,rerank]"
ctxlab longmemeval /path/to/longmemeval_s_cleaned.json .local/lme.json --embedder neural
ctxlab longmemeval /path/to/longmemeval_s_cleaned.json .local/lme-joint.json --embedder neural --rerank
~~~

The hash embedder is the explicitly named, offline default for smoke tests. It
does not reproduce the neural results. Neural embeddings use the existing
model2vec provider. The LongMemEval command now pins its embedding snapshot as well
as the cross-encoder revision, so checkpoint resume cannot mix model revisions.
Existing callers of the embedding adapter retain their original unpinned behavior.

For a first run, add --limit 30. This is a smoke test on the first 30 eligible
questions, not a randomized holdout or a quality result. Repeat a command with
--resume to reuse its checkpoint. The optional weights are fetched once and
cached; HF_ENDPOINT can select a mirror when the default Hub is unreachable.

## Decision rule

The first full run is now complete:
[joint reranking results](longmemeval-joint-2026-10-07.md).
On 278 answerable questions (18 abstention cases counted separately), all-gold@5
rose from 0.6835 for the matched-candidate fusion to 0.8165 for joint reranking,
44 paired wins against 7 losses. This is a measured retrieval gain; answer
quality still needs the separate gate below.

Compare joint against hybrid and router on the same full set at the same depth,
and report its runtime. A retrieval gain is established only when both paired
tests pass the repository's existing rule. An unsuccessful reranker remains
experimental; it is not promoted to a default to make the project look successful.

If retrieval improves, the next gate is end-to-end answer accuracy on held-out
questions, with a fixed generator, matched token budgets, explicit abstention,
and the benchmark's reference answers. Until that gate is measured, generated
answer quality remains unestablished.

The offline [answer exchange](answer-exchange-2026-10-07.md) now implements the
generation/independent-judging/reporting chain. It requires complete paired
imports and common generator settings, hides reference labels from generation,
and records character or declared local-tokenizer budgets. The local batch runner
previews by default, journals each attempt, and requires explicit execution against
a loopback Chat Completions service. No live generation or paid calls have been made.
A new held-out protocol and real answer/abstention experiment are still required.

The full local plan contains 278 questions across hybrid/router/joint/query-only,
1,112 generation requests, and a shared 12,000-character memory cap. Six questions
contain identical session copies with conflicting timestamps; all dates are
preserved and marked ambiguous, without dropping questions. Reference integer
answers in the cleaned release are supported. The plan remains local and its
status is pending generation, not a new answer-quality result.

Refusal questions can now be retrieved separately with --abstention-only or
included with --include-abstention. Their retrieval metrics are null and never
enter the answerable recall denominator. Answer plans keep refusal labels private;
independent judging uses the missing-information policy, and quality reports give
answerable and abstention strata separate summaries and paired comparisons. An
empty or failed response never counts as a correct refusal.

A lightweight hash-embedding smoke run prepared all 18 hard-type refusal cases
from the local cleaned S file, then exported 54 requests for hybrid/router/query-only.
It does not reproduce or extend the neural/joint retrieval result. All 1,112 existing
answerable requests also pass the batch dry run, with zero network calls or output
writes. Mock transports exercise generation, independent judging, scoring, interruption,
partial-tail recovery, explicit failure retries and strict response validation.

Local validation passes 332 tests, Ruff lint/format, strict mypy, and a package
wheel build. Validation uses mock transports and offline fixtures; no desktop
input or live generation is involved. The real-data refusal plan remains
pending_generation, with one question containing ambiguous duplicate-session dates.

The next [rendering audit](rendering-audit-2026-10-07.md) reproduces all archived
generation inputs and finds 101 strict annotated-turn rendering losses among 211
successful, annotation-eligible joint retrievals. A chronological overlap union
does not improve full retention (one paired win and one loss), so it is not promoted.
Whole-turn BM25 selection improves strict retention from 110/248 to 175/248,
70 paired wins against five losses; the same 30 unknown-annotation questions are
explicitly excluded from this span metric, not from session recall or answer scoring.
The full-turn renderer remains opt-in, and the original answer-quality null is intact.

This phase passes 348 tests, Ruff, strict mypy and a wheel build. Two new full input
plans remain pending generation. These are exploratory input-retention results,
with unchanged sessions and budget caps; they do not establish answer accuracy.

The [cross-rendering quality comparison](quality-comparison-2026-10-07.md) now
exports a frozen primary joint comparison over all 278 questions: 556 mixed,
blinded generation inputs, followed by one common judging batch and question-ID
aligned paired answer scoring. Dataset, retrieval, budget, questions, references
and selected sessions must match. Missing imports, stale hashes, mixed settings
and a judge sharing the generator's identifier are rejected. Empty normal-stop
answers now also contribute to the failed-completion rate. Sample criteria cover
accuracy, failures and memory; inspected data cannot authorize default promotion.

This phase passes 389 tests, Ruff lint/format and strict mypy. All 556 projected
prompts reproduce the archived source inputs in a real-data rendering audit; the
old source plan hashes remain intact. Local batch preview makes zero calls and
writes no answers. The generator/configuration is not yet selected or frozen,
answers and independent judgments remain pending, and GUI work stays paused.

Model source: [MS MARCO MiniLM cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
This is a passage ranker trained for MS MARCO, not a model validated for long
conversation reasoning. Its use here tests the untried joint-reading architecture.
