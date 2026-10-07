# Evidence retention after memory rendering, 2026-10-07

Session recall does not show which evidence survives passage selection and the
memory budget. This audit reconstructs the exact archived generation prompts and
tracks fully visible whitespace words back to their original session/turn positions.
It makes no generator, network or model-weight calls.

## Metric and its limits

[LongMemEval's dataset description](https://github.com/xiaowu0162/LongMemEval#-dataset-format)
defines `has_answer=true` for evidence-bearing turns. These labels are read only
after label-free rendering. The renderer accepts dated `(role, content)` pairs;
reference answers, session gold labels and turn annotations never enter selection.

The strict metric requires every whitespace word in every annotated gold turn to
remain fully visible. A truncated final word does not count. Repeated text from
another turn cannot substitute for the original source position. Overlap is counted
once for retention; repeated appearances of the same source word are reported separately.
Speaker/date/excerpt metadata is not evidence content.

This is a diagnostic proxy. A long annotated turn can contain unnecessary words,
and a generator can answer from other evidence or prior knowledge. Complete span
retention is neither necessary nor sufficient for correct answers. The audit also
reports word coverage and whether each annotated turn is touched by any source word;
touching one word does not establish semantic coverage.

Gold sessions without positive turn annotations, empty annotated spans, or conflicting
annotations on identical duplicate sessions produce unknown retention, never a perfect
score. Unknown counts are explicit. Abstention questions have null evidence metrics
and their own counts. Actual answer accuracy remains `not_measured`.

## Frozen baseline and bottleneck

The existing full answerable plan contains 278 questions, four arms and 1,112 prompts.
The memory cap is 12,000 **characters** per question/arm, not model tokens. The audit
reproduces every original prompt and memory count without modifying the plan.
Turn annotations are complete for 248 questions; 30 questions are unknown for this metric.
Session recall on all 278 questions is still reported independently.

For the joint arm, all required sessions are retrieved in 211 of the 248 eligible
questions. Only 110 retain every annotated turn in full: **101 rendering losses**
after successful session retrieval. Another 37 eligible questions fail session retrieval.
All annotated turns are touched in 131 questions, so 80 of the 211 successful session
retrievals leave at least one annotated turn untouched. These are span diagnoses, not
claims that those answers will be wrong.

The legacy joint prompts contain 21,269 duplicate source-word appearances out of
536,539 retained appearances. The legacy renderer also places windows in BM25 score
order inside a session, rather than the original conversation order.

## Optional chronological union

`chronological-union-v2` keeps the same retrieved sessions, dates, question set and
per-session budget shares. It ranks 180-word windows with 40-word overlap using the
same query-only BM25 policy, then greedily fits their union. Windows that overlap
are merged, final selection is clipped at whole source-word boundaries, and fragments
are rendered in their original order **within each session**, preserving whitespace.
A mid-turn fragment carries its original speaker; gaps use an explicit excerpt marker.
Sessions keep their retrieval slot order. Neither annotations nor reference text guide fitting.

The implementation is opt-in. The default legacy renderer and its archived inputs
are retained. Paired auditing requires identical dataset, retrieval-report hash, arms,
budget, question set and selected sessions; it independently reconstructs both prompts.
Local tokenizer budgets require the same tokenizer file/hash and declared generator.
Full-retention comparisons use exact McNemar and 20,000 paired bootstrap draws with
seed 20261007. All arm comparisons are exploratory on previously inspected questions.

The full union comparison is a **null result** for strict retention: each memory arm
has one newly retained question and one newly lost question, zero net difference,
exact p=1.0 and bootstrap interval [-0.01210, +0.01210]. In joint, touched-turn retention
falls from 131/248 to 129/248 even though duplicate source-word appearances become zero.
Neither readability nor duplicate removal establishes a quality improvement.

## Optional whole-turn selection

Only one of the 101 lost joint questions contains an annotated turn longer than
2,300 characters; the median annotated-turn length among successfully retrieved,
annotation-eligible joint questions is 281 characters. This suggests inspecting the
selection unit rather than assuming all lost evidence is too long. It does not locate
the necessary answer phrase inside each annotated turn.

`whole-turn-bm25-v3` uses the same sessions and budget shares but ranks complete
visible turns with BM25, for both speakers with no role priority. It greedily accepts
turns whose union fits, skips oversized turns and preserves original turn order. If
no turn fits, it falls back to bounded query-selected windows; the private trace marks
this fallback. It never uses `has_answer`, gold session status or reference answers.

Skipping an oversized relevant turn in favor of shorter turns can still harm answers.
Whole-turn selection is an experimental alternative, and its strict-retention metric
favors selecting entire annotated turns by construction. Actual answer judging and
independent data remain necessary. This proposal follows inspection of the same dataset;
any resulting paired comparison is exploratory, not a pre-registered confirmation.

## Whole-turn result on the matched full plan

All 278 questions and selected session IDs are unchanged; strict turn retention uses
the same 248 annotation-eligible questions. The 30 unknowns consist of 10 multi-session
and 20 temporal questions without complete positive gold-turn annotations.

| Arm | Legacy full retention | Whole-turn full retention | Paired wins/losses | Rendering losses, legacy → whole-turn |
|---|---:|---:|---:|---:|
| hybrid | 100/248 (40.32%) | 153/248 (61.69%) | 57/4 | 81 → 28 |
| router | 99/248 (39.92%) | 155/248 (62.50%) | 60/4 | 83 → 27 |
| joint | 110/248 (44.35%) | 175/248 (70.56%) | 70/5 | 101 → 36 |
| query_only | 0/248 | 0/248 | 0/0 | 0 → 0 (all 248 are retrieval misses) |

For joint, the difference is **+26.21 percentage points**, exact McNemar p=9.8178e-16,
95% paired bootstrap interval [+20.16, +32.26] points. Among its 211 successful,
annotation-eligible session retrievals, full retention rises from 52.13% to 82.94%.
Mean annotated-word coverage rises from 65.93% to 83.60%. Mean rendered memory over all
278 joint questions falls from 11,906 to 10,361 characters (12.98% less); both share
the same maximum cap. No input was padded to simulate equal actual length.

| Joint question type | Eligible / unknown | Legacy full retention | Whole-turn full retention |
|---|---:|---:|---:|
| multi-session | 111 / 10 | 36/111 (32.43%) | 64/111 (57.66%) |
| temporal-reasoning | 107 / 20 | 63/107 (58.88%) | 89/107 (83.18%) |
| single-session-preference | 30 / 0 | 11/30 (36.67%) | 22/30 (73.33%) |

Five joint questions regress on the strict span metric; they are retained in the
denominator. An audit of successfully retrieved joint cases finds 443 annotated user
turns and one assistant turn. This dataset-specific label distribution limits
generalization; the algorithm applies the same selection rule to both speakers.
The opt-in result is an input-evidence improvement, not a generated-answer gain or
a reason to change the default. The original answer-quality null remains unchanged.

Local validation: 348 tests, Ruff lint/format, strict mypy and wheel build pass.
All three policies use label-free rendering; old prompts are reproduced exactly.
The two new plans each contain 1,112 requests and remain pending generation. They
retain the same six ambiguous-date questions. No live generation or paid calls occurred.

Local artifacts (not committed):

- Original plan: `.local/answer-plan-full-2026-10-07/plan.private.json`, plan SHA-256
  `6927362ec48ed57572a69c3bfbdd3df5e7fb44a80932252155a4f390844457f9`.
- Union plan: `.local/answer-plan-union-2026-10-07/plan.private.json`, plan SHA-256
  `f77fee867dbd7018cc361c4fd5beb0d8de2e6c595a25b4f803116bd83c9e2bae`.
- Whole-turn plan: `.local/answer-plan-turn-2026-10-07/plan.private.json`, plan SHA-256
  `055655b8c96a9ae9f9f1719f510381e343e4e464fdcb9dfa332dc111227a8b81`.
- Union paired audit: `.local/render-audit-paired-2026-10-07.json`, file SHA-256
  `4189bbaa6a9ac2b4408f183bd89a1a0aac0380ec16d2df66ba4ef7e2720acf07`.
- Whole-turn paired audit: `.local/render-audit-turn-paired-2026-10-07.json`, file SHA-256
  `67d5efc9b7b10423899a009f1754d0fa7684c19867197966f7d86cbccc937b04`.

## Reproduce without generation

Use the same complete retrieval report for both plans:

~~~sh
ctxlab longmemeval-answer-plan /path/to/longmemeval_s_cleaned.json .local/joint.json .local/legacy-plan
ctxlab longmemeval-answer-plan /path/to/longmemeval_s_cleaned.json .local/joint.json .local/union-plan --rendering chronological-union-v2
ctxlab longmemeval-render-audit /path/to/longmemeval_s_cleaned.json .local/legacy-plan/plan.private.json .local/rendering-audit.json --compare-plan .local/union-plan/plan.private.json
ctxlab longmemeval-answer-plan /path/to/longmemeval_s_cleaned.json .local/joint.json .local/turn-plan --rendering whole-turn-bm25-v3
ctxlab longmemeval-render-audit /path/to/longmemeval_s_cleaned.json .local/legacy-plan/plan.private.json .local/turn-audit.json --compare-plan .local/turn-plan/plan.private.json
~~~

The audit accepts the original legacy plan, including plans saved before source tracing
was added. It never trusts saved trace metadata to improve a score: source spans are
reconstructed and the generated prompt must match exactly. Existing files and source
inputs are protected by the CLI. Plans, private labels and audit reports stay local.

For a declared token-budget plan, add `--tokenizer-file /path/to/tokenizer.json` to
the audit command. This counts the rendered memory only, not a service's full chat template.
To later generate or score answers, use the [answer exchange](answer-exchange-2026-10-07.md)
and report the rendering policy alongside model/configuration and answer metrics.
