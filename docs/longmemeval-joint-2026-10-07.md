# Joint query-passage reranking: a measured retrieval gain

2026-10-07. Local CPU inference, no model API, no LLM judge. This is evidence
retrieval, **not generated-answer accuracy**.

## Dataset and protocol

The cleaned LongMemEval_S file on this machine has SHA-256:

    d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442

The three hard types contain 296 instances. Eighteen have abstention IDs and
are counted separately, leaving **278 answerable questions**: 121 multi-session,
127 temporal-reasoning, and 30 single-session-preference. Identical repeated
session IDs are collapsed; conflicting repeats and missing gold sessions fail
validation. The primary metric requires every gold session to appear in the top
five. Finding one gold session is reported separately and never counts as success.

No answer text, has_answer annotation, or answer_session_ids enters a retriever
or reranker. Labels are used only after selection.

- Candidate generation: each retriever offers 10; the fused list is capped at 20.
- Fusion: the same candidate list as the router, equal-weight RRF with k=60.
- Dense channel: minishlab/potion-base-8M, locally cached snapshot
  bf8b056651a2c21b8d2565580b8569da283cab23.
- Joint model: cross-encoder/ms-marco-MiniLM-L6-v2 at immutable commit
  233902d25c440f23af6f7d6e94d2946bac0bee0a, CPU, max_length=384, batch_size=16.
- Per candidate: BM25 selects two 180-word passages with 40-word overlap; joint
  scores are max-pooled by session. These settings were fixed before the smoke run.

The original static embedding adapter records its model name without pinning a
Hub revision. The cached revision above records what this run used. This remains
a limitation of that archived report's metadata. The current LongMemEval CLI
pins the embedding to that recorded snapshot for fresh runs, while the joint
model remains explicitly pinned.

## Result

| arm | any-gold@5 | all-gold@5 | fractional coverage@5 |
|---|---:|---:|---:|
| BM25 | 0.9065 | 0.6511 | 0.7902 |
| static dense | 0.8741 | 0.6187 | 0.7540 |
| equal-weight fusion | 0.9209 | 0.6835 | 0.8183 |
| existing router | 0.9173 | 0.6835 | 0.8180 |
| joint reranking | **0.9640** | **0.8165** | **0.9067** |

Joint versus fusion, paired on all-gold: **+0.1331**, **44 won / 7 lost**,
two-sided exact McNemar **p=1.2115e-7**, 20,000-resample paired bootstrap
**95% interval [+0.0863, +0.1835]**, seed 20260420. Joint versus the existing
router has the same discordant counts and difference. Both pass this repository's
existing requirement that the exact test and interval agree.

| question type | n | fusion all-gold | joint all-gold |
|---|---:|---:|---:|
| multi-session | 121 | 0.6777 | 0.8017 |
| temporal-reasoning | 127 | 0.6929 | 0.8268 |
| single-session-preference | 30 | 0.6667 | 0.8333 |

The front end is identical: **20 candidate misses** in every candidate-based arm.
Ordering losses fall from **68 to 31**, with recovered questions rising from
190 to 227. The candidate list's depth-aware ceiling is 256/278 = **0.9209**.
This is an improvement inside the stage the earlier decomposition identified.

These are a new protocol's numbers. The historical 296-question table included
abstention IDs and its fusion baseline truncated each channel at five, whereas
the new matched-candidate fusion uses the router's configured candidate list.
The two tables must not be subtracted from one another.

## Reproduce and inspect

~~~sh
python -m pip install -e ".[dev,neural,rerank]"
ctxlab longmemeval /path/to/longmemeval_s_cleaned.json .local/joint.json --embedder neural --rerank
~~~

The first smoke run covered 30 questions. Its flushed checkpoint was then
continued over all eligible questions, using exactly the same model and retrieval
settings. The full local report's SHA-256 is:

    8d0a7a9662037b97f2dabc46bbeee0cb7887a0d7d26532b71de176479ff15afa

That checksum describes the archived result before later report-metadata
extensions. Summary values above are read directly from its per-question records.
Transcripts, source databases and generated reports remain local.

## What this permits next

Keep the joint scorer **opt-in**. The result establishes a retrieval improvement
on one external benchmark and one candidate policy, with additional CPU work.
It does not establish a production latency target, multilingual behavior,
calibrated routing confidence, abstention accuracy, or better generated answers.

The next quality gate must generate and score answers at matched memory budgets,
using a fixed main model and the benchmark's reference answers. The original
real-conversation answer-quality null result remains unchanged.
