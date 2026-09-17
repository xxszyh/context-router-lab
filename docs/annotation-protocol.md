# Annotation protocol for real-replay query types

Written after the first pass used `cross_context` on 7 of 26 checkpoints and a re-derivation
could not confirm most of them. The cause was that the types were assigned by reading and
judging, so two readings of the same conversation could reasonably disagree. This replaces
judgement with a decision procedure.

The rule the protocol is built on:

> **A query's type is a property of how its referent relates to the set of contexts, not a
> property of its wording.**

An earlier draft keyed on words (`回到` → return, `现在` → switch) and it failed immediately,
because `这个` appears in nearly every turn and the demonstrative rule became a catch-all.
Wording is evidence for finding the referent; it is not the label.

## Definitions

Two mechanical quantities, both computed from the annotation's own `context_members`:

- **`previous_active(seq)`** — the context owning the most recent assigned event strictly
  before `seq`. This is the context the conversation is currently "in".
- **`active_before(seq)`** — every context that owns at least one assigned event before `seq`.

And one derived from the conversation itself:

- **`reply_contexts(seq)`** — the contexts the assistant's next reply materially draws on.
  Computed by scoring each context descriptor against the reply text and keeping those within
  half of the best score. The gold answer defines what was required; this is gold-label
  construction, not leakage. Leakage would be letting the *system under test* see it.

## The procedure

Apply in order. The first rule that matches decides the label.

| # | Condition | Type | `required_context_ids` |
|---|---|---|---|
| 1 | `reply_contexts` is empty **and** the reply declines to answer | `unanswerable` | `[]`, `must_abstain=True` |
| 2 | `reply_contexts` is empty, reply does not decline | `new_context` | `[]`, `must_abstain=True` |
| 3 | `len(reply_contexts) >= 2` | `cross_context` | `reply_contexts` |
| 4 | one context, it is `previous_active`, and the query has no task noun | `short_coreference` | that context |
| 5 | one context and it is `previous_active` | `continue` | that context |
| 6 | one context, `active_before` it, but not `previous_active` | `return` | that context |
| 7 | one context, never active before | `switch` | that context |

> **The right-hand column no longer exists in the schema.** Real-replay schema 2.0 dropped
> `required_context_ids`, for the reason this page ends on. It is kept in the table because the
> procedure is a correct *specification* of a label that turned out to be unlabelable --
> deleting the column would hide what was being attempted. What replaced it is
> `answer_requirements`: readable off the assistant's own reply, so it needs no retrieval and
> no counterfactual. See `v0.3-necessity-is-circular.md`.

**Task noun** means any of: a file name, `题`/`问`, `图`/`表`, `附录`, `代码`, `模型`, `算法`,
`文件`, `材料`, `网格`. Rule 4 exists because a query made only of a demonstrative carries no
task information at all — it can only be resolved from the immediately preceding turns, which
is what `short_coreference` means. A query that names a task is `continue` even if it is short.

Rule 6 versus 7 is the sharpening that matters. Both mean "move to a different context", and
the earlier draft could not separate them. The operational difference is **whether the target
has ever been active**: re-entering a context that was open before is a `return`; entering one
for the first time is a `switch`. That is checkable from `context_members` alone, so two
annotators applying it must agree.

Rule 3 is deliberately conservative. `cross_context` requires that the *answer* drew on two
contexts, not that the query happened to mention two. A query naming two contexts but
answerable from one is rule 5 or 6, not this.

## The mechanical input does not exist, and that is the real finding

This procedure was implemented in `annotation_policy.py` and applied to all 26 checkpoints.
It changed 23 of them, and the results were plainly wrong: `q-1227`, which asks whether the
whole problem could be done with a PINN, came out as `cross_context` requiring six contexts.

Two attempts were made to compute `reply_contexts` and both failed, in opposite directions:

| signal | failure |
|---|---|
| entity overlap, argmax | cannot express two contexts at all, so `cross_context` is unconfirmable by construction; and filtering entities to three characters or more stripped `dr`, `cn`, `50`, `0.05` out of their own contexts |
| BM25 of the descriptor against the reply, threshold at half the best | every descriptor is generic Chinese prose, so nearly every reply clears the bar for nearly every context |
| distinctive identifiers only (file and artifact names) | still 3 of 26; a long reply legitimately names several problems |

Three signals, three failures, and the reason is not tuning. It is that the quantity being
asked for is the wrong shape for a text-overlap measure:

> **"The reply mentions context X" is neither necessary nor sufficient for "the query cannot
> be answered without X".**

A feasibility question can mention six contexts and require none. A one-line answer can
require one and mention none. `required_context_ids` is a **counterfactual** property --
remove the context and ask whether the answer survives -- and no similarity score computes a
counterfactual.

So rules 1–7 above are a correct *specification* and a useless *procedure*: every rule that
matters reads `reply_contexts`, and `reply_contexts` is not obtainable this way. The file
`annotation_policy.py` is kept for that diagnosis and marked as not in use, rather than
deleted, so the next attempt does not repeat the three signals above.

Consequences, recorded because they change what the next step is:

1. ~~**The first-pass labels stand.**~~ **They were removed instead.** Reading is a legitimate
   way to judge a counterfactual -- it is the *reproducibility* that is missing, not the method
   -- but a label only one person can reproduce, and that person cannot re-derive it, is not a
   label the project can gate on. Schema 2.0 dropped the field rather than keep values nobody
   could defend, and `answer_requirements` took its place.
2. **The over-used `cross_context` is not thereby confirmed or refuted.** The re-derivation
   that first raised the suspicion was measuring mention, not necessity, so it was never
   evidence about the label. The suspicion remains open, with no measurement.
3. **Necessity needs a counterfactual judge** -- an independent annotator, or a model asked
   "if this context were removed, could the query still be answered?". A lexical threshold is
   not a cheap version of that; it is a different question.
