# Enlarging the real-replay sample: what is already there and what is missing

2026-09-21. The gate runs on 24 checkpoints from one conversation. Every result in
`docs/answer-gate-documented-model-2026-09-20.md` rests on 9 decided judge pairs, so the binding
constraint is sample. This records what enlargement actually requires, because the first two
things I believed about it were both wrong.

## What I had wrong

**"Six labelled checkpoints never entered the gate."** That was true when the gate ran on 20.
It now runs on **24 of the 26** in the dataset. Only two are outside it:

| checkpoint | why |
|---|---|
| `q-0219` | Visio figure request; the reference turn was interrupted, so its reply is empty |
| `q-2450` | convergence-plot request; same |

Neither is blocked by its emptiness -- the query-derived method does not read the answer. They
are blocked by **where the checkable anchors would come from**: the specification is in the query
itself (`r`, `δr`, `L`; `N=1000, 2000, 4000, 8000`), and every arm sees the query, so a
requirement anchored there is satisfied by any answer that echoes it. That is the failure the
discrimination check exists to catch, and it is a labelling judgement rather than a mechanical
one.

**"The other conversations need importing."** They do not. The store already holds them:

| events | contexts | assignments | session |
|---:|---:|---:|---|
| 3,104 | 8 | 2,787 | `d22f2593` -- the current dataset |
| 2,125 | 0 | 0 | `3e9f38ee` -- this project's own session |
| 1,209 | 0 | 0 | `139471e7` -- 实训3, sea ice, four sub-questions |
| 1,037 | 0 | 0 | `66c90492` -- 实训7, path planning |
| 327 | 0 | 0 | `96978eac` -- 实训6, population |
| 182 | 0 | 0 | `c5a830f8` |
| 136 | 0 | 0 | `6e7a5b28` |

The **events** are imported. The **context catalog, the event-to-context assignments, and the
checkpoints** are not, and those are the annotation.

## What enlargement costs

Per conversation, in order:

1. **Context catalog** -- name, goal, summary and member events for each interleaved task. Not
   derivable: deciding that two episodes are the same logical context is the labelling act.
2. **Event-to-context assignments** -- which events belong to each context.
3. **Checkpoints** -- the query positions, their `query_type` and `relation_label`.
4. **`answer_requirements`** -- the query-derived labels, written without reading the reply.

Steps 1-2 are what the draft at the local annotation folder is for; steps 3-4 are the worksheets
that already exist as a workflow.

## The expected yield

The current dataset: **3,104 events -> 26 checkpoints**, roughly one per 120 events. The three
usable candidates total **2,573 events**, so of order **20 more checkpoints** -- which would take
n from 24 to about 44 and the decided judge pairs from 9 to roughly 16.

That is the difference between "a direction the sample cannot turn into a magnitude" and a
number. It is also a few hours of labelling per conversation, which is why it has not happened
yet.

## What was built

A **context-catalog draft** for `139471e7`, at the local annotation folder (not committed: real
conversation). It does the mechanical half and refuses the judgement half:

- segments the conversation into **19 contiguous episodes**, each starting at an explicit task
  marker (`第N题`, a file name, a task keyword, or a skill invocation);
- reports each episode's range, event count and user turns;
- asks the annotator to **group episodes into contexts** and to correct boundaries;
- flags the three episodes under 8 events as probable false splits, since a passing mention of a
  task name is enough to start one.

The first version of this draft merged every occurrence of a task into one context. It produced a
913-event "paper" blob spanning the whole conversation, because the rule treated any mention of
`图片` or `文档` as a task marker. Deciding that seq 93 and seq 240 are the same logical context
is exactly the judgement the draft must not make, so the second version reports episodes and
leaves the grouping open.
