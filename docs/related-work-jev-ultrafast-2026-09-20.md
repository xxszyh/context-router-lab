# Related work: `browser-use/jev-ultrafast`

2026-09-20. Assessed because it makes a structurally similar claim in a different domain, and
because it is popular enough that a reader may ask about it. Short version: it is a useful
related-work anchor and the source of one arm this project does not have. It is not a dependency
and it does not touch this project's blocker.

## What it is, and how much the numbers are worth

A browser agent that replaces the usual screenshot loop with a compact **indexed table** of the
page's accessible elements -- one node, one index, with role, name and current value -- and asks
one model request per decision cycle over that table. Reported: a Google Flights search in
7,073 ms; median task time 9.450 s -> 7.092 s over six alternating runs; median browser protocol
calls 1,092 -> 101.

The repository is four days old (created 2026-09-16), has three commits, and 10,884 stars. That
is a launch-video profile: the stars measure a demo, not independent reproduction. Its own README
says the measurement is "three repeats of one task on one profile, not a general reliability
benchmark", and its design notes say "two websites do not establish broad reliability". Those
disclaimers are correct and are the reason the numbers should be read as a demonstration of
feasibility rather than as evidence of a general speedup.

## The one thing worth taking: form is a second axis of context reduction

This project's arms all answer one question -- **which subset** of the event log does the main
model see. `hybrid_router` selects by relevance, `query_recent_only` by recency,
`sliding_window` by position, the global retrievers by lexical or dense similarity, `full_history`
by taking everything. Every arm is a subsetting policy.

Jev answers a different question: **in what form**. Its observation is not a smaller DOM, it is
the same DOM re-rendered as a structured index -- the model is shown `[14] combobox "From" =
"Zürich"` rather than the markup. Nothing is dropped for relevance; the representation changes.

Those are the two canonical answers to context reduction, and this project measures only one of
them. That suggests a cheap ablation that isolates form from content:

> **`indexed_router`** -- `hybrid_router`'s exact selection, rendered as an index rather than as
> raw event text. Same events, same order, same token budget, but each event appears as one
> compact row (index, actor, context, sequence, truncated head) instead of its full prose.

If the two arms score the same, form does not matter at this scale and the subsetting result
stands on its own. If the index scores higher at equal tokens, then part of what looks like a
routing win is really a formatting win, and the paper needs to say so. Either outcome is a
result, and the arm is a few dozen lines on top of the existing assembly path.

Two smaller transfers, both already partly present:

- **"Send visible text only"** -- offscreen content never enters the model's context. The
  structural analogue here is that tool traffic outnumbers prose two to one in the real store
  (1,157 tool calls and 1,156 tool results against 575 assistant messages). Excluding it is a
  free structural filter, not a semantic one.
- **Freshness guards keyed on semantic state**, with a generated value reused only while its
  entire input is unchanged. The analogue is memoising a routing decision against the exact
  context state that produced it.

## Where the relationship runs the other way

Jev is a concrete instance of the thing this project argues is under-measured. It ships an
aggressive context-reduction strategy, reports a headline speedup, and its own documentation
concedes that the evaluation cannot support a general claim. Its design notes also state that
"a valid action can still be wrong" and that "independent checks, rather than the model's DONE
choice, determine whether the demonstrated task succeeded" -- which is this project's position on
the LLM judge, arrived at independently.

So the useful framing is not "related system, similar technique". It is: **the field is
converging on structured, reduced context in production, and the measurement discipline for
claiming it is the open problem.** That is a positioning asset for the write-up, and it is a
citation rather than a dependency.

## What it does not do

It does not address this project's blocker. That blocker is a documented, pinned model plus
sample size, and no amount of browser-agent engineering speaks to either. There is no shared
code, no shared data, and no shared metric between the two projects.
