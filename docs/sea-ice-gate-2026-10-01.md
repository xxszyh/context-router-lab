# The second conversation's gate, and a worksheet defect that put it on the floor

2026-10-01. The first run of the answer gate on a conversation other than the one the project has
been reporting from. It lands at the floor, and the cause is a defect in how the labels were
collected rather than anything about the arms -- a defect this project had already found and
fixed once, and which the new worksheet reintroduced.

## What ran

| | |
|---|---|
| conversation | 实训3 北极海冰, 1,209 events, 7 contexts, **44 checkpoints**, 125 requirements |
| arms | `query_recent_only`, `global_hybrid`, `hybrid_router`, `indexed_router` |
| model | `kimi-k2.5`, `max_tokens=8192` |
| calls | 175 of 176, 1 failure, 0 truncations |

`full_history` is excluded: at a third of the published conversation's size its median input
still exceeds a documented model's 260k ceiling, and the calls that did fit would cost more than
every other run in this project combined.

## The result is a floor, not a routing finding

| arm | memory tokens (median) | strict coverage | refused |
|---|---:|---:|---:|
| `query_recent_only` | 978 | 0.070 | 7 |
| `global_hybrid` | 2,029 | 0.076 | 4 |
| `hybrid_router` | 2,005 | **0.083** | 3 |
| `indexed_router` | 862 | 0.061 | 7 |

The published conversation sits at **0.299**. This one is at 0.083, and 80% of checkpoints score
exactly zero on every arm. Nothing about the arms can be read from that.

## Where the anchors come from, against the published set as a control

Every requirement anchor is classified by where it can be found, and by whether any arm's memory
or answer contains it. Same measurement, both datasets:

| | published (225 anchors) | sea ice (351 anchors) |
|---|---:|---:|
| visible before the query | **94.7%** | 80.9% |
| **only in the reply** | **4.9%** | **17.1%** |
| in neither | 0.4% | 2.0% |
| in some arm's answer | 60.0% | 42.2% |
| in memory, not in answer | **16.0%** | **15.4%** |
| in no arm's memory | 24.0% | **42.5%** |

Two readings, and the second matters as much as the first.

**The sea-ice set has 3.5x the share of anchors that exist only in the assistant's reply.** An
anchor that exists only in the reply cannot be retrieved by any arm, by construction: the reply
is the thing the arms are trying to produce. Those 60 anchors are the difference between the two
datasets, and removing them brings the reachable share from 57.5% to 69.4% against the published
set's 79.9%.

**The "in memory, not in answer" share is the same in both** -- 16.0% and 15.4%. That component
is a stable property of the metric and the model, not a defect in either dataset. It is the share
of anchors a model had the material for and still did not produce, and it does not move when the
conversation changes.

## The defect is mine, and it is the one the project already fixed

The 2026-09-18 gate failed for exactly this reason: requirements written by reading the reply
encode what that one answer happened to say. The 2026-09-19 fix was a worksheet that withholds
the reply, and it worked -- the published set's reply-only share is 4.9%.

The sea-ice worksheet could not withhold the reply, because `reply_contexts` is a judgement about
what the reply drew on and cannot be made without seeing it. So the reply was shown -- and
`answer_requirements` were asked for on the same page. Both tasks were legitimate; putting them
on one page was not.

**The fix is two passes over the same checkpoints:**

1. **with the reply shown** -- label `reply_contexts` only. This pass is done and its quality is
   high; it is the part that could not be derived, and the derivation from it was mechanical.
2. **with the reply withheld** -- label `answer_requirements` from the query and the conversation
   strictly before it. This is the pass that has not happened.

Until the second pass runs, this dataset's coverage numbers mean nothing and must not be
reported. The 44 checkpoints, the 7 contexts and the `reply_contexts` labels are all sound and
worth keeping; it is the requirements that need redoing.

## What is not affected

The routing labels. `query_type` and `relation_label` were derived from the annotator's
`reply_contexts` by the protocol's table, and that derivation was checked under both readings of
the disputed `ctx-skill` context with identical results. The context catalog and the members are
the annotator's and are unchanged by any of this.
