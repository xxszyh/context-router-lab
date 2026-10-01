# The source conversations are being deleted, and the store is the only copy

2026-10-01. Found while standing up the second conversation's experiment: the plan was to
re-ingest its events into a separate store, and the ingest returned `discovered_sessions: 0`.
The project directory was gone.

## What is gone

| conversation | events | Claude Code project directory | still in the store |
|---|---:|---|---|
| 实训3 北极海冰 | 1,209 | **deleted** | yes |
| 实训6 人口预测 | 327 | **deleted** | yes |
| 实训7 路径规划 | 1,037 | present | yes |
| cumcm2026 (the published dataset) | 3,104 | present | yes |
| this project's own session | 2,125 | present | yes |

Claude Code removes project directories on its own retention schedule -- the surviving ones are
dated 2026-08-30, 2026-09-11 and later, while the two that went missing were dated 2026-09-16. The
conversations were imported into `.local/real-answer-experiment.sqlite` on 2026-09-19 and 2026-09-21,
which is the only reason their events still exist at all.

## Why this matters more than it looks

**`.local/` is gitignored.** The repository's rule is that real records are never committed, and
it is a good rule -- but it means the store is a single copy on a single disk, and the thing it
is the only copy of has already been deleted once by a process nobody was watching.

The published dataset is in better shape than it appears: `datasets/real-replay/claude-d22f2593.json`
is committed and carries the contexts, the members and the labels, and its source directory still
exists. But the members are event ids, so the dataset without the store is a set of pointers into
nothing.

## What this changes

**The enlargement plan loses two of its three candidates from disk.** `docs/enlargement-2026-09-21.md`
counted 1,209 + 1,037 + 327 events across three conversations and estimated about twenty more
checkpoints. All three are still in the store, so the plan is unchanged in substance -- but it can
no longer be executed by re-importing, and any further conversation not yet imported is at risk of
disappearing before it is.

**The store needs a copy that is not `.local/`.** It cannot be the repository, which is public.
It can be anywhere else: another disk, an encrypted archive, a private remote. This document does
not do that; it records that it has not been done.

## The immediate consequence for the work

The second conversation's experiment was built by **copying its events out of the shared store**
rather than re-ingesting them. That path works and is what `build_seaice_benchmark.py` does, but
it is a fallback that only exists because the import happened in time -- the next conversation may
not have that luck.
