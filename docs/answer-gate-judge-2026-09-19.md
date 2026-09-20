# The paid blinded judge on the query-derived labels

2026-09-19. The query-derived label set passed its offline gates (scoreability, contamination,
discrimination -- `docs/query-derived-labels-2026-09-19.md`), so the paid judge was re-run
against it. This is the first answer-quality measurement on real conversations whose labels are
not capped at the floor. The result is recorded whole, including the part that does not flatter
the routing claim.

## What ran

| | |
|---|---|
| judge | `deepseek-v4.1-flash` (gateway echoes `deepseek-v4-1-flash-260910`), swapped-pair protocol |
| pair | `hybrid_router` vs `full_history` |
| answers | the 2026-09-18 run, untouched -- only the requirements changed |
| labels | query-derived, 2026-09-19 |
| pairs | 18, judged in both orders = 36 calls |
| cost | 26,672 in / 36,272 out judge tokens; 0 parse failures; 1 refusal-gate hit |

Output: `.local/real-answer-judge-hybrid-vs-full-querylabels-2026-09-19.json`. It was recorded here
as "clean, committable" on the day, and that was **not verified at the time** -- re-checked on
2026-09-20 it tripped the export gate on the drive-path pattern, which turned out to be a false
alarm in the gate rather than a leak in the file. The judge's arm label followed by a newline is
stored as a letter, a colon and an escaped newline -- five characters that are, to a
shape-matching pattern, indistinguishable from a Windows drive path. The file contains no path:
one hit in the raw text, zero in the parsed values. The gate was fixed to read JSON at the layer
its content lives in, and the claim holds under it. See
`docs/documented-model-selection-2026-09-20.md`.

## The verdict, against the old-labels baseline

| | pairs | judged | agreed | ties | hybrid wins | full wins |
|---|---:|---:|---:|---:|---:|---:|
| old labels (2026-09-18) | 18 | 17 | 14 | **14** | 0 | 3 |
| **new labels (2026-09-19)** | 18 | 17 | 14 | **9** | 2 | 5 |

Two things are true at once, and both matter.

**The label fix worked.** Under the old labels 14 of 17 judged pairs tied and neither arm won a
single answerable checkpoint -- the floor. Under the new labels the ties drop to 9 and the judge
now reaches a verdict on 7 pairs (5 for `full_history`, 2 for `hybrid_router`). The judge's
order-agreement is unchanged at 82% (14/17), so the instrument is as stable as before; what it
can now *see* is the difference the old labels hid.

**The direction favours `full_history`, 5 to 2.** This is the part that must not be smoothed
over. On the checkpoints where a correct answer needs the prior conversation, the arm that reads
everything out-scores the arm that reads 221x less, more often than not.

## Where `full_history` wins is the uncomfortable part

The two `must_refuse` checkpoints -- `q-1227` (can the whole problem be done with a PINN) and
`q-1370` (can COMSOL help) -- were both won by `full_history`. These are the questions whose
correct behaviour is to decline from insufficient grounds, and they are the questions a router
is *most* supposed to get right cheaply. That `full_history` took both says the selective arm is
not yet declining when it should, or is declining when it should engage. At n = 2 this is a
pointer, not a finding, but it is the sharpest pointer in the run.

## The two instruments now disagree on direction, and that is the honest headline

The free coverage judge and the paid LLM judge ran the **same answers against the same new
labels** and came out opposite:

| instrument | hybrid | full | ties | what it measures |
|---|---:|---:|---:|---|
| coverage (lexical) | 9 | 2 | 6 | did the answer name the right identifiers |
| LLM judge | 2 | 5 | 9 | did the answer actually satisfy the requirement |

`hybrid_router` writes short, dense answers that hit the checkable nouns, so the lexical scorer
credits it. The LLM judge, asked whether each requirement is *actually addressed*, credits the
fuller answer that `full_history`'s larger context produces. Neither is wrong; they measure
different things, and this run is the first time they have diverged on real data in a way that
can be pointed at. It is the concrete reason the README's rule -- never report coverage as
answer quality -- is load-bearing.

## What is still not established

The 5–2 is **not** a conclusion that `full_history` is better, and the routing claim is **not**
refuted. Three reasons, all measured above:

1. **n = 18 pairs, 7 decisions.** A net difference of 3 decisions over 7 decided pairs is within
   the noise of a judge that disagrees with itself 1 in 5 times.
2. **The judge is a private proxy alias.** `deepseek-v4.1-flash` is not a documented, fixed
   model; the numbers would move under one. Nothing here is publishable until that changes.
3. **The strata are thin.** The one-refuses stratum, where the router's failure mode would show
   most clearly, has 6 pairs and 67% agreement -- too noisy to read.

## Next, in the order decided

1. **Re-run the whole gate under a documented, fixed model.** The blocker was that
   `deepseek-v4.1-flash` is a private proxy alias. The gateway does not accept DeepSeek's
   documented public id (`deepseek-flash` → `400 模型不存在` -- it is a proxy with its own
   naming, not the official endpoint), but it serves several **publicly documented** model ids
   confirmed answering with text on 2026-09-19: `deepseek-v3.2`, `qwen3-max`, `kimi-k2.5`,
   `glm-4.7`, `minimax-m2.5`, and the dated `deepseek-v4-flash-0731`. Re-answer + re-judge under
   one of these so the result is reproducible off the proxy alias.
2. **Enlarge the sample.** The 6 labelled checkpoints that never entered the gate
   (`q-0219, q-1426, q-1892, q-1944, q-1998, q-2450`) and any new conversations raise n from 20.
3. Only then read a quality direction off the result. Until both happen, the honest statement is:
   **routing cuts tokens by ~221x and never fails to run where `full_history` fails 10% of the
   time; whether it preserves answer quality is measurable now and unresolved.**
