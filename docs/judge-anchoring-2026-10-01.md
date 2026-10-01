# Anchoring the judge: where the order disagreement came from, and what removed it

2026-10-01. `quality-verdict-2026-10-01.md` closed with a target rather than a result: the
judge was the binding constraint, and the thing that would move the quality claim was "a judge
whose `mean_consistency` is above 0.80 on every conversation rather than on one of them". The
published conversation sat at **0.7971**. This is the work that took it to **0.8913**, and the
reason it is not just a number going up.

## The target was gameable, so it was not the target

`mean_consistency` is the mean share of a pair's six calls that landed on the modal verdict. A
judge that answered `tie` every time would score **1.000** and be worthless. So the number was
treated as a constraint to satisfy while three things that cannot be gamed were watched at the
same time: the tie rate, the decisions both runs made, and a hand read of specific pairs.

The last of those is the check the project's own error log records as missing the last time
consistency was chased: four judge rounds each watched a different number, and the run with a
perfect 100% consistency was reported without anyone noticing the judge was wrong on the only
pair that could be checked by hand.

## The diagnosis was free, and it decided the fix

`run_pairwise_judging` appends the forward verdict and then the reversed one, per repeat, so a
three-repeat pair's six stored verdicts are `[f0, r0, f1, r1, f2, r2]`. Splitting them recovers
each order's own triple, which separates the two ways a judge can contradict itself:

| | unanimous | noise, one order | noise, both orders | position bias |
|---|---:|---:|---:|---:|
| published `d22f2593` | 10 | 9 | 4 | **0** |
| sea ice `139471e7` | 25 | 11 | 3 | 4 |

- **position bias** — each order is internally unanimous, the two orders disagree. Repetition
  cannot remove this, because the bias *is* the signal; only the prompt can.
- **sampling noise** — one order's own repeats disagree. Repetition is the remedy and the
  prompt is not.

**On the published conversation every one of the thirteen contradictions was sampling noise,
with no position bias at all.** So repetition was already the right tool and the remaining
variance was somewhere else. Three cheaper explanations were checked and eliminated before the
prompt was touched:

- **temperature** — already `0.0` (`providers/anthropic.py`), and the judge is stochastic anyway;
- **truncation** — all 138 calls ended `end_turn`, output 68–416 tokens against a 4096 cap;
- **parsing** — `parse_failures` was 0.

## What the judge was actually doing

Reading the six replies for a contradicted pair showed the variance is in the **reading**, not
in the final decision. On `q-0157`, three forward calls on identical input counted A as
satisfying **2**, then **0**, then **2** requirements. On `q-1760` the judge attributed the same
statement to A in one call and to B in another.

The prompt asked for exactly that. `JUDGE_INSTRUCTIONS` said:

> For each, note in one short line whether A satisfies it and whether B does. **Do not restate
> either answer.**

The judge was forbidden from quoting the words that decided each requirement, so every call
re-derived "satisfied" from an unanchored impression. The README had already recorded the cost
of this trade once, from the other side: bounding the judge's analysis cut output tokens 64% and
took order agreement down with it.

## The change

`--judge-style anchored` — the same blinding, the same three rules, the same final line, and one
difference: a requirement may not be marked satisfied without a quote from the answer that
decides it. `terse` remains the default so that every number already published stays
reproducible under the prompt that produced it.

## The result on the published conversation

Same answers file, same model, same pair, same three repeats. The prompt is the only variable.

| | terse | anchored |
|---|---:|---:|
| **`mean_consistency`** | **0.7971** | **0.8913** |
| pairs agreeing on all six calls | 10 | 14 |
| ties | 12 | 12 |
| decided | 12 | 12 |
| `hybrid_router` wins | 7 | 9 |
| `query_recent_only` wins | 5 | 3 |
| judge output tokens | 19,623 | **72,448** |

Three things make this more than a number moving:

**It is not a hedge.** The tie count is unchanged at 12. A judge that became more consistent by
declining to decide would show a rising tie rate, and it does not.

**The decided pairs did not move.** Nine pairs were decided by both runs and **zero flipped**.
Repetition was validated the same way and for the same reason: a fix that moves the decisions is
a different question, not less noise.

**The judge got more correct, not just steadier.** Five pairs were read by hand against the
rubric:

| pair | hand read | terse | anchored |
|---|---|---|---|
| `q-0157` | tie | tie ✓ | tie ✓ |
| `q-1841` | a | tie ✗ | a ✓ |
| `q-1760` | tie | tie ✓ | tie ✓ |
| `q-1227` | a | tie ✗ | a ✓ |
| `q-1897` | a | tie ✗ | a ✓ |
| | | **2/5** | **5/5** |

The last two were read after the anchored verdicts were already visible, which is an anchoring
risk, so they were put to a second reader given only the question, the rubric and the two
answers. It returned **A** on both, on its own reasoning.

The full hand table, both conversations, is at the end of this document.

**No position bias was introduced.** Among calls that picked a winner rather than a tie, the
share going to position A is 0.545 under `terse` and **0.442** under `anchored` — both near
0.5, and the anchored judge is the less A-leaning of the two. The three tie→decided pairs all
resolving to `a` is therefore a reading of the content, not of the layout; the tally move from
7-5 to 9-3 is what the hand checks above are checking.

## The same run on the sea-ice conversation, and why it is not the same result

Same answers, same model, same three repeats, 264 calls.

| | terse | anchored |
|---|---:|---:|
| **`mean_consistency`** | **0.8527** | **0.8643** |
| pairs agreeing on all six calls | 25 | 27 |
| ties | 18 | **21** |
| decided | 26 | **23** |
| `hybrid_router` wins | 9 | 9 |
| `query_recent_only` wins | 17 | **14** |
| judge output tokens | 41,491 | **166,284** |

**Anchoring does a different job on this conversation, and the difference is the finding.** The
published conversation had **zero** position-bias pairs and thirteen noise pairs; anchoring
removed five of the noise pairs and left the tie count untouched. Sea ice had **four**
position-bias pairs, and anchoring removed all four — but left `noise_one_order` at eleven,
*raised* `noise_both_orders` from three to five, and raised the tie count by three.

| failure mode | published terse → anchored | sea ice terse → anchored |
|---|---|---|
| unanimous | 10 → 14 | 25 → 27 |
| noise, one order | 9 → 7 | 11 → 11 |
| noise, both orders | 4 → 1 | 3 → **5** |
| **position bias** | 0 → 1 | **4 → 0** |

So it is not a uniform improvement. It targets the failure it was designed for — a verdict not
anchored in the text — and a pair whose disagreement is *between* orders rather than within one
is only partly that failure. On the four sea-ice pairs it fixed, the outcome was better: three
of the four became decisive (`q-0145` and `q-0465` unanimous, `q-0392` five of six) where
`terse` had collapsed all four into ties.

**The tie rise on sea ice is the hedge risk, and it is real.** Three pairs `terse` decided
unanimously became ties under `anchored` (`q-0109`, `q-0240`, `q-0621`). A consistency gain
bought by declining to decide is the failure mode this whole exercise was supposed to avoid.

## The tally moved toward the routing arm on both conversations, so it was checked

| | hybrid | recent | ties | decided | p |
|---|---:|---:|---:|---:|---:|
| published, terse | 7 | 5 | 12 | 12 | 1.00 |
| published, anchored | 9 | 3 | 12 | 12 | 1.00 |
| sea ice, terse | 9 | 17 | 18 | 26 | 0.17 |
| sea ice, anchored | 9 | 14 | 21 | 23 | 0.40 |
| **pooled, terse** | 16 | 22 | 30 | 38 | 0.42 |
| **pooled, anchored** | **18** | **17** | 33 | 35 | **1.00** |

Both conversations moved toward `hybrid_router`, on the one where the routing claim was already
ahead and on the one where it was behind. A change that moves a result in the direction the
project hopes for has to be checked rather than welcomed, so the pairs that moved were read.

The three sea-ice pairs that carried the movement — one that `terse` tied and `anchored`
decided, and two that `terse` gave to `query_recent_only` and `anchored` tied — were put to a
reader given only the question, the rubric and the two answers:

| pair | independent read | terse | anchored |
|---|---|---|---|
| `real-139471e7-q-0392` | **A** | tie ✗ | a ✓ |
| `real-139471e7-q-0621` | **TIE** | b ✗ | tie ✓ |
| `real-139471e7-q-0924` | **TIE** | b ✗ | tie ✓ |

**All three agree with `anchored`.** Across both conversations the independent reads agree with
`terse` on **2 of 8** pairs and with `anchored` on **8 of 8**. The movement is a better reading
of the same answers, not a prompt that flatters one arm — and the position-A share of decided
calls, which would show it if it were positional, is 0.545 → 0.442 on the published
conversation and 0.525 → 0.538 on sea ice, both near 0.5.

## The fix that was ruled out, for free

The obvious next move was to stop asking the judge for a verdict at all: have it mark each
requirement and compute the winner in code from the marks, since the rule it was given — "a" if
A satisfies strictly more requirements — is already mechanical. That would delete the last line
of every reply, which is where a gestalt decision could drift.

**It cannot work, and the argument needs no new calls.**

1. The judge's verdict is a **faithful function of the marks it writes down**. On the calls
   where it states counts (`A 满足2项，B 满足1项`; `A: 0/3, B: 0/3`), the stated verdict follows
   the stated counts in **29 of 29** readable calls, across all four runs. Zero violations.
2. The verdict is nevertheless **unstable** — that is the whole measurement above.
3. Therefore the marks are unstable. Computing the winner from them reproduces exactly the
   variance that is already there.

The residual noise is in **reading a requirement**, not in **deciding the comparison**, and no
amount of output-format engineering reaches it. What would is a label change — several of these
requirements are compound (`必须在「第三题灵敏度分析」下放置相关代码、结论和解释文档` bundles
three checkable things), and a partial answer to a compound requirement is genuinely ambiguous,
which is the condition under which a reader lands differently on different passes. That is an
annotation question, not a judge question, and it belongs with the free parameters
`state-of-play` already records for the lexical label.

The counts parser reached only the calls that state counts, so (1) rests on 29 calls out of 524
and is reported with its coverage. It is the judge's own summary of its own marks, which is the
best evidence available without asking for structured output.

## What it costs, and what it does not fix

**3.7–4× the judge's output tokens**: 19,623 → 72,448 over 138 calls on the published
conversation, 41,491 → 166,284 over 264 on sea ice. Input is unchanged. This is the trade the
README recorded being made in the other direction; it is now being made back, knowingly.

**`q-0576` got worse**, 0.833 → 0.500, and two pairs that `terse` decided became ties. The
anchored judge is better on average and not better everywhere, and the average is over 24 pairs.

**One pair on the published conversation now shows position bias where none did before.** It is
one pair, and the aggregate moved the other way — five pairs' worth the other way — but it is
not zero, and the honest statement is "position bias fell from four pairs to one across both
conversations", not "position bias was eliminated".

**`mean_consistency` is still 0.891 and 0.864, not 1.0.** The instrument improved; it did not
become good. A judge whose calls disagree one time in eight is still the ceiling on how fine a
distinction the quality claim can carry, and that ceiling has not moved far.

## A run that was lost, and the hole it exposed

The first sea-ice re-run died at **exit 1 after 51 minutes and 264 calls** on an
`httpx.ReadTimeout`. The CLI writes its artefact once, at the end, so the whole run and the
whole bill were discarded by a single network hiccup at some call in the middle.

The judge already retried one kind of failure and not the other: an unparseable reply was
retried, a request that never completed was not. `providers/transport.py` now retries
transport failures with exponential backoff at every `client.post` in both provider modules.

**Only transport failures.** `httpx.HTTPStatusError` is excluded on purpose, and the exclusion
is load-bearing rather than incidental: a 400 is the server's answer, and one of this project's
results is measured in exactly those — `full_history` failing to run against a documented
model's ceiling is a count of HTTP 400s. A retry loop around status errors would turn that
finding into a timeout and quietly delete it. `tests/test_transport.py` pins both halves: a
timeout is retried, a 400 is answered once.

## The prompt is now a recorded parameter

Every judge number this project has published was produced by a prompt that was recorded
nowhere — the same defect `state-of-play` records for the lexical signal's rendering and
threshold, where three unrecorded free parameters moved the result further than the signal's own
behaviour did. Judge artefacts now carry `judge_style` and `judge_instructions_sha256`, and the
run above is the first that can say which instrument produced it.

## The full hand table

| pair | read by | verdict | terse | anchored |
|---|---|---|---|---|
| `real-d22f2593-q-0157` | hand, before either verdict | tie | tie ✓ | tie ✓ |
| `real-d22f2593-q-1760` | hand, before either verdict | tie | tie ✓ | tie ✓ |
| `real-d22f2593-q-1841` | hand, before either verdict | a | tie ✗ | a ✓ |
| `real-d22f2593-q-1227` | hand, then independently | a | tie ✗ | a ✓ |
| `real-d22f2593-q-1897` | hand, then independently | a | tie ✗ | a ✓ |
| `real-139471e7-q-0392` | independent only | A | tie ✗ | a ✓ |
| `real-139471e7-q-0621` | independent only | tie | b ✗ | tie ✓ |
| `real-139471e7-q-0924` | independent only | tie | b ✗ | tie ✓ |
| | | | **2/8** | **8/8** |

"Independent" means a reader given the question, the rubric and the two answers, and nothing
else — no arm names, no token counts, no verdicts. The three sea-ice rows and the two
published rows marked *hand, then independently* are the five where the first read risked
being anchored on a verdict already seen.

## Consequence

`quality-verdict-2026-10-01.md`'s tallies were produced by the `terse` prompt and are
superseded: at the same three repeats, the anchored prompt puts the published conversation at
9-3 rather than 7-5, and the pool at **18-17 rather than 16-22**. The null result is therefore
stronger than that document could state, not weaker — 35 decided pairs, p = 1.00, where the
terse instrument had it leaning toward recency at p = 0.42.

**And the target that started this is met.** `mean_consistency` is above 0.80 on both
conversations under both prompts, and 0.8913 / 0.8643 under the anchored one.

Nothing here changes the project's solid ground. The 221× token cut and the 72–79%
failure-to-run rate for `full_history` are properties of the arms and the model's context
ceiling, and no judge can affect them.
