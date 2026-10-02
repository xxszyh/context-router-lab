# The variable that decides it is corpus size, not noise

2026-10-02. Offline, deterministic, no judge, no API. SCALE-QA, `arrangement=interleaved seed=20260420`,
window 256 (depth 5), `embedder=neural`.

`docs/discriminating-set-2026-10-02.md` ended on something that did not replicate: on the
1,000-question package the router's hard-subset failures were **24 candidate misses against 37
selection losses**, and on the 3,000-question package they were **147 against 124**. The obvious
suspect was the filler -- the first package is 26.7% truth and the second 77.9%, because the two
were built to the same budget with three times the evidence in one of them. The sweep below was run
to test that. **The suspect is innocent, and the sweep names the variable that is guilty.**

## Two controlled axes

Both use SCALE-QA's own builder, offline, with its bundled noise. One router pass per package, over
that package's hard subset (the questions one lexical lookup does not already answer at depth 1).

- **The noise axis** holds the question set at 3,000 and varies only the context budget, which is
  what decides how much filler the stream has room for.
- **The corpus axis** holds the truth share near 78% and varies the number of questions -- and so
  the number of evidence-bearing dialogues in the index.

```bash
python build_eval_dataset.py --context-budget 400k --num-questions 3000 --output-dir pkg_400k_3000
python build_eval_dataset.py --context-budget 168k --num-questions 1000 --output-dir pkg_168k_1000
```

### The noise axis: nothing moves

| truth share | windows | hard | gold in candidates | recovered | selection loss | **candidate miss** |
|---:|---:|---:|---:|---:|---:|---:|
| 98.4% | 3,033 | 598 | 0.763 | 0.569 | 0.196 | **0.237** |
| 77.9% | 3,561 | 602 | 0.754 | 0.550 | 0.206 | **0.246** |
| 52.6% | 4,785 | 600 | 0.757 | 0.543 | 0.215 | **0.243** |
| 40.6% | 5,884 | 620 | 0.758 | 0.550 | 0.210 | **0.242** |
| 27.4% | 8,195 | 613 | 0.752 | 0.535 | 0.219 | **0.248** |

Across a 3.6× range in truth share and a 2.7× range in window count, the candidate-miss rate moves
by **0.011** where the standard error at n ≈ 600 is about 0.017. The amount of filler in the stream
does not move where the router's loss is, at all.

### The corpus axis: it doubles

| evidence dialogues | truth share | windows | hard | gold in candidates | recovered | selection loss | **candidate miss** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 79.3% | 1,175 | 168 | 0.869 | 0.679 | 0.196 | **0.131** |
| 1,500 | 79.1% | 1,759 | 257 | 0.840 | 0.588 | 0.257 | **0.160** |
| 2,000 | 79.0% | 2,351 | 370 | 0.800 | 0.597 | 0.205 | **0.200** |
| 3,000 | 77.9% | 3,561 | 602 | 0.754 | 0.550 | 0.206 | **0.246** |

**Monotone in the size of the evidence corpus, and it is only the front end that moves.** Candidate
miss nearly doubles; selection loss is flat across the same 3× range (0.196, 0.257, 0.205, 0.206).
Growing the corpus makes the router fail to *offer* the evidence more often. It does not change what
the router does with a candidate list once it has one.

### The control that separates the two

`pkg_512k_1000` is the same 1,000 questions as `pkg_168k_1000` with 26.7% truth instead of 79.3%,
and 2,810 windows instead of 1,175:

| | evidence dialogues | truth share | windows | candidate miss |
|---|---:|---:|---:|---:|
| `pkg_168k_1000` | 1,000 | 79.3% | 1,175 | **0.131** |
| `pkg_512k_1000` | 1,000 | 26.7% | 2,810 | **0.152** |
| `pkg_512k_3000` | 3,000 | 77.9% | 3,561 | **0.246** |

Tripling the filler at a fixed corpus moves the rate by 0.021, which is inside the standard error of
0.027 at that sample size. Tripling the corpus at a fixed truth share moves it by 0.094. **The
corpus decides; the filler does not.**

## The mechanism, and it is not subtle

The distractors that matter are the other dialogues. A SCALE-QA noise block is UltraChat: a
different corpus, a different register, and easy for BM25 and for an encoder to reject. Another
SCALE-QA dialogue is a **near-miss** -- same benchmark, same construction, same ten topics, similar
length -- and 3,000 of them compete for the same ten candidate slots that 1,000 did. Filler inflates
the window count without competing; evidence inflates it by competing.

## What this changes

1. **"Candidate generation is not the bottleneck" is a statement about corpus size, and it was
   written without one.** It is true at 1,000 evidence dialogues (0.131) and false at 3,000 (0.246).
   The sentence appears in `docs/discriminating-set-2026-10-02.md` and is qualified there.
2. **The number is now reported rather than left to the reader to infer.** `ArrangementReachability`
   carries `truth_blocks`, and `ctxlab scale-qa` prints it in the header beside the truth share:

   ```
   arrangement=interleaved seed=20260420  stream=3561 windows ... (77.9% truth, 3000 evidence dialogues)
   ```

   This is the third instrument in the same family, after `recency_reachability` and the difficulty
   split, and it is the same move each time: **name the number whose absence lets a benchmark answer
   a question it cannot answer.**
3. **The three ordering results hold across all ten packages.** `rrf_score` below `lexical_rank`
   everywhere it is measurable (p < 0.001 wherever n >= 600); `calibrated_probability` above
   `rrf_score` everywhere (same); and the calibrated order never separable from the lexical channel
   -- p between 0.080 and 1.000, point estimates between -0.043 and +0.005. Those are findings about
   the pipeline, not about the corpus, and this is the replication that says so.

## The hypothesis, withdrawn

The sweep was run to test "the composition of the stream decides where the loss is", with composition
read as the ratio of evidence to filler. That reading is wrong, and it was the obvious one. What
composition turns out to mean is **how many sessions carry evidence** -- a property of the question
set, not of the budget, and not reported by any benchmark this project has touched.

*Grade: the recalls are exact-evidence containment, deterministic and offline. Each point is one
package and one router pass, so the curve is ten measurements and not a distribution; the standard
errors quoted are the binomial ones at the stated n. The two axes are not fully orthogonal --
changing the question count changes the truth tokens, so holding the truth share fixed means
choosing the budget -- which is why the 1,000-question control is included: it varies filler at a
fixed corpus and settles the attribution directly.*
