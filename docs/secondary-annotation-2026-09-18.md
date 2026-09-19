# Independent answer annotation audit — 2026-09-18

## Provenance

- Source: `context-router-second-annotator-blind.xlsx`
- SHA-256: `16a22eec59f25e104c2ffbf571a9b0269da1b2f331d3e2c4d1c23a4eff1f2d7f`
- Selection seed: `context-router-second-annotator-v1`
- Coverage: 6 of 26 checkpoints (23.1%)
- Annotator id: `annotator-2`
- Annotator type: human (confirmed by the user on 2026-09-18)
- Independence: user-reported; the workbook itself does not record prior-context exposure

The original workbook remains outside the repository. The committed JSON preserves only the
blinded ids, mapped sample ids, statuses, requirement text and non-identifying provenance.

## Mechanical audit

| Check | Result |
|---|---:|
| Completed checkpoint records | 6 / 6 |
| Status agrees with whether a reference reply exists | 6 / 6 |
| Submitted requirements | 11 |
| Requirements supported by their own reference answer | 11 / 11 |
| Primary requirements recovered from the second label text | 4 / 13 |
| Second requirements recovered from the primary label text | 1 / 11 |

The last two rows are lexical diagnostics, not semantic agreement scores. They show that the
two label sets cannot be substituted for one another without adjudication.

## What the disagreement says

- `B01` captures the numerical discrepancy but omits the requested code fixes and unfinished
  functionality.
- `B02` identifies three method families, but the first label nests Chinese quote marks and the
  three labels omit several discriminating qualifiers used by the primary set.
- `B03` puts several independently checkable facts into one long copied sentence. It covers both
  primary requirements but is not a minimal label.
- `B05` captures the comparison and BDF reference, but does not separately preserve the
  conservation and grid-convergence requirement.
- `B06` selects apology/process sentences and `dt` stability while omitting the saved-version
  timeline, the changed `dr`, and the `0 / 35` convergence result. This is a substantive label
  selection disagreement, not a formatting discrepancy.
- `B04` correctly identifies that no substantive assistant answer exists.

## Adjudication received

The completed `context-router-answer-label-adjudication.xlsx` was imported as
`datasets/real-replay/adjudication-2026-09-18.json` (SHA-256
`2ce2ec443fe163ea901f1c4d3ee2007985cdef30cf6dff7c095e860b369bd945`). All six records are
complete and all 13 final requirements are supported by their reference replies. The final
requirements for every record exactly match the existing primary gold labels, so adjudication
changes no gold-label text. The canonical checkpoints now record the second annotator and the
otherwise-unknown adjudicator; both raw label sets remain untouched.

The source workbook selected “use secondary” for B06 even though its three final requirements
exactly match the primary label. The user confirmed on 2026-09-18 that “use primary” was the
intended decision. The imported adjudication records that correction in its notes; the final
requirements and benchmark labels are unchanged. The audit now has zero decision inconsistencies.

`annotator-2` is a user-confirmed human and worked through the blinded worksheet. The project's
20% independent-human coverage floor is therefore met (6/26 = 23.1%). The independent-label and
adjudication stages are complete; the real-replay answer-quality gate now requires a documented,
pinned model.
