# Field audit 2026-09-23 → v0.7.1

Real Japanese essays were run with TF-IDF, seed=11, 200 shuffles.
This note records what the checker can claim and what it must not.

## What held

- 40+ sentence essays with repeated lexical motifs produce `content_z ≥ 2`.
- 6-sentence literary sample does not. Short-document `Z` is not a production score.
- Synthetic vector tests still isolate current-sentence exclusion and shuffle nulls.
- 66 pre-existing unit tests remain the CI contract.

## What failed on real prose

| Failure | Example | v0.7.1 change |
|---|---|---|
| `single_cause_overcompression` on a list mention | 「人間関係も」 in an enumeration | monopoly warning requires support/confidence gates; unknown-only primaries do not warn |
| unknown cause from outline language | 「理由は、だいたい三つに分けられる」 | unknown cue no longer fires on enumerated outlines |
| missed somatic health cue | 「身体が先に落ちていた」 | health pattern includes 身体が落 / 身体が先に |
| missed desire | 「安心したい」「つながりたい」 | desire lexicon adds those actions |
| title hedge read as weak exclusive | 「〜なのかもしれない」 | scope non-exclusive class includes hedges |
| `v07_component_index` read as quality | 0.02–0.08 on competent essays | still rule coverage only |

## Writing-desk channels

Use separately: content_z, title/body mismatch, cause_monopoly+warning, transformation×autonomy.
Do not average into a 0-100 writing score.

## Still pending

- annotated title/body Macro-F1
- Japanese E5 split/merge tolerance
- metaphor families beyond OS
- persuasion branded-solution false positives
