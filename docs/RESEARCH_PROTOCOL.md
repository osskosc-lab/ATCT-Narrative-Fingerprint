# Research protocol

## Minimal proposition

> Do ATCT order-dependence features improve discrimination of text from an
> unseen AI model over a lexical TF-IDF baseline?

The first confirmatory run fixes one proposition, one primary metric, one
baseline, and one falsification experiment.

| Item | Pre-registered choice |
|---|---|
| Primary metric | AUROC |
| Baseline | document-level character TF-IDF + logistic regression |
| Intervention | add seven ATCT structural features |
| Falsification | independently shuffle sentence order before ATCT extraction |
| Support threshold | combined AUROC − baseline AUROC ≥ 0.05 |
| Mechanism failure | combined AUROC − shuffled-control AUROC ≤ 0.01 |

## Required CSV

The evaluator accepts UTF-8 CSV with these columns:

| Column | Meaning |
|---|---|
| `text` | complete document with at least four sentences |
| `label` | `0` human, `1` AI-generated or the declared positive class |
| `split` | explicit `train` or `test`; never assigned by the evaluator |
| `source` | human corpus or AI model identifier |
| `genre` | controlled genre identifier |

AI `source` values must be disjoint between train and test. The evaluator
raises an error instead of silently accepting source leakage. Both splits must
contain both labels.

## Controls that remain the researcher's responsibility

- Match genre, language, prompt, document length, sampling temperature, and
  editing condition.
- Keep prompt families from crossing splits.
- Report every excluded document and exclusion rule.
- Run multiple fixed seeds and bootstrap confidence intervals before making a
  research claim.
- Test AI-only, human-only, AI-edited-human, and human-edited-AI conditions
  separately.

## Interpretation

`supported` means only that the fixed dataset passed both numerical gates. It
does not prove AI authorship for an individual document. `unsupported` means
incremental AUROC was below 0.05. `mechanism_falsified` means the gain survived
sentence shuffling within the configured epsilon, so order dependence was not
the demonstrated mechanism.
