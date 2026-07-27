# ATCT Narrative Fingerprint

**文章がどのような履歴依存構造で生成されたかを測るチェッカー**

ATCT Narrative Fingerprint treats a document as an irreversible trajectory,
not a bag of interchangeable sentences:

\[
s_1 \rightarrow s_2 \rightarrow \cdots \rightarrow s_T
\]

It does **not** claim to prove whether AI or a human wrote a text. It measures
how strongly the observed structure depends on sentence order, long-range
history, local-to-global coherence, turning points, and semantic recurrence.

## Six diagnostic families

| Diagnostic | Question |
|---|---|
| Order sensitivity | Does the trajectory change under sentence shuffling? |
| Reverse sensitivity | Does reversing the text destroy its direction? |
| Long-history dependence | Do wide causal windows differ from short ones? |
| Local/global consistency | How do sentences align with the global theme? |
| Turning-point magnitude | Do apparent transitions move in vector space? |
| Semantic redundancy | Do non-adjacent sentences repeat the same meaning? |

The analyzer uses causal windows \(w \in \{3,5,8,13\}\). Raw values are the
research features. The 0–100 values are bounded display transforms, **not
calibrated probabilities**.

## Install

```bash
python -m pip install -e .
```

For the recommended multilingual semantic backend:

```bash
python -m pip install -e ".[semantic]"
```

## Analyze a document

Fast, deterministic lexical smoke run:

```bash
atct-fingerprint analyze examples/sample_ja.txt --encoder tfidf
```

Semantic run with multilingual E5:

```bash
atct-fingerprint analyze examples/sample_ja.txt --encoder e5
```

Output is JSON containing raw features, window-level perturbation results,
display scores, seed, shuffle count, and an explicit non-authorship warning.

## Test the minimal research proposition

The fixed proposition is:

> Do ATCT order-dependence features improve AUROC on an unseen AI model by at
> least 0.05 over a document-level lexical TF-IDF baseline?

Prepare an explicit train/test CSV using
[`examples/dataset_template.csv`](examples/dataset_template.csv), then run:

```bash
atct-fingerprint evaluate data/confirmatory.csv --encoder e5
```

The evaluator reports:

- lexical baseline AUROC;
- lexical + ATCT AUROC;
- sentence-shuffled control AUROC;
- incremental improvement and mechanism drop;
- `supported`, `unsupported`, or `mechanism_falsified`;
- source-leakage, genre, and length audits.

It rejects a test split whose AI model identifier also occurs in training.
See [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md) before collecting
or interpreting data.

## Current scope

This is an MVP for feature extraction and falsifiable evaluation design.
It has no trained authorship model and includes no benchmark corpus. Claims
about robustness to paraphrasing, human editing, translation, or a particular
AI model require controlled data and confidence intervals.

## Development

```bash
python -m unittest discover -s tests -v
```

CI runs the unit suite and CLI smoke test on Python 3.10 and 3.12.
