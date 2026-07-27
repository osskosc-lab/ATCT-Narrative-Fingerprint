# ATCT Narrative Fingerprint v0.2

**文章がどのような履歴依存構造で生成されたかを測るチェッカー**

v0.2 changes the central question from:

> Does this document have order structure?

to:

> Is the original history-conditioned coherence significantly higher than
> content-preserving order controls?

The primary metric is:

\[
Z_{\mathrm{order}} =
\frac{
C_{\mathrm{original}}-\mu(C_{\mathrm{random}})
}{
\sigma(C_{\mathrm{random}})+\varepsilon
}
\]

where the state used to evaluate sentence \(x_t\) is built only from preceding
sentences:

\[
h_t^{(w)} =
\frac{\sum_{j=1}^{w}\alpha_j x_{t-j}}
{\sum_{j=1}^{w}\alpha_j},
\qquad
C_t^{(w)}=\cos(x_t,h_t^{(w)}).
\]

The current sentence is never included in its own history state.

## Interpretation boundary

This is not an AI-authorship detector, probability, or writing-quality score.
It reports statistical evidence and locations for order-conditioned narrative
structure. The former fixed 0–100 aggregate display score has been removed.

| `Z_order` | Descriptive interpretation |
|---:|---|
| `< 1` | weak evidence |
| `1–2` | limited order dependence |
| `2–3` | clear order dependence |
| `≥ 3` | very strong order dependence |

The threshold is a research convention, not a universal calibration.

## v0.2 outputs

### Structure strength

- random-shuffle `Z_order`;
- adjacent-swap and 3–5-sentence block controls;
- reverse directionality;
- long-history predictive gain \(C^{(13)}-C^{(3)}\);
- normalized turning-point Z.

### Structure type

- `linear`;
- `circular`;
- `stepwise`;
- `branching`;
- `repetitive`;
- `mosaic`;
- `weak_or_mixed`.

### Explanation and editing risks

- one row per sentence with \(C_t\), history Z, history-state change, curvature,
  turning Z, long-history gain, and structural role;
- transformed motif returns separated from exact copied meaning;
- theme cohesion separated from between-segment diversity;
- short-sentence rate, length variance, lexical diversity, and technical-term
  density;
- explicit risk locations for duplication, topic deviation, unexplained
  transitions, short-sentence concentration, and opening/body mismatch.

## Encoder roles

| Use | Encoder |
|---|---|
| CI and unit tests | TF-IDF |
| fast lexical smoke checks | TF-IDF |
| Japanese production analysis | multilingual-E5 |
| motif analysis | multilingual-E5 |
| lexical repetition | TF-IDF |
| comparative study | `--encoder both` |

TF-IDF and E5 are separate analysis channels. TF-IDF smoke output must not be
treated as a substitute for semantic production analysis.

## Install

```bash
python -m pip install -e .
```

For multilingual E5:

```bash
python -m pip install -e ".[semantic]"
```

## Analyze and create a report bundle

```bash
atct-fingerprint analyze article.txt \
  --encoder e5 \
  --shuffles 200 \
  --controls local,block,random,reverse \
  --sentence-map \
  --motif-analysis \
  --output reports/article
```

Generated files:

```text
reports/article/
├── fingerprint.json
├── sentence_map.csv
├── turning_points.csv
├── motif_pairs.csv
├── report.md
└── report.pdf
```

The PDF is a portable summary. Japanese sentence-level evidence remains in the
UTF-8 JSON, CSV, and Markdown outputs.

## Secondary unknown-model benchmark

The v0.1 AUROC experiment remains available as a secondary benchmark:

```bash
atct-fingerprint evaluate data/confirmatory.csv --encoder e5
```

It rejects AI-source overlap and normalized duplicate documents between train
and test. This benchmark does not replace the v0.2 primary document-level
`Z_order` proposition.

## Validation status

The implementation includes deterministic tests for current-sentence
exclusion, random-null behavior, local/block/random/reverse controls, 30-seed
sign stability, shuffled-chain falsification, paragraph-order preservation,
motif deletion, exact-copy rejection, and report generation.

The PR remains a research preview until real Japanese E5 corpora pass the full
acceptance matrix, especially literal sentence split/merge tolerance within
10%. See [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md).

## Development

```bash
python -m unittest discover -s tests -v
```
