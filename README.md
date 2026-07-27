# ATCT Narrative Fingerprint v0.4

**文章がどのような履歴依存構造で生成されたかを測るチェッカー**

The fixed research question remains:

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

## v0.4 changes

Markdown input is separated before semantic analysis:

\[
\text{prose layer}\oplus\text{equation layer}\oplus\text{structure layer}.
\]

Headings, equations, quotes, lists, tables, and thematic breaks are preserved
as metadata instead of being counted as ordinary prose sentences. Equations
are tagged with roles such as definition, hypothesis, falsification, theorem,
or conclusion.

Order interventions now distinguish adjacent swaps, within-paragraph
shuffles, 3–5-sentence blocks, paragraph order, section order, full random
order, and reversal. A section graph records macro transitions.

Direction is no longer inferred from a symmetric embedding distance alone.
The lightweight CI baseline compares character-ngram conditional likelihood
under preceding versus following context. This is an asymmetric lexical
baseline; confirmatory semantic direction claims require a frozen
autoregressive model.

Long-history gain is measured on held-out future sentences with rolling-origin
ridge prediction:

\[
G_{\mathrm{long}} =
E_{\mathrm{short}}^{\mathrm{test}}-
E_{\mathrm{long}}^{\mathrm{test}}.
\]

Explicit boundary statements such as "this is a metaphor" or "does not prove"
are recorded as `Licensed Jump` evidence and are not automatically reported as
unexplained logical leaps.

The v0.4 explanation layer adds four evidence-retrieval diagnostics:

- **Motif Role Transition** follows a repeated phrase across sentence contexts
  and separates stable reuse from a distant return with changed context;
- **Concept Branch** retrieves explicit contrasts and alternatives such as
  `not ... but` and `それとも`;
- **Question–Answer Closure** ranks candidate ending statements against
  opening questions;
- **Claim Scope Audit** flags a qualified premise that is followed by a related
  categorical assertion without the original qualifier.

Every diagnostic returns sentence indices, original text, and matching
evidence. These are heuristic inspection aids, not proof of authorial intent,
logical validity, or literary quality. `Z_order` remains the only primary gate.

## Outputs

### Structure strength

- random-shuffle `Z_order`;
- adjacent, paragraph-internal, block, paragraph-order, section-order, random,
  and reverse controls;
- asymmetric conditional directionality;
- rolling-origin long-history predictive gain;
- normalized turning-point Z.

### Structure type

- `linear`;
- `circular`;
- `open_spiral`;
- `stepwise`;
- `branching`;
- `repetitive`;
- `mosaic`;
- `weak_or_mixed`.

### Explanation and editing risks

- one row per sentence with \(C_t\), history Z, history-state change, curvature,
  turning Z, held-out long-history gain, direction delta, section, Licensed
  Jump status, and structural role;
- Markdown layer inventory and section-transition graph;
- transformed motif returns separated from exact copied meaning;
- repeated-motif context changes, explicit conceptual branches, opening
  question/ending answer candidates, and claim-scope warnings;
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
  --controls local,paragraph_inner,block,paragraph_order,section_order,random,reverse \
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
├── motif_role_transitions.csv
├── concept_branches.csv
├── qa_closure.csv
├── claim_scope_audit.csv
├── section_graph.csv
├── document_blocks.csv
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
and test. This benchmark does not replace the v0.4 primary document-level
`Z_order` proposition.

## Validation status

The implementation includes 36 deterministic tests. They cover
current-sentence exclusion, Markdown layer separation, hierarchical order
controls, reversal of the asymmetric direction score, rolling prediction
without future-target leakage, 30-seed sign stability, shuffled-chain
falsification, motif deletion, exact-copy rejection, Licensed Jump handling,
motif role shifts, contrastive branches, question–answer closure, claim-scope
drift, and report generation.

The PR remains a research preview until real Japanese E5 corpora pass the full
acceptance matrix, especially literal sentence split/merge tolerance within
10%. See [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md).

## Development

```bash
python -m unittest discover -s tests -v
```
