# ATCT Narrative Fingerprint v0.3 research protocol

## Fixed proposition

> Is the original history-conditioned coherence significantly higher than
> content-preserving order controls?

This protocol follows one proposition, one primary metric, one baseline, and
one primary falsification.

| Item | Fixed choice |
|---|---|
| Primary metric | `Z_order` |
| Primary history window | 5 preceding sentences |
| Baseline | v0.1 history state that includes the current sentence |
| Primary null | complete random sentence order |
| Support | `Z_order ≥ 2` |
| Unsupported | `Z_order < 2` |
| Mechanism falsification | a fully shuffled input retains comparable Z |
| Implementation falsification | sentence count or short-sentence rate dominates Z |

## History separation

For every prose sentence \(x_t\), v0.3 constructs:

\[
h_t^{(w)} =
\frac{\sum_{j=1}^{w}\exp[-\lambda(j-1)]x_{t-j}}
{\sum_{j=1}^{w}\exp[-\lambda(j-1)]}.
\]

It then measures:

\[
C_t^{(w)}=\cos(x_t,h_t^{(w)}).
\]

The sentence being evaluated is excluded. The first sentence has no prior
history, so its sentence-level consistency and Z are reported as unavailable.

## Null models

| Control | Operation | Intended scale |
|---|---|---|
| local | swap one adjacent pair | local causal connection |
| paragraph_inner | shuffle sentences inside each paragraph | paragraph-local order |
| block | reorder 3–5 sentence blocks | scene or paragraph scale |
| paragraph_order | reorder paragraphs, preserve order inside each | paragraph sequence |
| section_order | reorder heading-defined sections | argument or scene sequence |
| random | unrestricted non-identity permutation | global order |
| reverse | reverse the complete prose sequence | intervention control |

All stochastic controls are generated from a fixed caller-visible seed.
Sentence-level null values are mapped back to original sentence identities
before Z values are calculated.

## Secondary measures

Secondary measures cannot overturn the primary gate:

- normalized history curvature and turning-point Z;
- rolling-origin long-history gain
  \(E_{\mathrm{short}}^{\mathrm{test}}-E_{\mathrm{long}}^{\mathrm{test}}\);
- forward/backward conditional character-ngram likelihood difference;
- opening/ending transformed closure;
- distance-weighted transformed motif recurrence;
- exact duplicate rate;
- theme cohesion;
- between-segment diversity.

Exact near-duplicates are excluded from transformed motif return. This keeps a
copied paragraph from being labeled as foreshadowing recovery.

The conditional character-ngram score is a lightweight asymmetric baseline,
not a production semantic language model. It cannot establish argumentative
directionality by itself.

## Markdown and hierarchy

Only the prose layer is embedded. Equations, headings, quotes, list items,
tables, code fences, and thematic breaks are retained as document metadata.
Equations receive contextual role labels. Heading-defined sections form a
sequential graph whose edges report transition distance, target-history
support, convergence/divergence, and Licensed Jump status.

An explicit boundary marker licenses a transition for editing diagnostics; it
does not make the cross-domain claim scientifically valid.

## Encoder separation

TF-IDF is a lexical channel used for CI and fast diagnostics.
Multilingual-E5 is the semantic channel required for Japanese production
claims. `--encoder both` produces separate lexical and semantic fingerprints;
the channels are not averaged into one score.

## Reporting

The report preserves:

- aggregate control distributions and `Z_order`;
- every sentence's evidence and structural role;
- ranked turning points;
- transformed motif and duplicate pair tables;
- document-block inventory and section-transition graph;
- asymmetric sentence direction deltas and Licensed Jump records;
- surface-confound warnings;
- editing-risk locations;
- the seed and null sample count.

No 0–100 writing-quality score is emitted.

## Confirmatory run requirements

- Freeze source texts, exclusions, E5 model revision, seed list, decay,
  history windows, motif thresholds, and null sample count before running.
- Use at least 200 shuffles per document and 30 fixed seeds for stability.
- Report the full Z distribution and sign agreement, not only a favorable seed.
- Match genre, language, editing condition, prompt family, and document length.
- Keep raw reports for negative and mechanism-falsified cases.

## Stop conditions

- **Support:** `Z_order ≥ 2` and the fully shuffled input loses the signal.
- **Unsupported:** `Z_order < 2`.
- **Mechanism falsified:** fully shuffled input retains comparable evidence.
- **Implementation falsified:** split/merge, short-sentence rate, or document
  length dominates the result.
