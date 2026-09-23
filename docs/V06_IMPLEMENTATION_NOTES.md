# ATCT Narrative Fingerprint v0.6 implementation notes

## Fixed proposition

> In what order does a document replace the reader's problem, cause, and
> solution recognition, and where does the replacement connect to an offer?

v0.6 retains all v0.5 lexical and relation-history evidence. It adds a
separate persuasion-history channel and does not treat persuasion structure as
proof of manipulation, falsity, authorship, or writing quality.

## Two displayed channels

| Channel | Null or intervention |
|---|---|
| `Z_content` | strongest preregistered content-order Z after promotional-tail exclusion |
| `Z_persuasion` | expected persuasion-milestone order against shuffled milestone positions |

The content components are still emitted separately:

```text
lexical_order_z
relational_order_z
```

The current `Z_content` is the upper envelope of those preregistered channels.
It is a descriptive gate, not a multiple-comparison-adjusted population
calibration. Confirmatory work must preregister the selected content channel or
calibrate the envelope on a frozen corpus.

## Reader-transformation frame

For sentence \(t\), v0.6 emits:

\[
D_t=(A_t,C_t,E_t,S_t,O_t,M_t),
\]

where:

- \(A_t\): reader state;
- \(C_t\): rejected or adopted cause;
- \(E_t\): pain, relief, or hope role;
- \(S_t\): generic or branded solution;
- \(O_t\): offer role;
- \(M_t\): modality and evidence type.

Every frame retains the original sentence and index.

## Causal substitution and responsibility

Cause replacement records:

```text
(outcome, rejected_causes, adopted_cause, evidence_types, source_indices)
```

Responsibility categories are `personality`, `state`, `method`,
`environment`, and `other`. Movement away from personality receives a relief
score only when the adopted cause is state, method, or environment. The report
explicitly states that relief can be appropriate reappraisal, persuasion
framing, or both.

## Evidence hierarchy

The deterministic CI baseline assigns one primary evidence form per sentence:

```text
personal_experience
analogy
mechanistic_explanation
observational_claim
comparative_evidence
experimental_evidence
authority_claim
unsupported_assertion
```

This is a form classifier. It does not verify a cited study or make a truth
judgment.

## Stage-order audit

Explicit arrow or multi-stage lists are compared with earlier first mentions.
The audit separately checks for:

1. measurable stages;
2. completion criteria;
3. order comparisons;
4. exclusion of concurrent progress;
5. reverse or reciprocal effects;
6. exceptions.

`necessity_score` is the fraction of these six checks supported by explicit
cues. `alignment_score` is pairwise concordance between the asserted order and
earlier stage-mention order.

## Metaphor and modality audit

Metaphor records distinguish:

- `explanatory_metaphor`;
- `mapping_claim`;
- `causal_guarantee`;
- uncommitted `metaphor_mention`.

`reification_score` increases when a metaphor is used as a mapping or causal
guarantee without a metaphor marker. `specificity_score` is reported
separately from concrete state or measurement vocabulary.

Modality uses a visible cue ladder from possibility to certainty. Opening and
closing certainty are compared only within editorial content. Evidence gain is
subtracted before an unsupported-escalation warning is emitted. Subject scope
tracks personal, reader, group, and universal claims the same way.

## Funnel null

The expected milestones are:

```text
pain
relief
causal_substitution
solution
branded_solution
offer
```

The observed statistic is the fraction of milestone pairs in expected order.
`Z_persuasion` compares it with permutations of the observed milestone
positions using the caller-visible seed. Missing milestones reduce coverage.
The descriptive `funnel_score` also requires an offer-completion factor, so
removing the offer lowers it even when the remaining editorial sequence stays
ordered.

## Layer boundary

The first `lead_magnet`, cross-promotion, follow request, backlink, or hashtag
marks the promotional tail. Content vectors are re-encoded from only the
preceding editorial sentences. This prevents offer deletion from changing
`Z_content` merely through document-local TF-IDF weights.

## Falsification coverage

Deterministic tests verify:

1. reversing five stages lowers sequence alignment while preserving vocabulary;
2. deleting the offer preserves `Z_content` and lowers funnel score by at least
   30%;
3. replacing a proprietary name with a generic name removes only the branding
   transition;
4. deleting personal testimony lowers testimony evidence while preserving the
   stage framework;
5. retaining hedges lowers modality escalation without changing cause or
   funnel structure;
6. replacing an OS metaphor with a concrete functional definition lowers
   reification and raises specificity;
7. changing the adopted cause updates the causal graph and responsibility
   category;
8. the persuasion null is exactly reproducible with the same seed.

## Interpretation boundary

All extractors in this module are deterministic Japanese cue and construction
baselines for CI. They do not perform general event extraction, causal
inference, sentiment modeling, named-entity resolution, citation verification,
or deception detection. Production use requires a manually annotated Japanese
persuasion corpus, frozen model revisions, calibration by genre, and
out-of-domain evaluation. The report keeps all source locations so false
positives can be audited.
