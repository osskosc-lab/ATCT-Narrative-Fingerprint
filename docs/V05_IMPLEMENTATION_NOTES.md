# ATCT Narrative Fingerprint v0.5 implementation notes

## Fixed proposition

> Do subject, predicate, target, role, and motif relations show a coherent
> transformation through document time?

v0.5 keeps the v0.4 lexical intervention channel and adds an independent
relation-history channel.

| Channel | Primary evidence |
|---|---|
| Lexical history | `Z_lexical` from sentence-vector coherence versus order nulls |
| Relation history | `Z_relational` from ordered subject-predicate-target transformations versus frame-order shuffles |

The two values are never averaged into one writing-quality or authorship score.

## Relation frame

Each detected clause emits:

```text
(subject, predicate, predicate_family, target, polarity, tense, modality,
 execution_intensity, motifs)
```

The deterministic CI baseline distinguishes self, other, named object, and
unspecified targets. Execution intensity uses the preregistered scale:

| Modality | Intensity |
|---|---:|
| executed | 1.0 |
| planned | 0.6 |
| possible | 0.4 |
| deferred | 0.2 |
| avoidance | 0.0 |
| refusal | -0.5 |

## Relation evidence

Target asymmetry compares execution intensity for the same predicate family
under other and self targets. Relation flips require the same subject and
predicate family with contrasting targets. `Z_relational` compares the
observed temporal coupling between target change and intensity change against
shuffled frame positions.

The four descriptive quadrants are:

| `Z_lexical` | `Z_relational` | Label |
|---|---|---|
| high | high | lexical and relational continuity |
| high | low | lexical repetition dominant |
| low | high | deep relational structure |
| low | low | weak or mosaic |

## Functional motifs and closure

v0.5 adds:

- same-object role transition;
- different-object/same-function clusters;
- ordered motif chains;
- recognition, execution, deferment, and openness evidence;
- recognition-action distance;
- paragraph-level `emphasis_or_pause` preservation for one-sentence paragraphs.

The combined structure labels can include `relational_mirror`,
`role_transforming_cycle`, and `deferred_self_judgment`.

## Falsification coverage

Deterministic tests verify:

1. exchanging self and other reverses target-asymmetry sign;
2. deleting the mirror reduces the self-confirmation motif cluster;
3. completing the final action changes deferred closure to resolved closure;
4. relation flips survive deletion of an explanatory thesis sentence;
5. note/mirror replacement preserves the functional motif class;
6. predicate paraphrases preserve their relation family;
7. target reassignment lowers relation coherence by at least 30%;
8. identical input and seed reproduce exactly.

## Interpretation boundary

The current relation extractor is a transparent Japanese lexical and
construction-rule baseline for CI and falsification. It is not a full
dependency parser, semantic-role labeler, or coreference resolver. Ellipsis,
indirect subjects, nested quotation, passive ambiguity, and unseen predicate
families can produce missing or incorrect frames. Production claims require a
frozen Japanese semantic-role model and a manually annotated evaluation
corpus. Every reported relation therefore retains its source sentence and
clause for human inspection.

