# ATCT Narrative Fingerprint v0.7 implementation notes

## Implemented proposition

> Can the checker distinguish an exclusive deep-reframing title from a
> non-exclusive body that retains competing real-world causes?

The deterministic Phase-1 contrast uses the same body with either
`転職したいんじゃない` or `転職したいだけではない`. The first form is
exclusive; the second is non-exclusive. The title is read from the first
Markdown heading, while the prose body remains in the existing semantic layer.

The target evaluation measure is Macro-F1. No Macro-F1 value is reported in
v0.7 because a frozen, independently annotated corpus does not yet exist. The
JSON field `evaluation_status` records this pending gate explicitly.

## Modules

| Module | Deterministic responsibility |
|---|---|
| `desire_frames.py` | surface, intermediate, and deep desire candidates |
| `scope_audit.py` | exclusive/non-exclusive title and body scope |
| `cause_candidates.py` | supported, possible, rejected, and unexamined causes |
| `cause_competition.py` | primary interpretation, retained alternatives, and monopoly |
| `causal_layers.py` | institution → exploration → agency → meaning order |
| `metaphor_roles.py` | OS role changes and value overcompression |
| `transformation.py` | variable, before/after state, observation, and re-change |
| `autonomy.py` | five operational autonomy conditions |
| `recursive_cycles.py` | Shadow–Seeking–Transformation–New Shadow recurrence |
| `evidence_spans.py` | source-span and rule provenance helpers |
| `semantic_structure.py` | aggregation without replacing component evidence |
| `reporting_v07.py` | ten additional CSV evidence tables |

`features.py` calls the orchestrator and only attaches the returned analysis,
sentence-map evidence, structure labels, and editing warnings.

## Desire and exclusion

Each explicit desire is stored as a `DesireFrame` containing subject, action,
object, depth, polarity, sentence index, source span, confidence, and
`rule_or_model`. Deep-redefinition evidence is:

\[
R_{deep}=D_{depth}\,S_{evidence}(1-E_{exclusion}).
\]

`E_exclusion` is the title/body mismatch in the current deterministic
baseline. A title-only deep claim cannot receive strong evidence because
`S_evidence` is computed from body frames.

## Cause preservation

The registered baseline retains these cause families:

- value conflict;
- work environment;
- income;
- health;
- relationship;
- job fit;
- family constraint;
- social norm.

An absent family is emitted as `not_examined` with zero confidence. This keeps
the audit vocabulary visible without pretending the document supplied
evidence. The cause-monopoly ratio is computed only from supported and possible
candidates. A high ratio or fewer than three active families produces
`single_cause_overcompression`.

## Causal layers

Institutional, exploration-possibility, individual-agency, and
meaning-reconstruction events are extracted separately. The bridge combines
role coverage with the longest expected ordered prefix. Reordering the same
sentences therefore changes causal order while retaining vocabulary.

\[
C_{multi}=B_{bridge}(1-C_{monopoly}).
\]

The rule does not infer that institutions alone create meaning, or that
individual intention overrides institutional constraints.

## Transformation and autonomy

Transformation operationality averages five independently displayed fields:

1. changed variable;
2. initial state;
3. changed state;
4. observable marker;
5. reversibility or next-change condition.

Autonomy is not inferred from `自分で選んだと思える` alone. It requires
separate evidence for reason explanation, trade-off awareness, comparison with
alternatives, revisability, and distinction between external and personal
criteria.

\[
T_{complete}=T_{op}A_{autonomy}.
\]

## Recursive cycle

At least four of five cycle criteria must be present, and an explicit New
Shadow is mandatory. This additional requirement makes deletion of the final
New Shadow change the classification from `recursive_transformation_cycle` to
`stepwise_transformation`.

## Component index

The requested weighted summary is preserved:

\[
S_{v0.7}=0.25R_{deep}+0.25C_{multi}+0.25T_{complete}
+0.15M_{transition}+0.10S_{cycle}.
\]

It is named `v07_component_index`, remains on a 0–1 scale, and is always shown
beside its components. It is rule coverage, not writing quality, authorship,
truth, diagnosis, or a calibrated probability.

## Falsification coverage

Deterministic tests verify:

1. exclusive and non-exclusive titles produce different mismatch values;
2. deleting real-world alternatives keeps desire depth but reduces the
   multi-cause score by at least 30%;
3. fixing the OS role removes role transition despite repeated vocabulary;
4. generic change language reduces Transformation operationality by at least
   30%;
5. explicit reasons and trade-offs raise operational autonomy;
6. deleting New Shadow removes recursive classification;
7. reversing the causal order with identical vocabulary reduces the bridge by
   at least 30%;
8. Markdown-title integration retains evidence spans;
9. identical input returns an identical result dictionary.

## Remaining validation gates

- frozen title/body exclusion labels and Macro-F1;
- Japanese semantic desire and cause annotations;
- out-of-domain genre evaluation;
- precision audits for negation scope and nested quotations;
- frozen E5 or semantic-model revision for the high-precision channel;
- calibration of every component by genre and document length.
