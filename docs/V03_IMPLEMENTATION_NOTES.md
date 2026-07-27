# v0.3 implementation notes

The v0.3 change is deliberately additive around the fixed primary proposition:

> Is original history-conditioned coherence significantly higher than
> content-preserving order controls?

`Z_order` remains the only primary gate. Hierarchy, directionality, predictive
gain, motif return, and editing diagnostics remain secondary explanations.

| Requested improvement | v0.3 implementation |
|---|---|
| Markdown three-layer separation | `document.py` extracts prose, equations, and structure metadata |
| current sentence exclusion | retained in `causal_history_states` |
| shuffle-null Z | retained as `Z_order` |
| multi-level interventions | paragraph-internal, paragraph-order, and section-order controls added |
| asymmetric direction | conditional character-ngram forward/backward log-likelihood baseline |
| external long-history gain | rolling-origin ridge prediction on held-out future sentences |
| motif versus repetition | transformed return and exact duplicate channels remain separate |
| short-sentence correction | curvature is compared with identity-aligned shuffles; punctuation density is audited |
| section hierarchy | heading-defined section graph with transition evidence |
| Licensed Jump | explicit boundary markers classify, but do not validate, cross-domain transitions |
| three-part output | strength, structural type, and editing risks remain separate |

## Interpretation limit

The conditional character-ngram direction score is suitable for deterministic
CI and falsification scaffolding. It is not equivalent to
\(\log P(s_t\mid s_{<t})\) from a frozen large autoregressive language model.
Production direction claims therefore remain pending.

Rolling-origin prediction is out-of-sample within a document. It does not by
itself establish generalization to unseen genres or authors.
