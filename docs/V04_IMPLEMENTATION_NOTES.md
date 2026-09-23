# v0.4 implementation notes

The fixed primary proposition is unchanged:

> Is original history-conditioned coherence significantly higher than
> content-preserving order controls?

`Z_order` remains the only primary gate. The v0.4 additions are secondary,
evidence-oriented diagnostics motivated by the essay analysis of a repeated
「もっと」 motif.

| Observed failure | v0.4 implementation | Interpretation limit |
|---|---|---|
| one motif changes function over time | repeated term plus sentence-context shift and distance | context change is a role proxy, not an intent label |
| one desire splits into two causal paths | explicit contrast/alternative retrieval | implicit branches remain out of scope |
| opening question is answered near the end | ranked opening-question/ending-answer pairs | match does not prove answer correctness |
| a motif returns with updated meaning | stable reuse and distant role-shift return are separated | production use should prefer E5 |
| a thought experiment becomes categorical | qualifier-to-assertion scope audit | warning requires human review |

## Motif Role Transition

Candidate motifs come from repeated quoted phrases, Latin terms, and
conservative Japanese character sequences. For two occurrences \(i,j\), the
current sentence embedding is used as a context proxy:

\[
\Delta_{\mathrm{role}}(i,j)=1-\cos(x_i,x_j).
\]

A distant pair with a sufficiently changed context is emitted as
`return_with_role_shift`. Identical repeated sentences are always
`stable_role`, so copied text is not presented as transformed return.

## Concept Branch

The lightweight channel recognizes explicit Japanese and English contrast or
alternative markers. It records both clauses and character-ngram overlap.
The resulting divergence is lexical evidence; E5 does not turn this heuristic
into a proof of conceptual opposition.

## Question–Answer Closure

Questions in the opening 40% are compared with statements in the closing 40%.
The ranking score combines the active encoder similarity, character-ngram
overlap, and an answer-cue indicator. Only the best pair can receive
`question_answer_closure`; the remaining rows are inspection candidates.

## Claim Scope Audit

A sentence containing a premise qualifier such as `もし`, `仮定`, or
`思考実験` is linked to a later categorical assertion when they share a
lexical anchor or sufficiently similar embedding. An explicit restatement such
as `この思考実験では` suppresses the warning.

This audit identifies possible qualifier loss. It does not decide whether the
assertion is true, false, theological, scientific, or otherwise justified.
