# v0.4 validation matrix

The code-level tests below validate implementation behavior. They are not a
substitute for a confirmatory Japanese multilingual-E5 corpus.

| Acceptance item | Current evidence | Status |
|---|---|---|
| exact same-seed reproduction | complete result dictionaries match | passed |
| 30-seed sign agreement | coherent synthetic chain retains positive sign | passed |
| random control near zero | orthogonal random-order vectors return Z=0 | passed |
| surface-length confound | same semantic vectors, different surface lengths give identical Z | passed at fixed segmentation |
| literal split/merge tolerance ≤10% | requires re-encoding a controlled Japanese corpus | pending |
| original vs full shuffle | coherent chain loses Z after permutation | passed |
| local vs global intervention | random effect exceeds adjacent-swap effect | passed |
| paragraph preservation | paragraph permutation preserves within-block order | passed |
| paragraph-internal control | block membership is preserved while sentence order changes | passed |
| Markdown separation | headings, equation, quote, and list are excluded from prose count | passed |
| equation role | heading context assigns the equation role | passed |
| asymmetric reversal | reversal flips the conditional direction score sign | passed |
| predictive holdout | changing later targets cannot change an earlier rolling gain | passed |
| section hierarchy | heading-defined nodes and section-order control are emitted | passed |
| Licensed Jump | explicit boundary markers suppress unexplained-leap misclassification | passed |
| closing motif deletion | removing the final return lowers closure | passed |
| copied paragraph rejection | exact duplicates excluded from transformed motif return | passed |
| motif role transition | distant changed context is separated from an exact sentence return | passed |
| conceptual branch | explicit `それとも` alternatives are localized | passed |
| question–answer closure | opening question is matched to a related ending answer | passed |
| claim-scope audit | dropped qualifier is flagged; explicit re-scoping is not | passed |
| current-sentence exclusion | new causal state omits \(x_t\) and improves the synthetic gap | passed |
| evidence localization | sentence map identifies history and turning evidence | passed |
| report bundle | JSON, nine CSV diagnostics, Markdown, and PDF generated | passed |
| frozen autoregressive direction model | requires model revision and corpus freeze | pending |

Operational status remains **research preview** while literal split/merge
tolerance and real-corpus E5 stability are pending.
