# v0.2 validation matrix

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
| closing motif deletion | removing the final return lowers closure | passed |
| copied paragraph rejection | exact duplicates excluded from transformed motif return | passed |
| current-sentence exclusion | new causal state omits \(x_t\) and improves the synthetic gap | passed |
| evidence localization | sentence map identifies history and turning evidence | passed |
| report bundle | JSON, two CSV maps, motif CSV, Markdown, and PDF generated | passed |

Operational status remains **research preview** while literal split/merge
tolerance and real-corpus E5 stability are pending.
