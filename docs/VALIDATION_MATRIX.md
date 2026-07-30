# v0.6 validation matrix

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
| target-label reversal | exchanging self/other reverses asymmetry sign | passed |
| relational order evidence | ordered other-to-self transformation reaches relational Z ≥2 | passed on synthetic baseline |
| relation destruction | target reassignment lowers relational coherence by at least 30% | passed |
| predicate paraphrase | mapped surface verbs retain their predicate family | passed |
| functional motif replacement | note/mirror and recording/video share a functional cluster | passed on registered lexicon |
| deferred vs resolved closure | completed final action changes the closure state | passed |
| explanatory-thesis deletion | relation flip remains after deleting the summary sentence | passed |
| numbered heading | numeric prefix is retained with the heading text | passed |
| one-sentence paragraph | paragraph unit is retained as emphasis/pause evidence | passed |
| causal substitution | rejected personality causes and adopted method cause are localized | passed on registered constructions |
| responsibility relief | personality-to-method or state movement is separated from moral judgment | passed |
| stage reversal | reversing five stages changes alignment from 1 to 0 | passed on synthetic baseline |
| sequence necessity | missing measurement, completion, comparison, reciprocity, concurrency, and exception evidence is warned | passed |
| offer deletion | `Z_content` is unchanged and funnel score falls by more than 30% | passed |
| generic-name replacement | branding transition disappears while stage order remains | passed |
| testimony deletion | personal-evidence count falls while framework remains | passed |
| modality hedge | unsupported certainty escalation falls without changing cause or funnel records | passed |
| metaphor concretization | reification falls and specificity rises | passed |
| adopted-cause replacement | causal graph and responsibility category update | passed |
| editorial/promotion split | first offer sentence is excluded from content re-encoding | passed |
| persuasion seed | same seed reproduces the exact milestone null | passed |
| current-sentence exclusion | new causal state omits \(x_t\) and improves the synthetic gap | passed |
| evidence localization | sentence map identifies history and turning evidence | passed |
| report bundle | JSON, twenty-four CSV diagnostics, Markdown, and PDF generated | passed |
| Japanese semantic-role model | requires frozen model and annotated corpus | pending |
| frozen autoregressive direction model | requires model revision and corpus freeze | pending |
| Japanese persuasion corpus | requires frozen annotations for cause, evidence, metaphor, modality, and offer boundaries | pending |

Operational status remains **research preview** while literal split/merge
tolerance, real-corpus E5 stability, and semantic-role validation are pending.
