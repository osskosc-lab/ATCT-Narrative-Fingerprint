import unittest

from atct_fingerprint.closure_states import analyze_closure
from atct_fingerprint.motif_functions import analyze_motif_functions
from atct_fingerprint.relations import analyze_relations


OTHER_EXECUTED = [
    "私は他人を咎めた。",
    "私は他人を裁いた。",
    "私は他人を評価した。",
    "私は他人の判断を検討した。",
    "私は他人の行為を批判した。",
    "私は他人の記録を点検した。",
]
SELF_AVOIDED = [
    "私は自分を咎めることを避けた。",
    "私は自分を裁くことを避けた。",
    "私は自分の評価をしなくなった。",
    "私は自分の判断を検討しなくなった。",
    "私は自分への批判を避けた。",
    "私は自分の記録を点検しなくなった。",
]


class RelationV05Tests(unittest.TestCase):
    def test_target_label_exchange_reverses_asymmetry_sign(self):
        sentences = OTHER_EXECUTED + SELF_AVOIDED
        swapped = [
            sentence.replace("他人", "__OTHER__")
            .replace("自分", "他人")
            .replace("__OTHER__", "自分")
            for sentence in sentences
        ]
        original = analyze_relations(sentences, shuffle_count=200, seed=42)
        exchanged = analyze_relations(swapped, shuffle_count=200, seed=42)
        self.assertAlmostEqual(
            original.target_asymmetries[0].asymmetry,
            -exchanged.target_asymmetries[0].asymmetry,
        )

    def test_ordered_relation_history_is_deep_relational_structure(self):
        result = analyze_relations(
            OTHER_EXECUTED + SELF_AVOIDED,
            lexical_z=0.0,
            shuffle_count=500,
            seed=42,
        )
        self.assertGreaterEqual(result.relational_order_z, 2.0)
        self.assertEqual(result.quadrant, "deep_relational_structure")

    def test_mirror_deletion_reduces_same_function_motif_cluster(self):
        sentences = [
            "ノートに誰かを咎めた記録がある。",
            "そのノートは開かれていない。",
            "鏡の中の自分を見ることを避けた。",
            "いつか自分の名前のページを開くだろう。",
        ]
        complete = analyze_motif_functions(sentences)
        deleted = analyze_motif_functions(
            [sentence for sentence in sentences if "鏡" not in sentence]
        )
        complete_cluster = next(
            item
            for item in complete.functional_clusters
            if item.function == "self_confirmation_device"
        )
        deleted_cluster = next(
            item
            for item in deleted.functional_clusters
            if item.function == "self_confirmation_device"
        )
        self.assertGreater(
            len(complete_cluster.motifs), len(deleted_cluster.motifs)
        )

    def test_completed_ending_changes_deferred_to_resolved_closure(self):
        body = [
            "私は自分を裁けない問題に気づいた。",
            "ノートは机に残っている。",
            "鏡から目を逸らしていた。",
        ]
        deferred = analyze_closure(
            body + ["いつか、自分の名前のページを開くだろう。"]
        )
        resolved = analyze_closure(
            body + ["その夜、ノートを開き、自分の名前を書いた。"]
        )
        self.assertEqual(deferred.primary_state, "deferred_closure")
        self.assertEqual(resolved.primary_state, "resolved_closure")
        self.assertGreater(deferred.action_distance, resolved.action_distance)

    def test_relation_flip_survives_explanatory_thesis_deletion(self):
        sentences = (
            OTHER_EXECUTED[:3]
            + ["自分を裁けなくなった正義が、悪意になる。"]
            + SELF_AVOIDED[:3]
        )
        deleted = [
            sentence for sentence in sentences if "正義" not in sentence
        ]
        result = analyze_relations(deleted, shuffle_count=200, seed=7)
        self.assertTrue(result.relation_flips)
        self.assertGreater(result.target_asymmetries[0].asymmetry, 0.5)

    def test_surface_object_replacement_keeps_function_cluster(self):
        original = analyze_motif_functions(
            [
                "ノートに判断の記録がある。",
                "鏡の中の自分から目を逸らした。",
            ]
        )
        replaced = analyze_motif_functions(
            [
                "録音データに判断の記録がある。",
                "過去の動画の自分から目を逸らした。",
            ]
        )
        self.assertEqual(
            original.functional_clusters[0].function,
            replaced.functional_clusters[0].function,
        )

    def test_predicate_paraphrase_keeps_relation_family(self):
        first = analyze_relations(
            [
                "私は他人を咎めた。",
                "私は自分を裁くことを避けた。",
            ],
            shuffle_count=40,
            seed=5,
        )
        paraphrased = analyze_relations(
            [
                "私は他人を評価した。",
                "私は自分を点検することを避けた。",
            ],
            shuffle_count=40,
            seed=5,
        )
        self.assertEqual(
            {frame.predicate_family for frame in first.frames},
            {frame.predicate_family for frame in paraphrased.frames},
        )
        self.assertAlmostEqual(
            first.target_asymmetries[0].asymmetry,
            paraphrased.target_asymmetries[0].asymmetry,
        )

    def test_relation_destruction_lowers_coherence_by_thirty_percent(self):
        intact = analyze_relations(
            OTHER_EXECUTED + SELF_AVOIDED,
            shuffle_count=200,
            seed=3,
        )
        destroyed = analyze_relations(
            [
                *OTHER_EXECUTED[:3],
                *SELF_AVOIDED[:3],
                *[
                    sentence.replace("自分", "他人")
                    for sentence in SELF_AVOIDED[3:]
                ],
                *[
                    sentence.replace("他人", "自分")
                    for sentence in OTHER_EXECUTED[3:]
                ],
            ],
            shuffle_count=200,
            seed=3,
        )
        self.assertLess(
            destroyed.relational_coherence,
            intact.relational_coherence * 0.70,
        )

    def test_same_seed_is_exactly_reproducible(self):
        first = analyze_relations(
            OTHER_EXECUTED + SELF_AVOIDED,
            shuffle_count=120,
            seed=11,
        )
        second = analyze_relations(
            OTHER_EXECUTED + SELF_AVOIDED,
            shuffle_count=120,
            seed=11,
        )
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
