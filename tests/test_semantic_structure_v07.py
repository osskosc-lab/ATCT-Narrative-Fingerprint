import unittest

from atct_fingerprint.autonomy import analyze_autonomy
from atct_fingerprint.causal_layers import analyze_causal_layers
from atct_fingerprint.cause_candidates import analyze_cause_candidates
from atct_fingerprint.features import analyze_text
from atct_fingerprint.metaphor_roles import analyze_metaphor_roles_v07
from atct_fingerprint.recursive_cycles import analyze_recursive_cycle
from atct_fingerprint.scope_audit import audit_title_body_scope
from atct_fingerprint.semantic_structure import analyze_semantic_structure
from atct_fingerprint.transformation import analyze_transformation


ARTICLE = [
    "私は転職したいと思っていた。",
    "働く環境を変えたい気持ちもある。",
    "本当に必要なのは、成功の判断基準を自分で選び直したいということだ。",
    "労働条件も原因として無視できない。",
    "賃金も影響する場合がある。",
    "健康や人間関係も重要である。",
    "会社から与えられた成功OSは社会の標準だった。",
    "成功OSを自分の中へ内面化していた。",
    "古い成功OSは現実と合わず、不整合を起こした。",
    "成功OSへの違和感がエラーを検出した。",
    "成功OSは書き換え対象になった。",
    "成功OSを自分で更新できる判断体系へ変える。",
    "制度が新しい働き方を可能にする。",
    "選択肢が増えると探索を試せる。",
    "その上で本人が自分で選択する。",
    "成功の判断基準を書き換え、意味を再構築する。",
    "会社から与えられた成功から、自分で定義する成功へ変わる。",
    "選択理由を説明でき、行動で確認できる。",
    "状況変化に応じて再検討し、会社の基準と自分の基準を区別する。",
    "古いモデルのShadowからSeekingが始まり、新しい判断基準へTransformationする。",
    "その基準は暫定で、修正できる。",
    "また新たなShadowが現れ、探索が再開する。",
]


class SemanticStructureV07Tests(unittest.TestCase):
    def test_exclusive_and_nonexclusive_titles_are_distinguished(self):
        exclusive = audit_title_body_scope("転職したいんじゃない", ARTICLE)
        nonexclusive = audit_title_body_scope("転職したいだけではない", ARTICLE)
        self.assertGreaterEqual(exclusive.mismatch, 0.70)
        self.assertLess(nonexclusive.mismatch, 0.20)
        self.assertEqual(exclusive.warning, "title_body_scope_mismatch")
        self.assertEqual(
            exclusive.suggested_revision,
            "転職したいだけではない",
        )
        self.assertEqual(
            exclusive.evaluation_status,
            "pending_annotated_corpus_macro_f1",
        )
        retained = analyze_cause_candidates(
            ["労働条件だけが原因ではない。"]
        )
        work_environment = next(
            item
            for item in retained.candidates
            if item.cause_type == "work_environment"
        )
        self.assertEqual(work_environment.status, "possible")

    def test_removing_competing_causes_preserves_depth_but_breaks_multi_cause(self):
        full = analyze_semantic_structure(
            ARTICLE,
            title="転職したいだけではない",
        )
        removed = [
            sentence
            for index, sentence in enumerate(ARTICLE)
            if index not in {3, 4, 5, 12}
        ]
        reduced = analyze_semantic_structure(
            removed,
            title="転職したいだけではない",
        )
        self.assertAlmostEqual(
            full.desires.depth_score,
            reduced.desires.depth_score,
        )
        self.assertLess(
            reduced.multi_cause_score,
            full.multi_cause_score * 0.70,
        )
        self.assertGreater(
            reduced.cause_competition.cause_monopoly,
            full.cause_competition.cause_monopoly,
        )
        self.assertEqual(
            reduced.cause_competition.warning,
            "single_cause_overcompression",
        )

    def test_fixing_os_role_lowers_transition_despite_repetition(self):
        original = analyze_metaphor_roles_v07(ARTICLE)
        static = analyze_metaphor_roles_v07(["価値観を振り返る。"] * 6)
        self.assertGreaterEqual(len(original.unique_roles), 4)
        self.assertGreater(original.transition_score, static.transition_score)
        self.assertEqual("\n".join(["価値観を振り返る。"] * 6).count("価値観"), 6)

    def test_vague_transformation_lowers_operationality_by_thirty_percent(self):
        specific = analyze_transformation(ARTICLE)
        vague = analyze_transformation(
            ["世界が変わる。", "人生が動き出す。", "未来へ進む。"]
        )
        self.assertGreaterEqual(specific.operationality, 0.80)
        self.assertLess(vague.operationality, specific.operationality * 0.70)

    def test_autonomy_sentence_increases_operational_autonomy(self):
        baseline = analyze_autonomy(["自分で選んだと思えることが大切だ。"])
        enriched = analyze_autonomy(
            [
                "自分で選んだと思えることが大切だ。",
                (
                    "自分の選択理由と、その選択で失うものの両方を"
                    "説明できるようになったとき、判断基準は初めて"
                    "自分のものになる。"
                ),
            ]
        )
        self.assertTrue(baseline.subjective_ownership_detected)
        self.assertGreater(enriched.autonomy_score, baseline.autonomy_score)
        self.assertEqual(
            baseline.warning,
            "subjective_ownership_without_operational_autonomy",
        )

    def test_deleting_new_shadow_changes_cycle_to_stepwise(self):
        full = analyze_recursive_cycle(ARTICLE)
        linear = analyze_recursive_cycle(ARTICLE[:-1])
        self.assertTrue(full.detected)
        self.assertFalse(linear.detected)
        self.assertEqual(full.structure_type, "recursive_transformation_cycle")
        self.assertEqual(linear.structure_type, "stepwise_transformation")

    def test_vocabulary_preserving_causal_reversal_lowers_bridge(self):
        forward = [
            "制度が選択を可能にする。",
            "探索によって選択肢を試せる。",
            "本人が自分で選択する。",
            "意味を再構築する。",
        ]
        reversed_cause = [
            "意味を再構築する。",
            "本人が自分で選択する。",
            "探索によって選択肢を試せる。",
            "制度が選択を可能にする。",
        ]
        original = analyze_causal_layers(forward)
        broken = analyze_causal_layers(reversed_cause)
        self.assertEqual(
            sorted("".join(forward)),
            sorted("".join(reversed_cause)),
        )
        self.assertLess(broken.bridge_score, original.bridge_score * 0.70)

    def test_integration_retains_evidence_and_markdown_title(self):
        text = "# 転職したいんじゃない。成功OSを書き換えたい\n\n" + "\n".join(ARTICLE)
        result = analyze_text(
            text,
            shuffle_count=12,
            controls=("random", "reverse"),
            seed=42,
        )
        analysis = result.semantic_structure_analysis
        self.assertEqual(result.version, "0.7.0")
        self.assertGreaterEqual(len(analysis.cause_competition.retained_cause_types), 3)
        self.assertGreaterEqual(len(analysis.metaphor_roles.unique_roles), 4)
        self.assertGreater(analysis.title_body_scope.mismatch, 0.70)
        self.assertTrue(analysis.title_body_scope.evidence_spans)
        self.assertIn("desire_frames", result.sentence_map[0])

    def test_same_input_is_exactly_reproducible(self):
        first = analyze_semantic_structure(ARTICLE, title="転職したいだけではない")
        second = analyze_semantic_structure(ARTICLE, title="転職したいだけではない")
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
