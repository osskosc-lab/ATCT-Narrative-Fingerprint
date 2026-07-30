import unittest

from atct_fingerprint.causal_frames import analyze_causal_frames
from atct_fingerprint.document_layers import analyze_document_layers
from atct_fingerprint.evidence_types import analyze_evidence_types
from atct_fingerprint.features import analyze_text
from atct_fingerprint.funnel import analyze_funnel
from atct_fingerprint.metaphor_audit import analyze_metaphors
from atct_fingerprint.modality import analyze_modality
from atct_fingerprint.responsibility import analyze_responsibility
from atct_fingerprint.sequence_audit import analyze_sequence_claims


ARTICLE = [
    "頑張っても人生が変わらないことに苦しんでいませんか。",
    "昔の僕も、努力しても動けない時期を経験した。",
    "あなたは怠け者ではない。",
    "真面目に頑張りすぎただけだ。",
    "原因は才能不足や努力不足、意志の弱さではない。",
    "本当の原因は努力する順番の誤りかもしれない。",
    "まず生活や身体を整えることで、ここから変えられる。",
    "最初に身を整える。",
    "次に脳を休める。",
    "その後に気を回復する。",
    "心を見直す。",
    "最後に志を定める。",
    "人生のOSを整えれば必ず動き始める。",
    "五つ星成功術でいう「身→脳→気→心→志」の順番である。",
    "人生は順番で必ず変わる。",
    "第一歩は自分を整えることだ。",
    "無料レポートはこちらから受け取ってください。",
    "関連記事もご覧ください。",
]


def component_analyses(sentences):
    evidence = analyze_evidence_types(sentences)
    causal = analyze_causal_frames(sentences, evidence)
    layers = analyze_document_layers(sentences)
    funnel = analyze_funnel(
        sentences,
        causal,
        layers,
        shuffle_count=500,
        seed=42,
    )
    return evidence, causal, layers, funnel


class PersuasionV06Tests(unittest.TestCase):
    def test_reversing_stages_breaks_sequence_alignment(self):
        reversed_article = [
            *ARTICLE[:13],
            (
                "五つ星成功術でいう"
                "「志→心→気→脳→身」の順番である。"
            ),
            *ARTICLE[14:],
        ]
        original = analyze_sequence_claims(ARTICLE)
        reversed_result = analyze_sequence_claims(reversed_article)
        self.assertEqual(
            set(original.claims[0].stages),
            set(reversed_result.claims[0].stages),
        )
        self.assertEqual(original.mean_alignment, 1.0)
        self.assertEqual(reversed_result.mean_alignment, 0.0)
        self.assertTrue(original.insufficient_evidence_warning)

    def test_offer_removal_preserves_content_and_reduces_funnel_by_thirty_percent(self):
        full = analyze_text(
            "\n".join(ARTICLE),
            shuffle_count=80,
            controls=("random", "reverse"),
            seed=42,
            parse_markdown=False,
        )
        body = analyze_text(
            "\n".join(ARTICLE[:-2]),
            shuffle_count=80,
            controls=("random", "reverse"),
            seed=42,
            parse_markdown=False,
        )
        self.assertAlmostEqual(full.content_z, body.content_z)
        self.assertLess(
            body.persuasion_analysis.funnel.funnel_score,
            full.persuasion_analysis.funnel.funnel_score * 0.70,
        )
        self.assertEqual(
            full.persuasion_analysis.document_layers.promotion_start_index,
            16,
        )
        self.assertIsNone(
            body.persuasion_analysis.document_layers.promotion_start_index
        )

    def test_generic_name_removes_brand_transition_but_keeps_stage_structure(self):
        generic = [
            sentence.replace("五つ星成功術", "心身を整える五段階")
            for sentence in ARTICLE
        ]
        original_sequence = analyze_sequence_claims(ARTICLE)
        generic_sequence = analyze_sequence_claims(generic)
        _, _, _, original_funnel = component_analyses(ARTICLE)
        _, _, _, generic_funnel = component_analyses(generic)
        self.assertEqual(
            original_sequence.claims[0].stages,
            generic_sequence.claims[0].stages,
        )
        self.assertEqual(
            original_sequence.mean_alignment,
            generic_sequence.mean_alignment,
        )
        self.assertEqual(
            original_funnel.branding_transition.branded_framework,
            "五つ星成功術",
        )
        self.assertEqual(
            generic_funnel.branding_transition.transition_strength,
            0.0,
        )

    def test_personal_testimony_removal_leaves_framework(self):
        without_testimony = [
            sentence
            for sentence in ARTICLE
            if "昔の僕" not in sentence
        ]
        original_evidence = analyze_evidence_types(ARTICLE)
        deleted_evidence = analyze_evidence_types(without_testimony)
        original_sequence = analyze_sequence_claims(ARTICLE)
        deleted_sequence = analyze_sequence_claims(without_testimony)
        self.assertGreater(
            original_evidence.counts.get("personal_experience", 0),
            deleted_evidence.counts.get("personal_experience", 0),
        )
        self.assertEqual(
            original_sequence.claims[0].stages,
            deleted_sequence.claims[0].stages,
        )

    def test_hedging_reduces_only_modality_warning(self):
        hedged = [
            sentence.replace(
                "人生は順番で必ず変わる。",
                (
                    "人生は、順番によって変わりやすくなる"
                    "場合がある。"
                ),
            )
            for sentence in ARTICLE
        ]
        original_evidence, original_causal, original_layers, original_funnel = (
            component_analyses(ARTICLE)
        )
        hedged_evidence, hedged_causal, hedged_layers, hedged_funnel = (
            component_analyses(hedged)
        )
        original_modality = analyze_modality(
            ARTICLE,
            original_evidence,
            body_indices=original_layers.content_indices,
        )
        hedged_modality = analyze_modality(
            hedged,
            hedged_evidence,
            body_indices=hedged_layers.content_indices,
        )
        self.assertLess(
            hedged_modality.unsupported_escalation,
            original_modality.unsupported_escalation,
        )
        self.assertEqual(
            original_causal.substitutions[0].adopted_cause,
            hedged_causal.substitutions[0].adopted_cause,
        )
        self.assertEqual(original_funnel.funnel_score, hedged_funnel.funnel_score)
        self.assertEqual(
            original_funnel.branding_transition.branded_framework,
            hedged_funnel.branding_transition.branded_framework,
        )

    def test_concrete_definition_reduces_reification_and_increases_specificity(self):
        metaphor = analyze_metaphors(
            ["人生のOSを整えれば必ず動き始める。"]
        )
        concrete = analyze_metaphors(
            [
                "不安や怒りによって注意・睡眠・実行機能が"
                "乱れている状態を整える。"
            ]
        )
        self.assertGreater(metaphor.reification_score, concrete.reification_score)
        self.assertLess(metaphor.specificity_score, concrete.specificity_score)

    def test_adopted_cause_replacement_updates_graph(self):
        replaced = [
            sentence.replace(
                "本当の原因は努力する順番の誤りかもしれない。",
                "本当の原因は環境制約かもしれない。",
            )
            for sentence in ARTICLE
        ]
        _, original_causal, _, _ = component_analyses(ARTICLE)
        _, replaced_causal, _, _ = component_analyses(replaced)
        replaced_responsibility = analyze_responsibility(replaced_causal)
        self.assertEqual(
            original_causal.substitutions[0].adopted_cause,
            "努力する順番の誤り",
        )
        self.assertEqual(
            replaced_causal.substitutions[0].adopted_cause,
            "環境制約",
        )
        self.assertEqual(
            replaced_responsibility.shifts[0].adopted_category,
            "environment",
        )

    def test_scope_expansion_evidence_and_seed_are_explicit(self):
        evidence, causal, layers, funnel = component_analyses(ARTICLE)
        modality = analyze_modality(
            ARTICLE,
            evidence,
            body_indices=layers.content_indices,
        )
        responsibility = analyze_responsibility(causal)
        repeated = component_analyses(ARTICLE)[3]
        self.assertGreater(modality.unsupported_scope_expansion, 0.0)
        self.assertGreater(responsibility.relief_score, 0.5)
        self.assertGreaterEqual(funnel.persuasion_z, 2.0)
        self.assertEqual(funnel.to_dict(), repeated.to_dict())
        self.assertEqual(layers.editorial_end_index, 15)


if __name__ == "__main__":
    unittest.main()
