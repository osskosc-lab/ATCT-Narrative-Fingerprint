import unittest

import numpy as np

from atct_fingerprint.features import analyze_text
from atct_fingerprint.rhetoric import (
    analyze_motif_roles,
    analyze_qa_closure,
    audit_claim_scope,
    detect_concept_branches,
)


class RhetoricTests(unittest.TestCase):
    def test_distant_motif_with_changed_context_is_transformed_return(self):
        sentences = [
            "朝、「もっと」という声が私を急かした。",
            "私はその命令から離れた。",
            "地図の端で休んだ。",
            "好奇心なら歩けると思った。",
            "夕方、「もっと」という声を呼びかけとして聞いた。",
        ]
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.2],
                [0.0, 0.0, 1.0],
            ]
        )
        result = analyze_motif_roles(sentences, vectors)
        transitions = [
            item
            for item in result.transitions
            if item.motif == "もっと"
            and item.kind == "return_with_role_shift"
        ]
        self.assertEqual(len(transitions), 1)
        self.assertGreater(transitions[0].role_shift, 0.9)

    def test_exact_sentence_return_is_not_a_role_shift(self):
        sentence = "朝、「もっと」という声がする。"
        sentences = [sentence, "歩く。", "休む。", "考える。", sentence]
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
                [1.0, 0.0],
            ]
        )
        result = analyze_motif_roles(sentences, vectors)
        matching = [item for item in result.transitions if item.motif == "もっと"]
        self.assertTrue(matching)
        self.assertTrue(all(item.kind == "stable_role" for item in matching))

    def test_contrastive_question_branch_is_retrieved(self):
        sentences = [
            "この「もっと」は、新しい場所へ連れていくのか？",
            "それとも、今の自分を否定する命令なのか？",
        ]
        branches = detect_concept_branches(sentences)
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0].marker, "それとも")
        self.assertEqual(branches[0].paired_sentence_index, 1)
        self.assertGreater(branches[0].lexical_divergence, 0.0)

    def test_opening_question_matches_closing_answer(self):
        sentences = [
            "自分には何かが足りないからなのだろうか。",
            "私は声を聞いた。",
            "地図を広げた。",
            "少し休んだ。",
            "その声は、あなたが足りないという証明ではない。",
            "物語はまだ終わっていない。",
        ]
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.2],
                [0.0, 0.2, 0.8],
                [0.95, 0.05, 0.0],
                [0.7, 0.0, 0.3],
            ]
        )
        result = analyze_qa_closure(sentences, vectors)
        self.assertTrue(result.closed)
        self.assertEqual(result.pairs[0].question_index, 0)
        self.assertEqual(result.pairs[0].answer_index, 4)
        self.assertEqual(result.pairs[0].kind, "question_answer_closure")

    def test_unscoped_assertion_after_qualified_premise_is_flagged(self):
        sentences = [
            "もし神を完全な存在と定義するなら、欠けるものはない。",
            "ここでは人間との違いを考える。",
            "神は求めない。",
            "人間は変わる。",
        ]
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.4, 0.6],
                [0.9, 0.1],
                [0.0, 1.0],
            ]
        )
        result = audit_claim_scope(sentences, vectors)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].assertion_index, 2)
        self.assertIn("神", result.findings[0].shared_anchors)

    def test_explicitly_scoped_assertion_is_not_flagged(self):
        sentences = [
            "もし神を完全な存在と定義するなら、欠けるものはない。",
            "この思考実験では、神は求めない。",
            "人間は変わる。",
            "問いは残る。",
        ]
        vectors = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.2, 0.8]]
        )
        result = audit_claim_scope(sentences, vectors)
        self.assertEqual(result.findings, ())

    def test_fingerprint_integrates_all_rhetorical_evidence_channels(self):
        text = """朝、「もっと」という声は、私が足りないからなのだろうか。
もし神を完全な存在と定義するなら、欠けるものはない。
私は地図を開いた。
もっと知りたいのか、それとも、もっとやらなければならないのか。
地図の端で少し休んだ。
神は求めない。
その声は、あなたが足りないという証明ではない。
朝、また「もっと」という声を呼びかけとして聞く。
"""
        result = analyze_text(
            text,
            controls=("random",),
            shuffle_count=8,
            seed=9,
        )
        self.assertGreater(result.motif_role_analysis.transformed_return_count, 0)
        self.assertTrue(result.concept_branches)
        self.assertTrue(result.qa_closure.closed)
        self.assertTrue(result.claim_scope_audit.findings)
        self.assertTrue(any(row["qa_role"] for row in result.sentence_map))
        self.assertTrue(any(row["scope_warning"] for row in result.sentence_map))


if __name__ == "__main__":
    unittest.main()
