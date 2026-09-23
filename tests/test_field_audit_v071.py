import unittest

from atct_fingerprint.cause_candidates import analyze_cause_candidates
from atct_fingerprint.cause_competition import analyze_cause_competition
from atct_fingerprint.desire_frames import analyze_desires
from atct_fingerprint.scope_audit import audit_title_body_scope


class FieldAuditV071Tests(unittest.TestCase):
    def test_list_mention_does_not_overcompress_relationship(self):
        analysis = analyze_cause_candidates(
            [
                "思い出も、経験も、好き嫌いも、人間関係も、失敗も、成功も、現在の自分に流れ込んできている。"
            ]
        )
        competition = analyze_cause_competition(analysis)
        self.assertNotEqual(competition.warning, "single_cause_overcompression")

    def test_enumerated_reasons_are_not_unknown_primary(self):
        analysis = analyze_cause_candidates(
            ["理由は、だいたい三つに分けられる。", "一つ目は、観測そのものだ。"]
        )
        competition = analyze_cause_competition(analysis)
        self.assertNotEqual(competition.primary_cause_type, "unknown")
        self.assertNotEqual(competition.warning, "single_cause_overcompression")

    def test_somatic_drop_is_health_candidate(self):
        analysis = analyze_cause_candidates(
            ["給料は上がるが、身体が先に落ちていた。"]
        )
        health = next(
            item for item in analysis.candidates if item.cause_type == "health"
        )
        self.assertNotEqual(health.status, "not_examined")

    def test_hedged_title_is_nonexclusive(self):
        audit = audit_title_body_scope(
            "まぐわいはワンネスへの回帰なのかもしれない",
            ["境界が緩む。", "同じだとは言わない。"],
        )
        self.assertLess(audit.title_exclusion, 1.0)
        self.assertLess(audit.mismatch, 0.70)

    def test_security_desire_is_extracted(self):
        analysis = analyze_desires(
            ["ずばり、安心したい、が究極なのだろう。", "つながりたい気持ちがある。"]
        )
        actions = {frame.action for frame in analysis.frames}
        self.assertTrue(actions & {"安心し", "つながり"})


if __name__ == "__main__":
    unittest.main()
