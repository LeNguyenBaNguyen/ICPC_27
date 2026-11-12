import unittest
from adjudicate_validation import cohens_kappa, compute_inter_annotator_agreement


class AdjudicationTests(unittest.TestCase):
    def test_cohens_kappa_perfect(self):
        a = ["supported", "contradicted", "unresolved"] * 10
        self.assertAlmostEqual(cohens_kappa(a, a), 1.0)

    def test_cohens_kappa_known_values(self):
        a = ["supported"] * 5 + ["contradicted"] * 5
        b = ["supported"] * 4 + ["contradicted"] * 1 + ["supported"] * 1 + ["contradicted"] * 4
        self.assertAlmostEqual(cohens_kappa(a, b, ["supported", "contradicted"]), 0.6)

    def test_compute_inter_annotator_agreement(self):
        rows1 = [
            {
                "task_id": f"V{i:04d}",
                "status": "supported",
                "evidence": "ev",
                "decomposition_ok": "yes",
                "claim_type": "behavioural",
                "needs_context": "no",
            }
            for i in range(1, 401)
        ]
        rows2 = [
            {
                "task_id": f"V{i:04d}",
                "status": "supported" if i <= 360 else "contradicted",
                "evidence": "ev",
                "decomposition_ok": "yes",
                "claim_type": "behavioural",
                "needs_context": "no",
            }
            for i in range(1, 401)
        ]
        metrics = compute_inter_annotator_agreement(rows1, rows2)
        self.assertEqual(metrics["sample_size"], 400)
        self.assertEqual(metrics["status_raw_agree_count"], 360)
        self.assertEqual(metrics["status_raw_agreement"], 0.9)


if __name__ == "__main__":
    unittest.main()

