"""Scientific endpoint and cohort reporting checks; no model/network calls."""
import tempfile
import unittest
from pathlib import Path
from run_redraw_inference import effective_indices, observation_rows, survival_at, summarize


class FakeClient:
    calls = 0
    hits = 0

    def __init__(self, directory):
        self.directory = Path(directory)

    def quality_counts(self):
        return {"cached_responses": 0, "cached_length_limited_responses": 0,
                "failed_request_records": 0, "rejected_length_limited_responses": 0}


def fixture():
    f = {"touches": [{"date": "2025-01-02"}, {"date": "2025-01-03"}],
         "censor": {"reason": "head"}}
    task = {"fid": "NEW-R000", "function": f, "effective_followup_touches": [1, 2],
            "anchor_date": "2025-01-01", "head_date": "2026-01-01"}
    return task


class RedrawTests(unittest.TestCase):
    def test_format_placebo_and_overload_stop_are_not_effective(self):
        f = {"touches": [{}, {"format_only": True}, {"paren_only": True}, {},
                         {"overload_stub": True}, {}]}
        self.assertEqual(effective_indices(f), [1, 4])

    def test_first_contradiction_absorbs_later_support(self):
        task = fixture()
        rec = {"complete": True, "anchor": [{"idx": 0, "type": "structural", "status": "supported"}],
               "touches": {"1": [{"idx": 0, "status": "contradicted"}],
                           "2": [{"idx": 0, "status": "supported"}]}}
        rows = observation_rows(task, rec)
        self.assertEqual(rows[0]["time_k"], 1)
        self.assertEqual(rows[0]["event"], 1)
        self.assertEqual(rows[0]["days"], 1)

    def test_unresolved_is_not_event_and_initial_errors_are_excluded(self):
        task = fixture()
        rec = {"complete": True, "anchor": [{"idx": 0, "type": "structural", "status": "supported"},
                                               {"idx": 1, "type": "intent", "status": "contradicted"}],
               "touches": {"1": [{"idx": 0, "status": "unresolved"}],
                           "2": [{"idx": 0, "status": "supported"}]}}
        rows = observation_rows(task, rec)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], 0)
        self.assertEqual(rows[0]["time_k"], 2)

    def test_removed_function_is_separate_competing_cause(self):
        task = fixture()
        task["function"]["censor"] = {"reason": "file_removed", "date": "2025-02-01"}
        rec = {"complete": True, "anchor": [{"idx": 0, "type": "intent", "status": "supported"}],
               "touches": {"1": [], "2": []}}
        row = observation_rows(task, rec)[0]
        self.assertEqual(row["event"], 0)
        self.assertEqual(row["removed"], 1)
        self.assertEqual(row["removal_or_contradiction"], 1)
        self.assertEqual(row["days"], 31)

    def test_censoring_at_zero_does_not_enter_touch_risk_set(self):
        rows = [{"time_k": 0, "event": 0}, {"time_k": 1, "event": 1}]
        self.assertEqual(survival_at(rows, 6), 0)

    def test_partial_prefix_does_not_publish_population_survival(self):
        task = fixture()
        plan = {"tasks": [task, {**task, "fid": "NEW-R001"}], "sample_n": 2,
                "reliability_rule": {"target_k": 6, "minimum_claims_at_risk": 30, "minimum_functions_at_risk": 5},
                "plan_sha256": "test", "max_fresh_calls": 8}
        records = {"NEW-R000": {"complete": True,
                    "anchor": [{"idx": 0, "type": "intent", "status": "supported", "parsed": True}],
                    "touches": {"1": [], "2": []}}}
        with tempfile.TemporaryDirectory() as tmp:
            report = summarize(plan, records, FakeClient(tmp))
        self.assertFalse(report["inference_completed"])
        self.assertIsNone(report["S6"])
        self.assertIsNone(report["calendar_365"])
        self.assertEqual(report["S6_withheld_reason"], "Incomplete cohort")


if __name__ == "__main__":
    unittest.main()
