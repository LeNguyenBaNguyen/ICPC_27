"""Endpoint and blinding checks for the supplemental validation workflow."""
import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from prepare_validation import km, sensitivity, without_docstrings


class ObservationTests(unittest.TestCase):
    def records(self, statuses, format_k=(), anchor="supported"):
        f = {"cohort": "main", "touches": [{"format_only": k in format_k} for k in range(1, len(statuses) + 1)]}
        v = {"anchor": [{"arm": "llm", "idx": 0, "status": anchor}],
             "commits": [{"k": k, "judged": [["llm", 0]],
                          "verdicts": [{"status": status}]} for k, status in enumerate(statuses, 1)]}
        return {"F": f}, {"F": v}

    def run_scenario(self, statuses, expected, format_k=()):
        funcs, verdicts = self.records(statuses, format_k)
        with patch("prepare_validation.load", return_value={"main_S_K": expected}):
            return sensitivity(funcs, verdicts)

    def test_unresolved_then_contradiction(self):
        result = self.run_scenario(["supported", "unresolved", "contradicted"], 0)
        self.assertEqual(result["claims_with_unresolved_before_endpoint"], 1)
        self.assertEqual(result["censor_before_first_unresolved_S6"], 1)
        self.assertEqual(result["first_unresolved_as_event_S6"], 0)

    def test_format_only_contradiction_is_censoring(self):
        result = self.run_scenario(["supported", "contradicted"], 1, format_k=(2,))
        self.assertEqual(result["primary_S6"], 1)
        self.assertEqual(result["claims_with_unresolved_before_endpoint"], 0)

    def test_unresolved_after_first_contradiction_does_not_change_endpoint(self):
        result = self.run_scenario(["contradicted", "unresolved"], 0)
        self.assertEqual(result["claims_with_unresolved_before_endpoint"], 0)

    def test_km_censors_before_unknown_touch(self):
        self.assertEqual(km([(0, 0), (1, 1)])[1], 0)
        self.assertEqual(km([(1, 0), (1, 1)])[1], 0.5)

    def test_decorated_class_slice_masks_docstring_preserves_body(self):
        source = '@wrap\n    def example(self):\n        """secret docstring"""\n        return 42\n'
        result = without_docstrings(source)
        self.assertNotIn("secret", result)
        self.assertIn("    return 42", result)
        compile(result, "example", "exec")

    def test_nested_function_does_not_shift_outer_indentation(self):
        source = '@wrap\ndef outer():\n    def nested():\n        return 1\n    return nested()\n'
        compile(without_docstrings(source), "example", "exec")

    def test_docstring_only_body_has_explicit_placeholder(self):
        source = 'def empty():\n    """Only documentation."""\n'
        result = without_docstrings(source)
        self.assertNotIn("Only documentation", result)
        self.assertIn("pass  # docstring removed", result)
        compile(result, "example", "exec")


class PacketTests(unittest.TestCase):
    def test_archived_packet_has_unique_ids_weights_and_no_label_fields(self):
        root = Path(__file__).resolve().parents[1] / "validation"
        packets = [json.loads(line) for line in (root / "blinded/tasks.jsonl").read_text(encoding="utf-8").splitlines()]
        with (root / "private/sampling_key.csv").open(encoding="utf-8", newline="") as stream:
            keys = list(csv.DictReader(stream))
        ids = [p["task_id"] for p in packets]
        self.assertEqual(len(ids), 400)
        self.assertEqual(len(set(ids)), 400)
        self.assertEqual(set(ids), {p["task_id"] for p in keys})
        allowed = {"task_id", "claim", "anchor_code", "previous_code", "current_code", "diff"}
        self.assertTrue(all(set(p) == allowed for p in packets))
        for row in keys:
            self.assertAlmostEqual(float(row["selection_probability"]) * float(row["weight"]), 1)
        for number in (1, 2):
            with (root / f"annotator_{number}.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["task_id"] for r in rows], ids)
            # Both raters submitted complete labels. The blinded packet itself
            # still carries no label field; rater agreement is analyzed separately.
            self.assertTrue(all(r['status'] in {'supported','contradicted','unresolved'} for r in rows))
            self.assertTrue(all(r['evidence'].strip() for r in rows))


if __name__ == "__main__":
    unittest.main()
