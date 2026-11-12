"""Safety and design checks for supplemental inference, without model calls."""
import collections
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_missing_experiments as runner


class AtomicWriteTests(unittest.TestCase):
    def test_transient_windows_file_lock_keeps_atomic_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observations.json"
            path.write_text('{"previous": true}', encoding="utf-8")
            original_replace = runner.os.replace
            attempts = []

            def locked_twice(source, target):
                attempts.append((source, target))
                if len(attempts) <= 2:
                    raise PermissionError(13, "Access is denied")
                return original_replace(source, target)

            with patch.object(runner.os, "replace", side_effect=locked_twice), patch.object(runner.time, "sleep"):
                runner.atomic_json(path, {"completed": 567})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"completed": 567})
            self.assertEqual(len(attempts), 3)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


class FrozenDesignTests(unittest.TestCase):
    def test_resurrection_does_not_absorb_after_original_contradiction(self):
        f = {"id": "F", "cohort": "main", "qualname": "f", "path": "f.py",
             "anchor": {"text": "def f():\n    return 1", "params": [], "facts": {}},
             "touches": [{"new": {"text": f"def f():\n    return {k}", "params": [], "facts": {}}} for k in range(2, 6)]}
        claims = {"F": {"llm": [{"text": "f returns one", "type": "behavioural"}], "doc": []}}
        v = {"F": {"anchor": [{"arm": "llm", "idx": 0, "status": "supported"}],
                   "commits": [{"k": 1, "judged": [["llm", 0]], "verdicts": [{"arm": "llm", "idx": 0, "status": "contradicted"}]},
                               {"k": 2, "judged": [], "verdicts": []}, {"k": 3, "judged": [], "verdicts": []},
                               {"k": 4, "judged": [], "verdicts": []}]}}
        with patch.object(runner, "make_judge_task", side_effect=lambda fid, f, k, items, task_id, extra: {"k": k, "items": [{"arm": a, "idx": i, **c} for a, i, c in items], **extra}):
            plan = runner.resurrection_plan({"F": f}, claims, v)
        self.assertEqual(plan["eligible_claims"], 1)
        self.assertEqual([t["k"] for t in plan["tasks"] if t["role"] == "followup"], [2, 3, 4])
        self.assertEqual([t["k"] for t in plan["tasks"] if t["role"] == "death_control"], [1])
        self.assertTrue(all(len(t["items"]) == 1 for t in plan["tasks"]))

    def test_cross_family_weights_and_original_context(self):
        plan = json.loads((runner.EXPERIMENTS / "plans/cross_family.json").read_text(encoding="utf-8"))
        cases = [case for task in plan["tasks"] for case in task["sampled_cases"]]
        self.assertEqual(len(cases), 60)
        self.assertEqual(set(collections.Counter((c["stage"], c["bulk_status"]) for c in cases).values()), {10})
        for task in plan["tasks"]:
            original = runner._key("ollama", runner.CONFIG["bulk_model"], runner.CONFIG["seed"], task["system"], task["prompt"])
            self.assertTrue((runner.RESULTS / "llm_cache" / (original + ".json")).exists())
            for case in task["sampled_cases"]:
                self.assertAlmostEqual(case["weight"], case["original_stratum_population_n"] / 10)
                item = task["items"][case["statement_position"]]
                self.assertEqual((item["arm"], item["idx"]), (case["arm"], case["idx"]))

    def test_death_controls_reuse_original_context_and_cover_every_eligible_claim(self):
        plan = json.loads((runner.EXPERIMENTS / "plans/resurrection.json").read_text(encoding="utf-8"))
        controls = [task for task in plan["tasks"] if task["role"] == "death_control"]
        self.assertEqual(len(controls), 163)
        self.assertEqual(sum(len(t["scored_cases"]) for t in controls), 232)
        for task in controls:
            original = runner._key("ollama", runner.CONFIG["bulk_model"], runner.CONFIG["seed"], task["system"], task["prompt"])
            self.assertTrue((runner.RESULTS / "llm_cache" / (original + ".json")).exists())

    def test_weighted_agreement_and_binary_kappa(self):
        rows = [("F1", "supported", "supported", 1), ("F2", "contradicted", "contradicted", 1)]
        result = runner.weighted_agreement_stats(rows)
        self.assertEqual(result["agreement"], 1)
        self.assertEqual(result["binary_contradiction_kappa"], 1)
        ci = runner.cluster_agreement_intervals(rows, 20261112, 30)
        self.assertEqual(ci["agreement"]["low"], 1)
        self.assertEqual(ci["agreement"]["high"], 1)

    def test_lexical_coverage_is_word_bounded(self):
        f = {"anchor": {"params": ["data", "x"], "facts": {"calls": ["self.helper"]}}}
        result = runner.lexical_coverage(f, "The function examines metadata then calls helper.")
        self.assertEqual(result["parameters"]["mentioned"], 0)
        self.assertEqual(result["helpers"]["mentioned"], 1)

    def test_new_claim_survival_is_first_contradiction_even_if_later_supported(self):
        f = {"id": "F", "touches": [{}, {}, {}]}
        record = {"anchor": [{"status": "supported"}],
                  "touches": {"1": [{"status": "unresolved"}], "2": [{"status": "contradicted"}], "3": [{"status": "supported"}]}}
        with patch.object(runner, "load", return_value={"F": {"commits": [{"k": 1}, {"k": 2}, {"k": 3}]}}):
            result = runner.new_claim_survival(f, record)
        self.assertEqual(result, [{"idx": 0, "time_k": 2, "event": 1}])


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.patch = patch.object(runner, "EXPERIMENTS", self.directory)
        self.patch.start()
        runner.atomic_json(self.directory / "plans/resurrection.json", {"model": "qwen2.5-coder:14b", "options": runner.OPTIONS, "plan_sha256": "frozen"})

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_zero_budget_never_contacts_server(self):
        client = runner.Client("resurrection", "http://invalid", 0)
        with patch.object(client, "request", side_effect=AssertionError("network forbidden")):
            with self.assertRaises(runner.BudgetExhausted):
                client.generate("one", "prompt", "system", 100)

    def test_cached_resume_never_calls_model_twice(self):
        client = runner.Client("resurrection", "http://invalid", 1)
        response = {"response": "S1: supported", "done": True, "prompt_eval_count": 10}
        with patch.object(client, "fingerprint_model", return_value={"digest": "fixed"}), patch.object(client, "request", return_value=response):
            self.assertEqual(client.generate("one", "prompt", "system", 100), response)
        resumed = runner.Client("resurrection", "http://invalid", 0)
        with patch.object(resumed, "request", side_effect=AssertionError("network forbidden")):
            self.assertEqual(resumed.generate("one", "prompt", "system", 100), response)
        self.assertEqual(resumed.hits, 1)

    def test_context_limit_response_is_not_accepted_into_cache(self):
        client = runner.Client("resurrection", "http://invalid", 1)
        response = {"response": "S1: supported", "done": True, "prompt_eval_count": runner.OPTIONS["num_ctx"]}
        with patch.object(client, "fingerprint_model", return_value={"digest": "fixed"}), patch.object(client, "request", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "context limit"):
                client.generate("one", "prompt", "system", 100)
        self.assertFalse(list((self.directory / "resurrection/cache").glob("*.json")))
        self.assertEqual(len(list((self.directory / "resurrection/errors").glob("*.json"))), 1)

    def test_output_reserve_can_trigger_context_rejection(self):
        client = runner.Client("resurrection", "http://invalid", 1)
        response = {"response": "S1: supported", "done": True, "prompt_eval_count": runner.OPTIONS["num_ctx"] - 50}
        with patch.object(client, "fingerprint_model", return_value={"digest": "fixed"}), patch.object(client, "request", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "output reserve"):
                client.generate("one", "prompt", "system", 100)

    def test_length_limited_explanation_is_rejected_but_raw_output_is_kept(self):
        client = runner.Client("resurrection", "http://invalid", 1)
        response = {"response": "Incomplete explanation", "done": True, "done_reason": "length", "prompt_eval_count": 50}
        with patch.object(client, "fingerprint_model", return_value={"digest": "fixed"}), patch.object(client, "request", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "output length limit"):
                client.generate("F000:generate", "prompt", "system", 100)
        saved = json.loads(next((self.directory / "resurrection/errors").glob("*.json")).read_text())
        self.assertEqual(saved["response"], response)
        self.assertEqual(client.quality_counts()["rejected_length_limited_responses"], 1)

    def test_runtime_fingerprint_change_is_rejected(self):
        client = runner.Client("resurrection", "http://invalid", 1)
        runner.atomic_json(client.directory / "runtime.json", {"digest": "old", "options": runner.OPTIONS})
        tags = {"models": [{"name": "qwen2.5-coder:14b", "digest": "new"}]}
        with patch.object(client, "request", side_effect=[tags, {}]):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                client.fingerprint_model()

    def test_failure_replaces_stale_complete_summary(self):
        client = runner.Client("resurrection", "http://invalid", 1)
        item = {"arm": "llm", "idx": 0, "type": "structural", "text": "claim"}
        client.plan.update({"tasks": [{"task_id": "one", "fid": "F", "mining_k": 2, "items": [item], "role": "followup"}],
                            "eligible_claim_details": [{"fid": "F", "arm": "llm", "idx": 0, "type": "structural", "cohort": "main"}]})
        runner.atomic_json(client.directory / "summary.json", {"complete": True})
        with patch.object(runner, "judge", side_effect=RuntimeError("server failed")):
            result = runner.run_judgments("resurrection", client)
        self.assertTrue(result["failed"])
        self.assertFalse(result["complete"])
        saved = json.loads((client.directory / "summary.json").read_text())
        self.assertEqual(saved["error"], "RuntimeError('server failed')")


if __name__ == "__main__":
    unittest.main()
