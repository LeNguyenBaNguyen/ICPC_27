import unittest
import json
from analyze_human_annotation import metrics, validate, read_csv, VAL, current_summary


class HumanAnalysisTests(unittest.TestCase):
    def test_unresolved_reference_excluded_from_binary_not_three_way(self):
        rows = [{'bulk_status':'contradicted','status':'contradicted'},
                {'bulk_status':'supported','status':'contradicted'},
                {'bulk_status':'contradicted','status':'unresolved'},
                {'bulk_status':'unresolved','status':'supported'}]
        m = metrics(rows,[1,1,1,1])
        self.assertEqual(m['three_way_agreement'],.25)
        self.assertEqual(m['resolved_reference_weight'],3)
        self.assertEqual(m['contradiction_sensitivity_resolved_reference'],.5)
        self.assertEqual(m['contradiction_specificity_resolved_reference'],1)
        self.assertEqual(m['human_unresolved_share'],.25)

    def test_weights_change_target(self):
        rows = [{'bulk_status':'supported','status':'supported'},
                {'bulk_status':'supported','status':'contradicted'}]
        self.assertAlmostEqual(metrics(rows,[99,1])['three_way_agreement'],.99)
        self.assertEqual(metrics(rows,[1,1])['three_way_agreement'],.5)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            validate([{'task_id':'V0001'}]*400,{},[],{'strata':[]})

    def test_submitted_packet_valid_and_hash_bound(self):
        _,labels = read_csv(VAL/'annotator_1.csv')
        _,keys = read_csv(VAL/'private/sampling_key.csv')
        tasks = [json.loads(line) for line in (VAL/'blinded/tasks.jsonl').read_text(encoding='utf-8').splitlines()]
        manifest = json.loads((VAL/'private/sampling_manifest.json').read_text(encoding='utf-8'))
        validate(labels,{r['task_id']:r for r in keys},tasks,manifest)
        self.assertEqual(current_summary()['validated_rows'],400)


if __name__=='__main__': unittest.main()
