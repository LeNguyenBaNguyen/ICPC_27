"""Validate two-case adjudication and derive a transparent consensus reference.

Agreement rows retain their shared label.  Disagreement rows use only the
submitted adjudicator label.  This is a reference for recorded judgments, not
semantic ground truth; optional provenance is recorded rather than inferred.
"""
import csv
import hashlib
import json
from pathlib import Path

from analyze_human_annotation import FIELDS, LABELS, ROOT, VAL, metrics, read_csv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    analysis = VAL / 'two_rater_analysis'
    prior = json.loads((analysis / 'summary.json').read_text(encoding='utf-8'))
    expected = prior['adjudication_required']
    packet = analysis / 'adjudication_packet'
    with (packet / 'adjudication.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['task_id', 'status', 'evidence', 'adjudicator_notes']:
            raise ValueError('Adjudication CSV header mismatch')
        decisions = list(reader)
    if [row['task_id'] for row in decisions] != expected:
        raise ValueError('Adjudication IDs must exactly match the disagreement order')
    if any(row['status'] not in LABELS or not row['evidence'].strip() or not row['adjudicator_notes'].strip()
           for row in decisions):
        raise ValueError('Incomplete adjudication decision')
    adjudicated = {row['task_id']: row for row in decisions}

    _, labels_1 = read_csv(VAL / 'annotator_1.csv')
    _, labels_2 = read_csv(VAL / 'annotator_2.csv')
    by_1, by_2 = ({row['task_id']: row for row in labels} for labels in (labels_1, labels_2))
    _, key_rows = read_csv(VAL / 'private/sampling_key.csv')
    key = {row['task_id']: row for row in key_rows}
    rows = []
    decisions_out = []
    for task_id in sorted(by_1):
        first, second = by_1[task_id]['status'], by_2[task_id]['status']
        final = adjudicated[task_id]['status'] if task_id in adjudicated else first
        if task_id not in adjudicated and first != second:
            raise ValueError('Non-adjudicated disagreement: ' + task_id)
        rows.append(dict(key[task_id], status=final))
        if task_id in adjudicated:
            decisions_out.append({
                'task_id': task_id, 'rater_1': first, 'rater_2': second, 'adjudicated_status': final,
                'evidence': adjudicated[task_id]['evidence'], 'adjudicator_notes': adjudicated[task_id]['adjudicator_notes'],
            })
    raw_weights = [1.0] * len(rows)
    survey_weights = [float(row['weight']) for row in rows]
    provenance = packet / 'adjudicator_metadata.md'
    summary = {
        'schema_version': 1,
        'design': 'Two-rater blinded audit with consensus on agreements and adjudicator labels for two disagreements',
        'validated_rows': len(rows),
        'adjudicated_rows': len(decisions),
        'consensus_vs_bulk_model': {
            'raw': metrics(rows, raw_weights),
            'weighted': metrics(rows, survey_weights),
        },
        'input_sha256': {
            'two_rater_summary.json': sha(analysis / 'summary.json'),
            'adjudication.csv': sha(packet / 'adjudication.csv'),
            'annotator_1.csv': sha(VAL / 'annotator_1.csv'),
            'annotator_2.csv': sha(VAL / 'annotator_2.csv'),
        },
        'adjudicator_metadata_present': provenance.exists(),
        'limits': (
            'The adjudication CSV supplies decisions and evidence but no adjudicator provenance file was supplied.'
            if not provenance.exists() else
            'Adjudicator provenance is recorded in adjudicator_metadata.md; all self-reports remain unverified.'
        ),
    }
    (analysis / 'adjudicated_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['task_id', 'rater_1', 'rater_2', 'adjudicated_status', 'evidence', 'adjudicator_notes']
    with (analysis / 'adjudication_decisions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(decisions_out)
    print(json.dumps({
        'validated_rows': len(rows), 'adjudicated_rows': len(decisions),
        'metadata_present': provenance.exists(),
        'raw_agreement_with_bulk': summary['consensus_vs_bulk_model']['raw']['three_way_agreement'],
        'raw_kappa_with_bulk': summary['consensus_vs_bulk_model']['raw']['three_way_kappa'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
