"""Analyze two blinded human-rater files and prepare third-rater adjudication.

This script never changes submitted labels.  It reports agreement, retains the
two rater-to-model matrices separately, and writes a packet containing only the
original blinded material for records on which A1 and A2 disagree.
"""
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from analyze_human_annotation import FIELDS, LABELS, ROOT, VAL, metrics, read_csv, validate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kappa(rows, weights):
    """Three-way agreement and Cohen's kappa; rows contain rater_1/rater_2."""
    matrix = {a: {b: 0.0 for b in LABELS} for a in LABELS}
    for row, weight in zip(rows, weights):
        matrix[row['rater_1']][row['rater_2']] += weight
    total = sum(sum(cols.values()) for cols in matrix.values())
    observed = sum(matrix[label][label] for label in LABELS) / total
    expected = sum(
        sum(matrix[left].values()) / total
        * sum(matrix[right][left] for right in LABELS) / total
        for left in LABELS
    )
    return {
        'agreement': observed,
        'kappa': (observed - expected) / (1 - expected) if expected != 1 else 1.0,
        'matrix_rater_1_rows_rater_2_columns': matrix,
    }


def main():
    header_1, labels_1 = read_csv(VAL / 'annotator_1.csv')
    header_2, labels_2 = read_csv(VAL / 'annotator_2.csv')
    if header_1 != FIELDS or header_2 != FIELDS:
        raise ValueError('CSV header mismatch')
    _, key_rows = read_csv(VAL / 'private/sampling_key.csv')
    key = {row['task_id']: row for row in key_rows}
    tasks = [json.loads(line) for line in (VAL / 'blinded/tasks.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((VAL / 'private/sampling_manifest.json').read_text(encoding='utf-8'))
    validate(labels_1, key, tasks, manifest)
    validate(labels_2, key, tasks, manifest)

    by_1 = {row['task_id']: row for row in labels_1}
    by_2 = {row['task_id']: row for row in labels_2}
    if set(by_1) != set(by_2):
        raise ValueError('Rater task IDs differ')
    rows = []
    for task_id in sorted(by_1):
        row = dict(key[task_id])
        row['rater_1'] = by_1[task_id]['status']
        row['rater_2'] = by_2[task_id]['status']
        rows.append(row)
    raw_weights = [1.0] * len(rows)
    population_weights = [float(row['weight']) for row in rows]
    disagreements = [row for row in rows if row['rater_1'] != row['rater_2']]

    def rater_metrics(labels):
        model_rows = []
        for task_id in sorted(labels):
            model_rows.append(dict(key[task_id], **{field: labels[task_id][field] for field in FIELDS if field != 'task_id'}))
        return {
            'raw': metrics(model_rows, raw_weights),
            'weighted': metrics(model_rows, population_weights),
        }

    summary = {
        'schema_version': 1,
        'design': 'Post-plan two-rater blinded audit; adjudication pending for disagreements',
        'validated_rows': len(rows),
        'input_sha256': {
            name: sha(VAL / name) for name in (
                'annotator_1.csv', 'annotator_2.csv', 'annotator_notes.md',
                'annotator_2_notes.md', 'blinded/tasks.jsonl', 'private/sampling_key.csv',
                'private/sampling_manifest.json',
            )
        },
        'raw': kappa(rows, raw_weights),
        'weighted': kappa(rows, population_weights),
        'disagreements': len(disagreements),
        'rater_1_vs_bulk_model': rater_metrics(by_1),
        'rater_2_vs_bulk_model': rater_metrics(by_2),
        'rater_1_status_counts': dict(Counter(row['rater_1'] for row in rows)),
        'rater_2_status_counts': dict(Counter(row['rater_2'] for row in rows)),
        'adjudication_required': [row['task_id'] for row in disagreements],
        'limits': (
            'Agreement is between two self-reported blinded raters on the recorded-judgment frame. '
            'It does not validate semantic truth, recall, unobserved versions, or rater independence; '
            'one rater disclosed a social acquaintance of an author. No consensus label is created here.'
        ),
    }
    out = VAL / 'two_rater_analysis'
    out.mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (out / 'disagreements.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['task_id', 'rater_1', 'rater_2'])
        writer.writeheader()
        writer.writerows({'task_id': row['task_id'], 'rater_1': row['rater_1'], 'rater_2': row['rater_2']} for row in disagreements)

    packet = out / 'adjudication_packet'
    cases = packet / 'cases'
    cases.mkdir(parents=True, exist_ok=True)
    for row in disagreements:
        shutil.copy2(VAL / 'blinded/cases' / (row['task_id'] + '.md'), cases / (row['task_id'] + '.md'))
    (packet / 'adjudication.csv').write_text(
        'task_id,status,evidence,adjudicator_notes\n' + ''.join(row['task_id'] + ',,,\n' for row in disagreements),
        encoding='utf-8',
    )
    (packet / 'README.md').write_text(
        '''# Blind adjudication packet

Review each supplied case independently from its original code packet. Do not consult model labels, A1/A2 labels, their evidence, the manuscript, or reported results before recording a decision.

For every row in `adjudication.csv`, use exactly one status: `supported`, `contradicted`, or `unresolved`; provide concrete code evidence; and state any remaining uncertainty in `adjudicator_notes`. Preserve the task IDs and CSV header. This packet intentionally does not include either rater's answer or the bulk model label.
''',
        encoding='utf-8',
    )
    print(json.dumps({
        'validated_rows': len(rows), 'raw_agreement': summary['raw']['agreement'],
        'raw_kappa': summary['raw']['kappa'], 'disagreements': summary['disagreements'],
        'packet': str(packet),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
