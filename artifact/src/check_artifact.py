"""Read-only, standard-library integrity and result check for the review release."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    checksum_file = ROOT/'CHECKSUMS.sha256'
    if not checksum_file.is_file():
        raise SystemExit('Run this check inside the extracted reviewer ZIP.')
    checked = 0
    for line in checksum_file.read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ',1)
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise SystemExit('Missing/unsafe manifest entry: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit('Checksum mismatch: '+name)
        checked += 1
    validation = ROOT/'artifact/validation'
    with (validation/'annotator_1.csv').open(encoding='utf-8-sig',newline='') as stream:
        labels = list(csv.DictReader(stream))
    if len(labels)!=400 or len({r['task_id'] for r in labels})!=400:
        raise SystemExit('Human-label count/identity mismatch')
    human = json.loads((validation/'human_analysis/summary.json').read_text(encoding='utf-8'))
    for name,digest in human['input_sha256'].items():
        if hashlib.sha256((validation/name).read_bytes()).hexdigest()!=digest:
            raise SystemExit('Human analysis input mismatch: '+name)
    expected = json.loads((ROOT/'EXPECTED_RESULTS.json').read_text(encoding='utf-8'))
    primary = json.loads((ROOT/'artifact/results/summary.json').read_text(encoding='utf-8'))
    if primary['main_S_K']!=expected['primary_main_S6']:
        raise SystemExit('Primary point estimate mismatch')
    for field,value in expected['human_weighted'].items():
        if abs(human['weighted'][field]-value)>1e-12:
            raise SystemExit('Human metric mismatch: '+field)
    counts = {}
    ledger = json.loads((ROOT/'artifact/experiments/final_execution.json').read_text(encoding='utf-8'))
    for name,count in expected['supplemental_responses'].items():
        folder=ROOT/'artifact/experiments'/name
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        counts[name]=len(list((folder/'cache').glob('*.json')))
        if not summary['complete'] or counts[name]!=count:
            raise SystemExit('Supplemental count/completion mismatch: '+name)
        for filename,key in [('runtime.json','runtime_sha256'),('summary.json','summary_sha256')]:
            if hashlib.sha256((folder/filename).read_bytes()).hexdigest()!=ledger['conditions'][name][key]:
                raise SystemExit('Release-ledger hash mismatch: '+name+'/'+filename)
    if sum(counts.values())!=1800:
        raise SystemExit('Supplemental total mismatch')
    print(json.dumps({'files_verified':checked,'human_rows':len(labels),'supplemental_cached_responses':sum(counts.values()),
                      'primary_S6':primary['main_S_K'],'integrity_checks_passed':True,
                      'network_used':False,'fresh_model_calls':0},indent=2))


if __name__=='__main__': main()
