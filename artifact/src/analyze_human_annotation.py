"""Validate and analyze submitted single-rater labels without changing raw inputs.

Targets the finite frame of recorded judgments, not unobserved semantic failures.
Unresolved human judgments remain a separate category. All accuracy terminology
is conditional on this rater as the reference, not adjudicated ground truth.
"""
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / 'artifact/validation'
LABELS = ('supported', 'contradicted', 'unresolved')
FIELDS = ['task_id', 'status', 'evidence', 'decomposition_ok', 'claim_type', 'needs_context']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_summary():
    path = VAL/'human_analysis/summary.json'
    if not path.exists(): return None
    summary = json.loads(path.read_text(encoding='utf-8'))
    for name,digest in summary['input_sha256'].items():
        if sha(VAL/name) != digest: raise ValueError('Stale human analysis: '+name)
    return summary


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def validate(labels, key, tasks, manifest):
    ids = [r['task_id'] for r in labels]
    expected = {r['task_id'] for r in tasks}
    if len(ids) != 400 or len(set(ids)) != 400 or set(ids) != expected or set(key) != expected:
        raise ValueError('Missing, duplicate or unexpected task IDs')
    allowed = {'status': set(LABELS), 'decomposition_ok': {'yes', 'no', 'uncertain'},
               'claim_type': {'structural', 'dataflow', 'behavioural', 'intent', 'uncertain'},
               'needs_context': {'yes', 'no'}}
    for r in labels:
        if set(r) != set(FIELDS) or any(r[c] not in values for c, values in allowed.items()):
            raise ValueError('Invalid fields/labels: ' + r['task_id'])
        if not r['evidence'].strip():
            raise ValueError('Empty evidence: ' + r['task_id'])
    for s in manifest['strata']:
        group = [k for k in key.values() if (k['stage'], k['bulk_status']) == (s['stage'], s['bulk_status'])]
        if len(group) != s['sample_n']:
            raise ValueError('Stratum count mismatch')
        for k in group:
            if int(k['population_n']) != s['population_n'] or int(k['sample_n']) != s['sample_n']:
                raise ValueError('Stratum metadata mismatch')
            if not math.isclose(float(k['weight']), s['population_n']/s['sample_n']):
                raise ValueError('Invalid inverse inclusion weight')


def metrics(rows, weights):
    matrix = {m: {h: 0.0 for h in LABELS} for m in LABELS}
    for r, w in zip(rows, weights):
        matrix[r['bulk_status']][r['status']] += w
    total = sum(sum(v.values()) for v in matrix.values())
    def divide(a, b):
        return a/b if b else None
    agreement = sum(matrix[s][s] for s in LABELS)/total
    expected = sum(sum(matrix[s].values()) * sum(matrix[m][s] for m in LABELS) for s in LABELS)/total**2
    resolved = [r['status'] != 'unresolved' for r in rows]
    tp = sum(w for r,w in zip(rows,weights) if r['status']=='contradicted' and r['bulk_status']=='contradicted')
    fn = sum(w for r,w in zip(rows,weights) if r['status']=='contradicted' and r['bulk_status']!='contradicted')
    tn = sum(w for r,w in zip(rows,weights) if r['status']=='supported' and r['bulk_status']!='contradicted')
    fp = sum(w for r,w in zip(rows,weights) if r['status']=='supported' and r['bulk_status']=='contradicted')
    btotal = tp+tn+fp+fn
    ba = divide(tp+tn,btotal)
    be = divide((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp), btotal*btotal)
    return {'three_way_agreement': agreement, 'three_way_kappa': divide(agreement-expected,1-expected),
            'human_unresolved_share': sum(w for r,w in zip(rows,weights) if r['status']=='unresolved')/total,
            'binary_resolved_agreement': ba, 'binary_resolved_kappa': divide(ba-be,1-be) if ba is not None and be is not None else None,
            'contradiction_sensitivity_resolved_reference': divide(tp,tp+fn),
            'contradiction_specificity_resolved_reference': divide(tn,tn+fp),
            'contradiction_precision_resolved_reference': divide(tp,tp+fp),
            'resolved_reference_weight': btotal, 'total_weight': total, 'matrix_model_rows_human_columns': matrix}


def design_agreement_interval(rows, strata):
    """Stratified SRSWOR variance for fixed finite-frame agreement indicators.

    This is sampling-design uncertainty, not rater uncertainty. Repeated functions
    do not change the observation-level sampling units of the frozen design.
    """
    variance, total = 0.0, sum(s['population_n'] for s in strata)
    for s in strata:
        values = [int(r['status']==r['bulk_status']) for r in rows
                  if (r['stage'],r['bulk_status'])==(s['stage'],s['bulk_status'])]
        n, N = len(values), s['population_n']
        mean = sum(values)/n
        sample_var = sum((x-mean)**2 for x in values)/(n-1)
        variance += N*N*(1-n/N)*sample_var/n
    se = math.sqrt(variance)/total
    estimate = metrics(rows,[float(r['weight']) for r in rows])['three_way_agreement']
    return {'estimate':estimate,'standard_error':se,'normal_95_interval':[max(0,estimate-1.96*se),min(1,estimate+1.96*se)],
            'method':'Stratified observation-level SRSWOR finite-population variance; fixed-rater labels; normal approximation may be poor for sparse errors'}


def cluster_intervals(rows, B=2000, seed=20261111):
    rng = random.Random(seed)
    fids = sorted({r['fid'] for r in rows})
    groups = defaultdict(list)
    for i,r in enumerate(rows): groups[(r['stage'],r['bulk_status'])].append(i)
    base = [float(r['weight']) for r in rows]
    keys = [k for k,v in metrics(rows,base).items() if isinstance(v,(float,int)) and not k.endswith('weight')]
    draws = {k:[] for k in keys}
    for _ in range(B):
        multipliers = {fid:rng.expovariate(1) for fid in fids}
        w = [b*multipliers[r['fid']] for b,r in zip(base,rows)]
        for indices in groups.values():
            scale = sum(base[i] for i in indices)/sum(w[i] for i in indices)
            for i in indices: w[i] *= scale
        result = metrics(rows,w)
        for k in keys:
            if result[k] is not None: draws[k].append(result[k])
    def quantile(xs,p):
        xs = sorted(xs); index=(len(xs)-1)*p; low=int(index)
        return xs[low]+(xs[min(low+1,len(xs)-1)]-xs[low])*(index-low)
    return {'method':'Function-cluster exponential multiplier bootstrap, calibrated to each of six stratum totals; approximate descriptive intervals, no finite-population correction; not exact survey-design or rater-error intervals',
            'replicates':B,'seed':seed,'functions':len(fids),
            'intervals':{k:[quantile(v,.025),quantile(v,.975)] for k,v in draws.items() if v}}


def main():
    fields, labels = read_csv(VAL/'annotator_1.csv')
    if fields != FIELDS: raise ValueError('CSV header mismatch')
    _, keys = read_csv(VAL/'private/sampling_key.csv')
    key = {r['task_id']:r for r in keys}
    tasks = [json.loads(line) for line in (VAL/'blinded/tasks.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((VAL/'private/sampling_manifest.json').read_text(encoding='utf-8'))
    validate(labels,key,tasks,manifest)
    for name,digest in manifest['source_sha256'].items():
        if sha(ROOT/'artifact/results'/name) != digest: raise ValueError('Historical input hash mismatch: '+name)
    rows = [dict(key[r['task_id']],**{k:v for k,v in r.items() if k!='task_id'}) for r in labels]
    summary = {'schema_version':1,'design':'Post-plan single-rater exploratory audit; no second rater or adjudication',
               'annotation_complete':True,'annotators':1,'validated_rows':len(rows),
               'input_sha256':{name:sha(VAL/name) for name in ('annotator_1.csv','annotator_notes.md','blinded/tasks.jsonl','private/sampling_key.csv','private/sampling_manifest.json')},
               'provenance':'A human rater supplied these labels; experience/blinding are self-reported in annotator_notes.md, not independently verified',
               'raw':metrics(rows,[1]*len(rows)),'weighted':metrics(rows,[float(r['weight']) for r in rows]),
               'status_counts':dict(Counter(r['status'] for r in rows)),
               'decomposition_counts':dict(Counter(r['decomposition_ok'] for r in rows)),
               'claim_type_counts':dict(Counter(r['claim_type'] for r in rows)),
               'needs_context_counts':dict(Counter(r['needs_context'] for r in rows)),
               'design_based_agreement':design_agreement_interval(rows,manifest['strata']),
               'function_cluster_sensitivity':cluster_intervals(rows),
               'by_stage':{},'by_arm':{},'by_cohort':{},'by_stratum':{}}
    for field,out in [('stage','by_stage'),('arm','by_arm'),('cohort','by_cohort')]:
        for value in sorted({r[field] for r in rows}):
            subset=[r for r in rows if r[field]==value]
            summary[out][value]={'sample_n':len(subset),'raw':metrics(subset,[1]*len(subset)),
                                 'weighted':metrics(subset,[float(r['weight']) for r in subset])}
    for s in manifest['strata']:
        subset=[r for r in rows if (r['stage'],r['bulk_status'])==(s['stage'],s['bulk_status'])]
        summary['by_stratum'][s['stage']+'/'+s['bulk_status']]={**s,'human_counts':dict(Counter(r['status'] for r in subset))}
    out=VAL/'human_analysis'; out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'disagreements.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        names=['task_id','stage','bulk_status','status','weight','evidence','decomposition_ok','claim_type','needs_context']
        writer=csv.DictWriter(stream,fieldnames=names,extrasaction='ignore'); writer.writeheader()
        writer.writerows(r for r in rows if r['status']!=r['bulk_status'])
    with (out/'review_flags.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        names=['task_id','status','decomposition_ok','needs_context','evidence']
        writer=csv.DictWriter(stream,fieldnames=names,extrasaction='ignore'); writer.writeheader()
        writer.writerows(r for r in rows if r['decomposition_ok']!='yes' or r['needs_context']=='yes')
    raw, weighted = summary['raw'], summary['weighted']
    ci = summary['function_cluster_sensitivity']['intervals']
    table = '\n'.join('| '+m+' | '+' | '.join(str(int(raw['matrix_model_rows_human_columns'][m][h])) for h in LABELS)+' |' for m in LABELS)
    report = f'''# Kết quả đánh giá sơ bộ bởi một người

Đã nhận và kiểm tra **400/400 trường hợp**, 400 mã duy nhất, đầy đủ năm trường nhãn/giải thích và đúng tập giá trị cho phép. Không chỉnh sửa nhãn đã nộp. Dữ liệu nguồn, hash và phương pháp nằm trong [summary.json](artifact/validation/human_analysis/summary.json).

Một người chấm đã cung cấp bộ nhãn. Theo `annotator_notes.md`, A1 có 10 năm kinh nghiệm nghiên cứu/đọc Python, chấm ngày 11/11/2026, chưa xem manuscript hoặc nhãn mô hình và chỉ dùng bộ mã được cung cấp. Đây là khai báo của người chấm, không phải thông tin được agent xác minh độc lập. Không có người thứ hai hay phân xử giữa những người chấm. Thiết kế đã đổi sau kế hoạch từ hai người sang một người.

## Kết quả

| Chỉ số | Giá trị |
| --- | ---: |
| Nhãn người: supported / contradicted / unresolved | 160 / 137 / 103 |
| Đồng thuận ba nhãn, không trọng số | {100*raw['three_way_agreement']:.2f}% (369/400) |
| Kappa ba nhãn, không trọng số | {raw['three_way_kappa']:.4f} |
| Đồng thuận ba nhãn, có trọng số | {100*weighted['three_way_agreement']:.2f}% |
| Kappa ba nhãn, có trọng số | {weighted['three_way_kappa']:.4f} |
| Nhận định bị chấm gộp ý | 28/400 |
| Cần thêm ngữ cảnh | 103/400 |

Ma trận **số trường hợp thực chấm**, hàng là mô hình, cột là người; không phải ước lượng số trường hợp của toàn bộ quần thể:

| Mô hình / Người | supported | contradicted | unresolved |
| --- | ---: | ---: | ---: |
{table}

Có 31 bất đồng: 28 trường hợp mô hình unresolved nhưng người supported, hai trường hợp unresolved → contradicted và một trường hợp supported → contradicted. Xem [disagreements.csv](artifact/validation/human_analysis/disagreements.csv). Không đổi các nhãn gốc để xóa bất đồng. Các nhận định gộp ý hoặc cần ngữ cảnh nằm trong [review_flags.csv](artifact/validation/human_analysis/review_flags.csv); đây là danh sách cần lưu ý, không phải yêu cầu ép nhãn mới.

Trọng số nghịch đảo xác suất lấy mẫu đưa sáu tầng về tập **14.511 phán xét đã ghi**. Tập này có nhiều supported; vì vậy đồng thuận có trọng số cao hơn đồng thuận thô. Trọng số không biến 400 trường hợp thành 14.511 trường hợp do người trực tiếp chấm. Kết quả theo giai đoạn, nhánh và cohort nằm trong JSON; các nhóm nhỏ chỉ mang tính mô tả.

## Bất định và giới hạn

- Khoảng mô tả 95% từ 2.000 lần nhân trọng số ngẫu nhiên theo cụm hàm, hiệu chỉnh lại tổng từng tầng: đồng thuận có trọng số [{100*ci['three_way_agreement'][0]:.2f}%, {100*ci['three_way_agreement'][1]:.2f}%], kappa [{ci['three_way_kappa'][0]:.3f}, {ci['three_way_kappa'][1]:.3f}]. Đây là xấp xỉ theo cụm, không phải khoảng chính xác của thiết kế khảo sát hoặc sai số người chấm.
- JSON cũng báo phương sai thiết kế lấy mẫu ngẫu nhiên không hoàn lại trong từng tầng, với nhãn người được coi là giá trị cố định. Khoảng chuẩn xấp xỉ có thể kém khi sai khác thưa. Hai cách đánh giá bất định có giả định khác nhau; cả hai đều không đo được thiên lệch của một người chấm.
- Với 297 trường hợp người đưa ra kết luận supported/contradicted, sensitivity phát hiện contradiction có trọng số so với người này là {100*weighted['contradiction_sensitivity_resolved_reference']:.2f}%. Nhãn người unresolved bị loại khỏi riêng phép tính nhị phân này, vẫn giữ trong ma trận ba nhãn. Mô hình unresolved được tính là không phát hiện contradiction trong phép tính nhị phân. Specificity/precision quan sát bằng 100% chỉ phản ánh không thấy loại sai khác tương ứng trong mẫu, không chứng minh mô hình không mắc lỗi.
- Các chỉ số dùng **một người làm tham chiếu**, không phải độ chính xác đối với chân lý ngữ nghĩa đã phân xử. Không có đồng thuận giữa người với người. Bằng chứng và 28 nhãn gộp ý vẫn cần được diễn giải thận trọng; chưa kiểm tra đầy đủ các ý bị bỏ sót khi tách nhận định.
- Bộ mẫu chỉ chứa các phán xét đã được lưu; lần theo dõi sau đó phụ thuộc quy tắc dừng ở contradiction đầu tiên. Không có nhãn người cho toàn bộ phiên bản lịch sử, thời điểm failure, resurrection hay mọi nhận định bị bỏ sót. Không dùng audit này để tự động hiệu chỉnh hazard/survival hoặc gọi đường survival là đã xác thực ngữ nghĩa.

Các số liệu survival lịch sử và bốn thí nghiệm tự động được giữ nguyên. Artifact sửa đổi vẫn là gói cục bộ, chưa có URL phát hành ẩn danh đã xác minh.
'''
    (ROOT/'HUMAN_EVALUATION_RESULTS.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('validated_rows','status_counts','decomposition_counts','raw','weighted','design_based_agreement')},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
