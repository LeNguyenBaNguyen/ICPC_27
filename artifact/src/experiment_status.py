"""Summarize actual experiment files; plans and partial calls are not completion."""
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from analyze_human_annotation import current_summary

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifact/experiments"


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def snapshot():
    status = {"updated": datetime.now(ZoneInfo("Asia/Saigon")).isoformat(), "experiments": {}}
    for name in ("resurrection", "docstring_free", "cross_family", "random_redraw_inference"):
        plan = read(EXP / "plans" / (name + ".json"), {})
        folder = EXP / name
        summary = read(folder / "summary.json", {})
        observations = read(folder / "observations.json", {})
        status["experiments"][name] = {
            "plan_frozen": bool(plan), "complete": summary.get("complete", False),
            "observations_written": len(observations),
            "cached_calls": len(list((folder / "cache").glob("*.json"))),
            "planned_tasks": len(plan.get("tasks", [])), "plan_sha256": plan.get("plan_sha256"),
            "summary": summary, "last_failure": read(folder / "last_failure.json"),
        }
    status["random_redraw_history"] = read(EXP / "random_redraw/report.json", {})
    status["original_environment_replay"] = read(ROOT / "revisions/original-environment-replay/report.json", {})
    human = current_summary()
    status["human_annotation_completed"] = bool(human)
    status["human_annotators"] = human['annotators'] if human else 0
    status["human_annotation_design"] = 'single-rater exploratory audit' if human else 'pending'
    status["inter_rater_agreement_available"] = False
    status['publication']=read(ROOT/'artifact/publication.json')
    status["original_summary_unchanged"] = hashlib.sha256((ROOT / "artifact/results/summary.json").read_bytes()).hexdigest() == "3e63cdc3bce690064a9f917bb893ae8809a6b9256aa0baa53eff42f35cd1e8b3"
    try:
        with urllib.request.urlopen("http://127.0.0.1:11435/api/tags", timeout=2) as response:
            models = json.load(response)
        status["available_model_tags"] = [{"name": m["name"], "digest": m["digest"]} for m in models.get("models", [])]
    except OSError:
        status["available_model_tags"] = []
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = snapshot()
    if args.write:
        (EXP / "campaign_status.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rows = ["# Trạng thái chạy thí nghiệm", "", "Cập nhật: " + data["updated"], "",
                "Đây là trạng thái từ tệp kết quả thực tế. Có kế hoạch hoặc cache chưa có nghĩa là hoàn thành.", "",
                "| Điều kiện | Đã chạy đủ kế hoạch tự động? | Số tệp phản hồi đã lưu | Số bản ghi quan sát |",
                "| --- | --- | ---: | ---: |"]
        for name, value in data["experiments"].items():
            rows.append(f"| {name} | {'Có' if value['complete'] else 'Chưa'} | {value['cached_calls']} | {value['observations_written']} |")
        history = data["random_redraw_history"]
        rows.extend(["", f"Khôi phục lịch sử mẫu mới: {history.get('recovered_n', 0)}/80 hàm; đây là bước trước suy luận mô hình.",
                     "", "Đánh giá người: đã chuẩn bị 400 trường hợp và hai bảng nhãn trống; chưa thu thập nhãn người độc lập hoặc phân xử bất đồng. Kiểm tra của agent không phải đánh giá người.",
                     "", "Artifact sửa đổi mới được đóng gói cục bộ, chưa phát hành và chưa có URL ẩn danh đã kiểm tra quyền truy cập. Hoàn thành chạy không đồng nghĩa mọi nhãn đều đọc được; xem giới hạn từng điều kiện trong `EXPERIMENT_RESULTS.md`.",
                     "", "Agent review đã được thêm vào `review.md`. Kế hoạch cố định: `artifact/experiments/plans/`; kết quả và cache: từng thư mục trong `artifact/experiments/`.",
                     "", "Mô hình Qwen tải mới không có fingerprint gốc để đối chiếu. Kết quả mới là điều kiện bổ sung với runtime được ghi lại, không phải tái tạo chính xác mô hình lịch sử.",
                     "", "Các lệnh chạy, lỗi, phân tích thống kê và giới hạn được tổng hợp trong RUNLOG.md khi kết thúc đợt chạy."])
        if data['human_annotation_completed']:
            rows = [r.replace('Đánh giá người: đã chuẩn bị 400 trường hợp và hai bảng nhãn trống; chưa thu thập nhãn người độc lập hoặc phân xử bất đồng. Kiểm tra của agent không phải đánh giá người.', 'Đánh giá một người: đã nhận và phân tích đủ 400 nhãn. Không có người thứ hai hoặc phân xử; chưa đo đồng thuận giữa những người chấm. Kết quả: `HUMAN_EVALUATION_RESULTS.md`.') for r in rows]
        if (ROOT/'revisions/claim-survival-artifact-v1.zip').exists():
            rows.extend(['', 'Gói reviewer riêng: `revisions/claim-survival-artifact-v1.zip`. Người dùng chọn tự tải lên; hướng dẫn tại `ARTIFACT_UPLOAD.md`. Gói nội bộ `review-bundle.zip` không dùng để phát hành ẩn danh.'])
        publication=data.get('publication')
        if publication and publication.get('public_access_verified'):
            rows=[r for r in rows if not r.startswith('Artifact sửa đổi mới') and not r.startswith('Gói reviewer riêng:')]
            rows.extend(['',f"Artifact đã phát hành công khai: {publication['landing_url']}. ZIP đã tải lại không đăng nhập và khớp checksum; URL thật đã có trong manuscript. Xem `ARTIFACT_PUBLICATION.md`."])
        (ROOT / "EXPERIMENT_STATUS.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"updated": data["updated"], "experiments": {k: {x: v[x] for x in ("complete", "cached_calls", "observations_written")} for k, v in data["experiments"].items()},
                      "history_recovered": data["random_redraw_history"].get("recovered_n"), "original_summary_unchanged": data["original_summary_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
