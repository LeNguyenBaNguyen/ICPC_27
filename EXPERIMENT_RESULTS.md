# Kết quả bốn thí nghiệm tự động bổ sung

Cập nhật: 2026-11-11T19:02:38.212703+07:00. Cả bốn thí nghiệm tự động bổ sung đã chạy đủ kế hoạch cố định. Tổng cộng **1,800 phản hồi mới** được lưu riêng. Kiểm tra tự động của một agent riêng xác nhận toàn bộ 4.060 tệp kết quả và cache gốc giữ nguyên. Đây là hoàn thành thực thi; một số nhãn mô hình chưa đọc được vẫn được giữ nguyên.

| Điều kiện | Thực thi đã hoàn thành | Kết quả chính |
| --- | --- | --- |
| Mẫu ngẫu nhiên mới, seed 107 | 80/80 hàm; 308 phản hồi | 886 nhận định được hỗ trợ tại phiên bản gốc; S(3)=0.9363; không báo S(6) vì chỉ còn 19 nhận định trong 2 hàm |
| Judge khác họ bổ sung, Llama 3.1 8B | 60/60 trường hợp, 57 prompt | Đồng thuận thô=0.550; có trọng số=0.973; kappa nhị phân=0.719 |
| Được mô hình chấm supported ở phiên bản sau | 649/649 lượt: 163 đối chứng + 486 follow-up | 121/232 nhận định có lần sau được chấm supported; 111/219 trong nhóm vẫn bị judge mới chấm contradicted tại phiên bản sự kiện |
| Sinh lại không thấy docstring của hàm mục tiêu | 65/65 hàm, 65 đối chứng sinh mới; 786 phản hồi | 760/772 nhận định được hỗ trợ tại phiên bản gốc; S(6) mới=0.8643, bản lưu đối chiếu=0.8271 |

## Giới hạn cần giữ khi diễn giải

- **Redraw:** nguồn là tập hàm đủ điều kiện tại mốc gốc đã lưu, không phải điều tra mới toàn bộ hệ sinh thái. S(365 ngày)=0.9746 khi censor removal, hoặc 0.8894 khi tính removal là sự kiện. Có 11 hàm bị phát hiện mâu thuẫn và 7 hàm bị removal trong 365 ngày; CIF ở cấp nhận định không thể suy ra trực tiếp từ hai số hàm này. Một nhãn ngoài schema được giữ là chưa đọc được; không đổi thành contradicted.
- **Cross-family:** đồng thuận có trọng số cao do tập quan sát chứa nhiều supported; đồng thuận thô chỉ 55%. Khoảng bootstrap mô tả cho kappa là [0.484, 0.882]. Hai nhãn phụ bị bỏ sót ngoài 60 trường hợp được chấm vẫn được báo lỗi; 60 trường hợp đích đều đọc được. Đây là đồng thuận giữa mô hình, không phải độ đúng ngữ nghĩa hoặc cải thiện trực tiếp so với Claude cũ. Bootstrap chỉ xấp xỉ thiết kế lấy mẫu.
- **Resurrection:** đây là supported trở lại theo judge, không phải xác nhận ngữ nghĩa. Có 13 nhận định mà judge mới không còn đồng ý với contradiction gốc ngay tại phiên bản sự kiện. Cách nhóm nhận định ở follow-up khác, runtime lịch sử không có digest và thời lượng theo dõi không đều. Có 33 verdict không đọc được, trong đó 30 là nhãn follow-up được phân tích; giữ là thiếu nhãn, không gán supported/contradicted tùy ý. Kịch bản riêng cho nhãn thiếu cho giới hạn 121–129 nhận định có supported sau đó; đây không phải khoảng tin cậy hoặc xác minh ngữ nghĩa. Chi tiết: `resurrection/missing_label_sensitivity.json`.
- **Docstring-free:** tập nhận định, cách nhóm và runtime khác bản lưu, nên chênh lệch S(6) không phải hiệu ứng nhân quả đã kiểm soát. Đối chứng sinh mới chỉ dùng so sánh mức nhắc tên identifier; không có survival đối chứng mới. Chênh lệch proxy này trung bình so với đối chứng mới là 0.00714; không đo tính hữu ích hoặc lượng thông tin cần cho nhiệm vụ. Không có verdict chưa đọc được trong điều kiện này.

## Kiểm tra và sự cố đã xử lý

Một agent riêng đã kiểm tra tự động dữ liệu thô, vị trí claim, trọng số, first-event survival, coverage và fingerprint; xem `artifact/experiments/final_agent_audit.json`. Đây không phải đánh giá của người. Kế hoạch, runtime, response và hash nằm trong từng thư mục `artifact/experiments/`.

Lỗi OneDrive tạm khóa `os.replace` khiến lượt resurrection đầu dừng giữa chừng. Đã thêm retry ghi nguyên tử và kiểm thử; chạy tiếp dùng lại một response đã lưu, hoàn thành phần thiếu. `campaign_initial_attempt.json` giữ lịch sử lỗi trước resume; `campaign_run.json` được đối chiếu lại với trạng thái cuối, và **`final_execution.json` là bản tổng kết có kiểm tra hash**, không xóa dấu vết lỗi cũ.

Mô hình mới chạy cục bộ trên Ollama 0.40.16, T=0, seed 7, context 8192. Model digest thực tế được lưu; digest/quantization của mô hình lịch sử không có trong archive. Không thay dữ liệu gốc, không gộp phản hồi mới thành kết quả lịch sử.

## Đánh giá người đã nhận; phát hành artifact còn chờ

- **Đánh giá hai người đã hoàn tất:** A1 và A2 độc lập chấm cùng 400 case blinded vào ngày 11/11/2026; A2 không xem A1, nhãn model, manuscript hoặc kết quả. Raw/weighted agreement là 99,50%/99,980% (κ=.992/.998); hai bất đồng được adjudicator xử lý từ blind packet riêng ngày 11/11/2026. Provenance và quan hệ được tự khai báo; A2 disclosed a social acquaintance of an author. Xem [kết quả và giới hạn](HUMAN_EVALUATION_RESULTS.md). Các số liệu survival không được tự động hiệu chỉnh từ bộ nhãn này.
- **Artifact đã phát hành:** [trang reviewer](https://claim-survival-artifact.dimpledshrimp2.chatgpt.site); [tải ZIP](https://claim-survival-artifact.dimpledshrimp2.chatgpt.site/claim-survival-artifact-v3.zip). Đã tải lại không dùng đăng nhập, cookie hoặc Authorization header; SHA-256 khớp gói đã kiểm tra. URL thật đã được đưa vào manuscript. Xem `ARTIFACT_PUBLICATION.md` và `artifact/publication.json`.
- **Dữ liệu lịch sử:** vẫn thiếu tệp nhãn audit docstring gốc; không phục hồi diễn giải từ các số tổng hợp khi chưa có bằng chứng cấp nhận định.

Bốn thí nghiệm tự động bổ sung đã hoàn thành thực thi trong phạm vi kế hoạch cố định. Kết quả hiện tại mô tả **mâu thuẫn do mô hình phát hiện**, chưa xác nhận thời điểm sai về ngữ nghĩa. So sánh với detector H3 đã công bố, đo lợi ích cho lập trình viên và so sánh chính sách cập nhật là các nghiên cứu mở rộng chưa thực hiện.

## Reviewer follow-up (2026-11-12)

- Mutation recall fixture: 100/100 known mutations detected; Wilson 95% CI 96.3--100%; 0/100 unchanged controls false-positive. This is fixture recall, not semantic recall on the historical corpus.
- Full cross-family replay: 129/129 functions, 1,290 anchor-supported claims, and 766 grouped calls. The initial grouped pass omitted 274 verdict lines; a frozen one-statement repair pass recovered 274/274, leaving 0 unparsed verdicts in `repaired_analysis.json`. Llama S(6)=.6126 [.5727,.6515]; Qwen--Llama agreement 87.19%, kappa=.2415 (1,725 labels). These are model-conditional stability results, not semantic validity or human accuracy.

Detailed outputs are in `artifact/experiments/mutation_recall/` and `artifact/experiments/full_cross_family/`; historical result files remain unchanged.

## Blinded disagreement and subtle-mutation audit (2026-11-12)

One independent human rater, blinded to model labels and manuscript outcomes, reviewed 170 cross-model disagreement cases. The rater supported the Qwen label in 108/110 Qwen-supported cases (98.2%); among 60 Qwen-contradicted/Llama-supported cases, 16/60 (26.7%) were independently judged contradicted. Three cases were unresolved. The audit is directional evidence only: it is a single-rater sample and not a semantic gold standard.

The same rater reviewed 25 subtle synthetic mutations across five families (boundary comparison, condition ordering, exception type, default value, and callee/keyword argument). All 25 expected affected cases were detected. This is a targeted sanity check, not population recall. Full records and provenance are in `artifact/validation/cross_model_disagreements_audit.*`, `subtle_mutations_audit.*`, and `cross_model_audit_summary.json`.

The primary Qwen judge was then run on those same 25 subtle mutations with the frozen local prompt/runtime (temperature 0, seed 7). It returned 22 `contradicted`, 2 `supported`, and 1 `unresolved` verdicts; all 25 responses parsed. This is a mutation-suite judge sensitivity result, not a semantic recall estimate for repository histories. Raw responses are in `artifact/experiments/subtle_mutation_judge/observations.json`.
