# Kết quả đánh giá sơ bộ bởi một người

Đã nhận và kiểm tra **400/400 trường hợp** từ A1, 400 mã duy nhất, đầy đủ năm trường nhãn/giải thích và đúng tập giá trị cho phép. Đây là kết quả mô tả riêng của A1; sau đó A2 đã độc lập chấm cùng 400 case và hai bất đồng được adjudicate. Không chỉnh sửa nhãn đã nộp. Dữ liệu nguồn, hash và phương pháp nằm trong [summary.json](artifact/validation/human_analysis/summary.json) và [TWO_RATER_EVALUATION.md](TWO_RATER_EVALUATION.md).

Một người chấm đã cung cấp bộ nhãn. Theo `annotator_notes.md`, A1 có 10 năm kinh nghiệm nghiên cứu/đọc Python, chấm ngày 11/11/2026, chưa xem manuscript hoặc nhãn mô hình và chỉ dùng bộ mã được cung cấp. Đây là khai báo của người chấm, không phải thông tin được agent xác minh độc lập. Không có người thứ hai hay phân xử giữa những người chấm. Thiết kế đã đổi sau kế hoạch từ hai người sang một người.

## Kết quả

| Chỉ số | Giá trị |
| --- | ---: |
| Nhãn người: supported / contradicted / unresolved | 160 / 137 / 103 |
| Đồng thuận ba nhãn, không trọng số | 92.25% (369/400) |
| Kappa ba nhãn, không trọng số | 0.8837 |
| Đồng thuận ba nhãn, có trọng số | 99.27% |
| Kappa ba nhãn, có trọng số | 0.9164 |
| Nhận định bị chấm gộp ý | 28/400 |
| Cần thêm ngữ cảnh | 103/400 |

Ma trận **số trường hợp thực chấm**, hàng là mô hình, cột là người; không phải ước lượng số trường hợp của toàn bộ quần thể:

| Mô hình / Người | supported | contradicted | unresolved |
| --- | ---: | ---: | ---: |
| supported | 132 | 1 | 0 |
| contradicted | 0 | 134 | 0 |
| unresolved | 28 | 2 | 103 |

Có 31 bất đồng: 28 trường hợp mô hình unresolved nhưng người supported, hai trường hợp unresolved → contradicted và một trường hợp supported → contradicted. Xem [disagreements.csv](artifact/validation/human_analysis/disagreements.csv). Không đổi các nhãn gốc để xóa bất đồng. Các nhận định gộp ý hoặc cần ngữ cảnh nằm trong [review_flags.csv](artifact/validation/human_analysis/review_flags.csv); đây là danh sách cần lưu ý, không phải yêu cầu ép nhãn mới.

Trọng số nghịch đảo xác suất lấy mẫu đưa sáu tầng về tập **14.511 phán xét đã ghi**. Tập này có nhiều supported; vì vậy đồng thuận có trọng số cao hơn đồng thuận thô. Trọng số không biến 400 trường hợp thành 14.511 trường hợp do người trực tiếp chấm. Kết quả theo giai đoạn, nhánh và cohort nằm trong JSON; các nhóm nhỏ chỉ mang tính mô tả.

## Bất định và giới hạn

- Khoảng mô tả 95% từ 2.000 lần nhân trọng số ngẫu nhiên theo cụm hàm, hiệu chỉnh lại tổng từng tầng: đồng thuận có trọng số [98.53%, 99.61%], kappa [0.840, 0.955]. Đây là xấp xỉ theo cụm, không phải khoảng chính xác của thiết kế khảo sát hoặc sai số người chấm.
- JSON cũng báo phương sai thiết kế lấy mẫu ngẫu nhiên không hoàn lại trong từng tầng, với nhãn người được coi là giá trị cố định. Khoảng chuẩn xấp xỉ có thể kém khi sai khác thưa. Hai cách đánh giá bất định có giả định khác nhau; cả hai đều không đo được thiên lệch của một người chấm.
- Với 297 trường hợp người đưa ra kết luận supported/contradicted, sensitivity phát hiện contradiction có trọng số so với người này là 89.71%. Nhãn người unresolved bị loại khỏi riêng phép tính nhị phân này, vẫn giữ trong ma trận ba nhãn. Mô hình unresolved được tính là không phát hiện contradiction trong phép tính nhị phân. Specificity/precision quan sát bằng 100% chỉ phản ánh không thấy loại sai khác tương ứng trong mẫu, không chứng minh mô hình không mắc lỗi.
- Các chỉ số dùng **một người làm tham chiếu**, không phải độ chính xác đối với chân lý ngữ nghĩa đã phân xử. Không có đồng thuận giữa người với người. Bằng chứng và 28 nhãn gộp ý vẫn cần được diễn giải thận trọng; chưa kiểm tra đầy đủ các ý bị bỏ sót khi tách nhận định.
- Bộ mẫu chỉ chứa các phán xét đã được lưu; lần theo dõi sau đó phụ thuộc quy tắc dừng ở contradiction đầu tiên. Không có nhãn người cho toàn bộ phiên bản lịch sử, thời điểm failure, resurrection hay mọi nhận định bị bỏ sót. Không dùng audit này để tự động hiệu chỉnh hazard/survival hoặc gọi đường survival là đã xác thực ngữ nghĩa.

Các số liệu survival lịch sử và bốn thí nghiệm tự động được giữ nguyên. Artifact sửa đổi vẫn là gói cục bộ, chưa có URL phát hành ẩn danh đã xác minh.
