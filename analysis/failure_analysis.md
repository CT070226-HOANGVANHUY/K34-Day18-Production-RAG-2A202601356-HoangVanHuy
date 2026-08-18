# Failure Analysis — Lab 18: Production RAG

## Phạm vi và kết quả

Báo cáo được chuẩn hóa theo lần chạy không đọc PDF: `SKIP_PDF=true`. Corpus
gồm 25 tài liệu Markdown; 3 file PDF được bỏ qua trước bước trích xuất. Bộ
đánh giá có 20 câu hỏi.

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.895833 | 0.882917 | -0.012916 |
| Answer Relevancy | 0.756488 | 0.871456 | +0.114968 |
| Context Precision | 0.925000 | 0.954667 | +0.029667 |
| Context Recall | 0.925000 | 0.783333 | -0.141667 |

Production cải thiện rõ rệt answer relevancy và context precision nhờ
enrichment, hybrid search và reranking. Context recall lại giảm, chủ yếu vì
top-k cuối cùng ưu tiên các child chunk ngắn và chưa có query decomposition
cho câu hỏi cần nhiều nguồn. Faithfulness giảm nhẹ, cho thấy vẫn cần kiểm
soát chặt câu trả lời khi context chưa đủ bằng chứng.

## Bottom-5 failures

### #1 — Tạm ứng 15 triệu, thanh toán sau 20 ngày

- **Expected:** Quá hạn 5 ngày; phí 2%/tháng trên 15.000.000 đồng là 300.000
  đồng/tháng, tương đương khoảng 50.000 đồng cho 5 ngày.
- **Observed:** Faithfulness `0.125`, average score `0.627875`.
- **Root cause:** Câu hỏi cần tính cả thời hạn, số tiền, tỷ lệ và số ngày quá
  hạn. Answer có nguy cơ thêm phép tính ngoài context được chọn.
- **Suggested fix:** Tách truy vấn theo các biến số; giữ nguyên amount và
  boundary trong query; chỉ cho phép generate khi context chứa đủ công thức.

### #2 — Nghỉ phép không lương 20 ngày

- **Expected:** Nghỉ 16–30 ngày cần CEO phê duyệt; trên 14 ngày nhân viên tự
  đóng phần bảo hiểm.
- **Observed:** Context precision `0.333333`, average score `0.699017`.
- **Root cause:** Context bị nhiễu giữa điều kiện phê duyệt và điều kiện bảo
  hiểm; câu hỏi nhiều điều kiện nhưng retrieval chưa tách ý.
- **Suggested fix:** Decompose thành approval và insurance sub-query, rerank
  riêng rồi kiểm tra đủ cả hai bằng chứng trước khi trả lời.

### #3 — Hoàn trả khóa học 25 triệu sau 8 tháng

- **Expected:** Cam kết tối thiểu 12 tháng sau khi hoàn thành; nghỉ sau 8
  tháng phải hoàn trả 100%, tức 25.000.000 đồng.
- **Observed:** Faithfulness `0.200000`, average score `0.752045`.
- **Root cause:** Cần ghép số tiền với thời hạn cam kết. Nếu chỉ lấy chunk về
  đào tạo hoặc chỉ lấy chunk về nghỉ việc, LLM dễ suy diễn phần còn thiếu.
- **Suggested fix:** Numeric-aware retrieval cho số tiền và số tháng; thêm
  validation yêu cầu context chứa cả cam kết lẫn tỷ lệ hoàn trả.

### #4 — Senior có 9 năm thâm niên

- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên = 18 ngày; lương Senior
  P3–P4 là 20–35 triệu đồng/tháng.
- **Observed:** Context recall `0.500000`, average score `0.784007`.
- **Root cause:** Đây là multi-hop query, cần đồng thời lấy chính sách phép
  v2024 và bảng lương. Một ranking top-k chưa bảo đảm giữ cả hai nguồn.
- **Suggested fix:** Decompose truy vấn, retrieve từng nguồn, deduplicate rồi
  hợp nhất context; ưu tiên policy hiện hành bằng metadata version.

### #5 — Chu kỳ đổi mật khẩu

- **Expected:** Chính sách v2.0 hiện hành yêu cầu đổi mỗi 120 ngày; v1.0 yêu
  cầu 90 ngày đã bị thay thế.
- **Observed:** Context recall `0.500000`, average score `0.830214`.
- **Root cause:** Corpus có hai phiên bản chính sách. Retrieval có thể lấy
  chunk cũ hoặc không đưa trạng thái superseded vào context.
- **Suggested fix:** Gắn metadata `version` và `superseded`, lọc/boost bản hiện
  hành, đồng thời đưa quy tắc version-aware vào prompt.

## Diagnostic tree

1. **Output có bám context không?** Nếu faithfulness thấp, kiểm tra câu trả
   lời có thêm số liệu hoặc kết luận không xuất hiện trong context.
2. **Context có đủ và đúng phiên bản không?** Nếu context recall thấp, kiểm tra
   missing chunk, tài liệu superseded và số lượng nguồn được giữ lại.
3. **Query có multi-hop, phủ định hoặc numeric không?** Nếu có, tách sub-query
   và bảo toàn số, ngưỡng, phủ định trong bước rewrite.
4. **Chọn fix:** prompt/abstention cho lỗi faithfulness; retrieval/reranking
   cho lỗi recall/precision; metadata filter cho lỗi version.

## Case study

Câu hỏi Senior 9 năm là failure tiêu biểu. Query cần hai nguồn độc lập: phép
năm v2024 và bảng lương Senior. Nếu chỉ retrieve một top-k chung, reranker có
thể giữ chunk phép nhưng loại chunk lương, hoặc ngược lại. Thiết kế phù hợp là
decompose thành hai truy vấn, kiểm tra đủ evidence cho từng ý, sau đó mới sinh
câu trả lời. Nếu thiếu một nhánh, pipeline nên nói rõ phần chưa tìm thấy thay
vì suy diễn.
