# Adversarial Test Plan — Production RAG

## Mục tiêu

Kiểm tra những lỗi mà happy-path retrieval thường bỏ sót: lấy nhầm phiên bản,
đảo nghĩa phủ định, bỏ mất số/ngưỡng, thiếu một vế của câu hỏi multi-hop và
trả lời bịa khi corpus không có bằng chứng.

## Bộ testcase

Dataset đầy đủ nằm tại [adversarial_test_set.json](../adversarial_test_set.json)
và được chia thành các nhóm:

| Nhóm | Ví dụ | Tiêu chí đạt |
|---|---|---|
| `multi_hop` | Senior 9 năm: phép + lương | Context có evidence cho mọi vế |
| `negation` | Thử việc có PVI không? | Giữ đúng “KHÔNG/chưa/không được” |
| `version_conflict` | 90 hay 120 ngày? | Ưu tiên v2.0/v2024; nêu bản cũ nếu được hỏi |
| `numeric_threshold` | Thiết bị 55 triệu | Giữ số tiền và ngưỡng trên 50 triệu |
| `numeric_range` | Nghỉ không lương 20 ngày | Chọn đúng khoảng 16–30 ngày |
| `numeric_calculation` | Tạm ứng 15 triệu, trễ 5 ngày | Không nhầm số tiền với số ngày |
| `ambiguity` | Thâm niên được cộng phép | Xử lý xung đột v2023/v2024 |
| `out_of_scope` | Giá cổ phiếu hôm nay | Abstain, không bịa dữ liệu |
| `ocr_gap` | Nghị định trong PDF scan | Nói rõ chưa có text layer/cần OCR |
| `contradiction` | Chính sách 90 hay 120? | Trình bày trạng thái current/superseded |

## Giải pháp đã đưa vào code

1. `src/robustness.py` tạo `QueryPlan`, phát hiện multi-hop, phủ định, version
   và số liệu; query multi-hop được tách thành nhiều biến thể retrieval.
2. `HybridSearch.search()` chạy BM25 + Dense cho từng biến thể rồi hợp nhất bằng
   RRF, giúp tăng recall mà không bỏ mất query gốc.
3. `prioritize_results()` chấm thêm evidence signals: từ phủ định, version hiện
   hành, số/ngưỡng và độ phủ từ khóa.
4. Pipeline dùng 6 context cho câu hỏi multi-hop thay vì chỉ 3 context; prompt
   yêu cầu đủ evidence cho từng ý và phải nói rõ khi thiếu bằng chứng.
5. Các PDF scan vẫn được đánh dấu rõ là cần OCR; không coi nội dung rỗng là
   bằng chứng hợp lệ.

## Chạy kiểm thử

```powershell
.\\.venv\\Scripts\\Activate.ps1
python -m pytest tests/test_adversarial.py -q
python -m pytest tests/ -q
```

`test_adversarial.py` là regression test model-free, chạy nhanh và không gọi
OpenAI. Muốn đánh giá retrieval/LLM thật trên 15 case, dùng dataset này làm đầu
vào cho evaluator sau khi build pipeline; nên ghi riêng các metric theo nhóm,
đặc biệt `context_recall` cho multi-hop và `faithfulness` cho negation/version.

## Tiêu chí production đề xuất

- Multi-hop: context recall ≥ 0.90.
- Negation/version: không có câu trả lời đảo nghĩa hoặc dùng policy superseded.
- Numeric: 100% testcase giữ đúng số và khoảng/ngưỡng quan trọng.
- Out-of-scope/OCR gap: có abstention rõ ràng, không hallucinate.
- Mỗi failure phải lưu query, retrieved contexts, answer, metric thấp nhất và
  suggested fix để có thể regression ở lần thay model tiếp theo.
