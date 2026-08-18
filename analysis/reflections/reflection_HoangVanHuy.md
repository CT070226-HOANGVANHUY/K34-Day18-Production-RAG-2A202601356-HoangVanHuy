# Reflection — Hoàng Văn Huy

## 1. Mapping bài giảng vào code

| Lecture concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Encode câu bằng `all-MiniLM-L6-v2`, tách nhóm khi cosine similarity thấp. |
| Parent-child chunking | M1 | `chunk_hierarchical()` | Retrieve child nhỏ để tăng precision và giữ `parent_id` cho traceability. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 bắt exact terms tiếng Việt, dense bắt semantic intent; RRF hợp nhất ranking. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Chấm trực tiếp query-document trên tập ứng viên nhỏ để tăng precision. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Đo faithfulness, answer relevancy, context precision và context recall; có local fallback. |
| Contextual enrichment | M5 | `contextual_prepend()` và `_enrich_single_call()` | Bổ sung ngữ cảnh nguồn, summary, câu hỏi giả thuyết và metadata trước embedding. |

## 2. Khó khăn và cách giải quyết

1. **Qdrant chưa chạy:** pipeline gặp `httpx.ConnectError: [WinError 10061]` khi
   kết nối `localhost:6333`. Khởi động bằng `docker compose up -d` rồi chạy lại.
2. **Console Windows không in được emoji:** PowerShell dùng CP1252 và gây
   `UnicodeEncodeError`. Chạy Python với `-X utf8`; tùy chọn skip PDF cũng dùng
   thông báo ASCII để loader không dừng trên console này.
3. **PDF scan và chi phí OCR:** hai PDF scan không có text layer; ngoài ra có
   một PDF text layer. Để có vòng chạy nhanh, `SKIP_PDF=true` bỏ qua cả 3 PDF
   trước khi gọi `pypdf` hoặc OCR.
4. **Test environment:** Python hệ thống thiếu `pypdf` và `rank_bm25`, gây 3
   lỗi import. Chạy bằng `.venv` với `SKIP_PDF=true` và API key rỗng cho local
   fallback; kết quả cuối cùng là **49/49 tests passed**.
5. **RAGAS và API key:** evaluation thật phụ thuộc LLM và có thể chậm. Code
   bọc lời gọi trong `try/except`; khi phát triển offline, heuristic fallback
   giúp test M4 vẫn chạy được.

## 3. Kết quả và bài học

Ở lần chạy không đọc PDF, production đạt faithfulness `0.882917`, answer
relevancy `0.871456`, context precision `0.954667` và context recall `0.783333`.
So với baseline, precision và relevancy tăng nhưng recall giảm. Điều này cho
thấy tối ưu retrieval không chỉ là lấy context ít nhiễu; với câu hỏi multi-hop,
hệ thống phải bảo toàn đủ các nguồn liên quan.

## 4. Action plan cho project

### Hiện tại

- RAG đã có hierarchical chunking, hybrid retrieval, reranking, enrichment và
  evaluation.
- Known issues: multi-hop recall thấp hơn baseline, policy version có thể bị
  trộn, numeric/negation queries cần kiểm tra riêng, PDF đang được skip.

### Plan áp dụng

1. [x] Dùng hierarchical child 256 / parent 2048 cho policy dài.
2. [x] Dùng hybrid BM25 + BGE-M3 vì corpus có cả exact terms và semantic query.
3. [x] Dùng `BAAI/bge-reranker-v2-m3` cho top candidates.
4. [x] Đánh giá bằng RAGAS 4 metrics và local fallback khi offline.
5. [x] Dùng combined enrichment để giảm số API call mỗi chunk.
6. [ ] Thêm query decomposition và evidence checklist cho multi-hop.
7. [ ] Thêm version-aware metadata filter và numeric-aware retrieval.
8. [ ] Chỉ bật OCR sau khi cần khai thác PDF và đã kiểm soát chi phí.

### Timeline

- **Tuần 1:** triển khai decomposition, version filter và bộ test numeric/negation.
- **Tuần 2:** benchmark top-k, batching và latency trên CPU/GPU.
- **Tuần 3:** bật OCR có cache theo trang, đánh giá lại với corpus đầy đủ.
