# Group Report — Lab 18: Production RAG

**Thành viên:** Hoàng Văn Huy
**Ngày:** 2026-08-18

## Thành viên và phân công

| Tên | Module | Hoàn thành | Tests pass |
|---|---|---:|---:|
| Hoàng Văn Huy | M1: Chunking | ✅ | 13/13 |
| Hoàng Văn Huy | M2: Hybrid Search | ✅ | 5/5 |
| Hoàng Văn Huy | M3: Reranking | ✅ | 5/5 |
| Hoàng Văn Huy | M4: Evaluation | ✅ | 4/4 |
| Hoàng Văn Huy | M5: Enrichment | ✅ | 10/10 |
| Hoàng Văn Huy | OCR và robustness tests | ✅ | 12/12 |

Tổng kiểm tra: **49/49 tests passed** khi chạy trong `.venv` với
`SKIP_PDF=true` và tắt API key cho các test để dùng local fallback.

## Cấu hình và dữ liệu chạy

- Qdrant chạy local qua Docker Compose.
- Corpus: 25 tài liệu Markdown.
- PDF: bỏ qua cả 3 file bằng `SKIP_PDF=true`, không trích text và không OCR.
- Basic baseline: 56 paragraph chunks.
- Production chunking: 25 parent chunks và 97 child chunks trước enrichment.
- Evaluation: 20 câu hỏi.

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.895833 | 0.882917 | -0.012916 |
| Answer Relevancy | 0.756488 | 0.871456 | +0.114968 |
| Context Precision | 0.925000 | 0.954667 | +0.029667 |
| Context Recall | 0.925000 | 0.783333 | -0.141667 |

Production đạt ít nhất 3/4 metrics trên 0.70, với answer relevancy và context
precision cải thiện so với baseline. Context recall là điểm yếu chính.

## Key findings

1. **Biggest improvement:** Answer relevancy tăng `+0.114968`; enrichment,
   hybrid BM25+dense và CrossEncoder giúp context bám câu hỏi hơn.
2. **Biggest challenge:** Multi-hop và numeric queries làm giảm context
   recall; top-k reranking chưa giữ đủ bằng chứng từ nhiều nguồn.
3. **Surprise finding:** Precision tăng nhưng recall giảm, cho thấy context
   ít nhiễu hơn nhưng quá hẹp cho các câu hỏi cần ghép nhiều policy.

## Presentation notes

1. Trình bày bảng so sánh 4 metrics giữa baseline và production.
2. Biggest win là hybrid retrieval + reranking + enrichment, thể hiện rõ ở
   answer relevancy và context precision.
3. Case study là Senior 9 năm: cần ghép phép v2024 với bảng lương P3–P4.
4. Nếu có thêm một giờ: query decomposition, version-aware metadata filter,
   numeric-aware retrieval và sau đó mới bổ sung OCR cho PDF.
