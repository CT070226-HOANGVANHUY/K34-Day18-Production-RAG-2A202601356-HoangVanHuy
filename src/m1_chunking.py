from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD, OCR_ENABLED, SKIP_PDF)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR, enable_ocr: bool | None = None,
                   skip_pdf: bool | None = None) -> list[dict]:
    """Load markdown và PDF từ data/.

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf; nếu ``enable_ocr`` bật thì OCR các trang
      scan bằng Vision API và lưu cache theo từng trang.
    - Bỏ qua toàn bộ PDF bằng ``skip_pdf=True`` hoặc ``SKIP_PDF=true``.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    if skip_pdf if skip_pdf is not None else SKIP_PDF:
        pdfs = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
        if pdfs:
            print(f"  [skip] Bo qua {len(pdfs)} file PDF (SKIP_PDF=true).")
        return docs

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        elif enable_ocr if enable_ocr is not None else OCR_ENABLED:
            from src.ocr import ocr_pdf
            try:
                ocr_text = ocr_pdf(fp)
            except RuntimeError as exc:
                print(f"  ⚠️  {exc}")
                ocr_text = ""
            if ocr_text:
                docs.append({
                    "text": ocr_text,
                    "metadata": {"source": os.path.basename(fp), "ocr": True},
                })
            else:
                print(f"  ⚠️  OCR không tạo được text cho {os.path.basename(fp)}.")
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (bật ENABLE_OCR=true để OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = dict(metadata or {})
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**metadata, "strategy": "semantic", "chunk_index": 0})]

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        similarities = [float(np.dot(embeddings[i - 1], embeddings[i])) for i in range(1, len(sentences))]
    except Exception as exc:
        # Chunking remains usable if a model cannot be downloaded in an offline run.
        print(f"  ⚠️  Semantic model unavailable, using sentence fallback: {exc}")
        similarities = [1.0] * (len(sentences) - 1)

    groups = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(" ".join(group), {**metadata, "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = dict(metadata or {})
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    def split_by_size(value: str, size: int) -> list[str]:
        if len(value) <= size:
            return [value]
        words = value.split()
        pieces, current = [], []
        current_len = 0
        for word in words:
            extra = len(word) + (1 if current else 0)
            if current and current_len + extra > size:
                pieces.append(" ".join(current))
                current, current_len = [], 0
            current.append(word)
            current_len += len(word) + (1 if current_len else 0)
        if current:
            pieces.append(" ".join(current))
        return pieces or [value[:size]]

    parent_texts, current, current_len = [], [], 0
    for paragraph in paragraphs:
        for piece in split_by_size(paragraph, parent_size):
            extra = len(piece) + (2 if current else 0)
            if current and current_len + extra > parent_size:
                parent_texts.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(piece)
            current_len += len(piece) + (2 if current_len else 0)
    if current:
        parent_texts.append("\n\n".join(current))

    parents, children = [], []
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parents.append(Chunk(
            parent_text,
            {**metadata, "chunk_type": "parent", "parent_id": parent_id,
             "chunk_index": parent_index},
        ))
        for child_index, child_text in enumerate(split_by_size(parent_text, child_size)):
            children.append(Chunk(
                child_text,
                {**metadata, "chunk_type": "child", "parent_id": parent_id,
                 "chunk_index": child_index},
                parent_id=parent_id,
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = dict(metadata or {})
    lines = text.splitlines()
    chunks, current_lines, current_header = [], [], ""

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if not content:
            return
        section = current_header.lstrip("#").strip() if current_header else ""
        chunks.append(Chunk(
            content,
            {**metadata, "section": section, "header": current_header,
             "strategy": "structure", "chunk_index": len(chunks)},
        ))

    for line in lines:
        if re.match(r"^#{1,3}\s+.+$", line.strip()):
            flush()
            current_lines = [line.strip()]
            current_header = line.strip()
        else:
            current_lines.append(line)
    flush()

    if not chunks and text.strip():
        return [Chunk(text.strip(), {**metadata, "section": "", "strategy": "structure", "chunk_index": 0})]
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
