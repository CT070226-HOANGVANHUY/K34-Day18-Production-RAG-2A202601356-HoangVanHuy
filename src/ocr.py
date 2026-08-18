from __future__ import annotations

"""OCR support for image-only PDFs using OpenAI vision input.

The implementation is opt-in because OCR makes one API call per scanned page.
Results are checkpointed under ``data/.ocr_cache`` so an interrupted run can
resume without paying for completed pages again.
"""

import base64
import io
import json
import mimetypes
import os
from pathlib import Path

from config import OCR_CACHE_DIR, OCR_MODEL, OPENAI_API_KEY


def _image_data_url(data: bytes, filename: str = "page.png") -> str:
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _as_png(data: bytes) -> tuple[bytes, str]:
    """Normalize embedded PDF images to a vision-friendly PNG."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG", optimize=True)
            return output.getvalue(), "page.png"
    except Exception:
        return data, "page.png"


def _response_text(response) -> str:
    value = getattr(response, "output_text", None)
    if value:
        return value.strip()
    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def ocr_image(image_bytes: bytes, source: str, page_number: int, client=None) -> str:
    """Read one page image and return plain Vietnamese text."""
    if not OPENAI_API_KEY:
        return ""
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0, max_retries=0)

    normalized, filename = _as_png(image_bytes)
    prompt = (
        "Bạn là OCR cho tài liệu tiếng Việt. Hãy chép lại chính xác toàn bộ chữ, "
        "số, bảng và dấu câu nhìn thấy trong ảnh. Không giải thích, không tóm tắt, "
        "không tự bổ sung nội dung. "
        f"Nguồn: {source}, trang {page_number}."
    )
    data_url = _image_data_url(normalized, filename)

    # Chat Completions vision is kept as the primary path for compatibility
    # with the OpenAI SDK version used by this lab. The Responses path remains
    # available for injected clients and newer SDKs.
    if hasattr(client, "chat"):
        response = client.chat.completions.create(
            model=OCR_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            }],
            max_tokens=4096,
        )
        return (response.choices[0].message.content or "").strip()

    response = client.responses.create(
        model=OCR_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url, "detail": "high"},
            ],
        }],
    )
    return _response_text(response)


def ocr_pdf(path: str, cache_dir: str = OCR_CACHE_DIR, client=None) -> str:
    """OCR all image pages in a PDF, resuming from a JSON page cache."""
    from pypdf import PdfReader

    if not OPENAI_API_KEY:
        return ""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir) / f"{Path(path).stem}.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        cache = {}

    reader = PdfReader(path)
    page_texts = []
    for page_number, page in enumerate(reader.pages, start=1):
        key = str(page_number)
        if key not in cache or not str(cache[key]).strip():
            images = list(page.images)
            if images:
                fragments = []
                for image in images:
                    try:
                        fragments.append(ocr_image(
                            image.data, os.path.basename(path), page_number, client=client
                        ))
                    except Exception as exc:
                        status_code = getattr(exc, "status_code", None)
                        if status_code == 401 or "invalid_api_key" in str(exc) or "Incorrect API key" in str(exc):
                            raise RuntimeError(
                                "OPENAI_API_KEY không hợp lệ hoặc đã bị thu hồi. "
                                "Hãy cập nhật key trong .env/PowerShell rồi chạy lại."
                            ) from exc
                        print(f"  ⚠️  OCR lỗi {os.path.basename(path)} trang {page_number}: {exc}")
                page_text = "\n".join(fragment for fragment in fragments if fragment).strip()
                if page_text:
                    cache[key] = page_text
                    cache_path.write_text(
                        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
            elif key not in cache:
                cache[key] = ""
        if str(cache.get(key, "")).strip():
            page_texts.append(f"[Trang {page_number}]\n{cache[key].strip()}")
    return "\n\n".join(page_texts).strip()
