from types import SimpleNamespace

from src import ocr
from src.m1_chunking import load_documents


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="Văn bản OCR tiếng Việt")


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_ocr_image_sends_base64_vision_input(monkeypatch):
    monkeypatch.setattr(ocr, "OPENAI_API_KEY", "test-key")
    client = FakeClient()
    text = ocr.ocr_image(b"not-a-real-image", "scan.pdf", 1, client=client)
    assert text == "Văn bản OCR tiếng Việt"
    payload = client.responses.calls[0]["input"][0]["content"]
    assert payload[1]["type"] == "input_image"
    assert payload[1]["image_url"].startswith("data:image/png;base64,")


def test_load_documents_ocr_is_opt_in(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "plain.md").write_text("Nội dung markdown", encoding="utf-8")
    # Avoid parsing a fake PDF; OCR behavior is tested independently above.
    (data_dir / "scan.pdf").write_bytes(b"%PDF-invalid")
    monkeypatch.setattr("src.m1_chunking._extract_pdf_text", lambda _: "")
    docs = load_documents(str(data_dir), enable_ocr=False)
    assert [doc["metadata"]["source"] for doc in docs] == ["plain.md"]
