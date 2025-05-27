import os
import sys
from types import SimpleNamespace
import json
import builtins

# Ensure the repository root is on sys.path so that `codes` is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pytest

# Since pdf_to_json_simple imports modules that may not be available
# we create dummy modules if needed

modules_to_mock = {}
for name in ["fitz", "pdf2image", "pytesseract"]:
    if name not in sys.modules:
        modules_to_mock[name] = SimpleNamespace()
        sys.modules[name] = modules_to_mock[name]

# Provide minimal APIs for mocked modules
if hasattr(sys.modules["pdf2image"], "convert_from_path") is False:
    sys.modules["pdf2image"].convert_from_path = lambda *a, **k: ["img"]

if hasattr(sys.modules["pytesseract"], "image_to_string") is False:
    sys.modules["pytesseract"].image_to_string = lambda img: "ocr"

import codes.pdf_to_json_simple as pdf_simple

# restore modules after import
def teardown_module(module):
    for name, obj in modules_to_mock.items():
        sys.modules.pop(name, None)


def test_extract_page_text_direct():
    page = SimpleNamespace(get_text=lambda: "hello", parent=SimpleNamespace(name="a.pdf"), number=0)
    assert pdf_simple.extract_page_text(page) == "hello"


def test_extract_page_text_ocr(monkeypatch):
    page = SimpleNamespace(get_text=lambda: "", parent=SimpleNamespace(name="b.pdf"), number=0)
    called = {}

    def fake_convert(pdf_path, dpi=300, first_page=None, last_page=None):
        called["convert"] = True
        return ["img"]

    def fake_ocr(img):
        called["ocr"] = True
        return "scanned"

    monkeypatch.setattr(pdf_simple, "convert_from_path", fake_convert)
    monkeypatch.setattr(pdf_simple, "pytesseract", SimpleNamespace(image_to_string=fake_ocr))

    text = pdf_simple.extract_page_text(page)
    assert text == "scanned"
    assert called.get("convert") and called.get("ocr")


def test_extract_tables_no_camelot(monkeypatch):
    monkeypatch.setattr(pdf_simple, "camelot", None)
    assert pdf_simple.extract_tables("file.pdf", 1) == []


def test_extract_tables_with_camelot(monkeypatch):
    class DummyTable:
        def __init__(self, value):
            self.df = SimpleNamespace(to_dict=lambda: {"table": value})
    class DummyCamelot:
        def read_pdf(self, pdf_path, pages):
            assert pdf_path == "file.pdf"
            assert pages == "2"
            return [DummyTable(1), DummyTable(2)]
    monkeypatch.setattr(pdf_simple, "camelot", DummyCamelot())
    tables = pdf_simple.extract_tables("file.pdf", 2)
    assert tables == [{"table": 1}, {"table": 2}]


def test_extract_pages(monkeypatch):
    class DummyPage:
        def __init__(self, num):
            self.number = num
            self.parent = SimpleNamespace(name="c.pdf")
        def get_text(self):
            return f"text{self.number}"
    class DummyDoc:
        def __iter__(self):
            return iter([DummyPage(0), DummyPage(1)])
    monkeypatch.setattr(pdf_simple, "fitz", SimpleNamespace(open=lambda x: DummyDoc()))
    monkeypatch.setattr(pdf_simple, "extract_tables", lambda path, page_number: [page_number])
    pages = pdf_simple.extract_pages("c.pdf")
    assert pages == [
        {"page_number": 1, "text": "text0", "tables": [1]},
        {"page_number": 2, "text": "text1", "tables": [2]},
    ]


def test_build_json(tmp_path, monkeypatch):
    sample_pages = [{"page_number": 1, "text": "a", "tables": []}]
    monkeypatch.setattr(pdf_simple, "extract_pages", lambda path: sample_pages)
    out_json = tmp_path / "out.json"
    pdf_simple.build_json("x.pdf", out_json)
    data = json.loads(out_json.read_text())
    assert data == {"pages": sample_pages}
