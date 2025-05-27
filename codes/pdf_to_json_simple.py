"""Simple PDF to JSON conversion script.

This script provides a lightweight pipeline for extracting
text and tables from a PDF without requiring GROBID or
other complex services. It relies on PyMuPDF for direct
text extraction and falls back to Tesseract OCR when
no text is detected on a page. Detected tables are
extracted with camelot if available.
"""

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

try:
    import camelot  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    camelot = None


def extract_page_text(page):
    """Return text from a PyMuPDF page, using OCR if needed."""
    text = page.get_text()
    if text.strip():
        return text
    # Fallback to OCR
    images = convert_from_path(
        page.parent.name,
        dpi=300,
        first_page=page.number + 1,
        last_page=page.number + 1,
    )
    return pytesseract.image_to_string(images[0])


def extract_tables(pdf_path: str, page_number: int):
    """Extract tables from the specified page using camelot if available."""
    if camelot is None:
        return []
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_number))
        return [t.df.to_dict() for t in tables]
    except Exception:
        return []


def extract_pages(pdf_path: str):
    """Process all pages in the PDF and collect text and tables."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_num = page.number + 1
        text = extract_page_text(page)
        tables = extract_tables(pdf_path, page_num)
        pages.append({"page_number": page_num, "text": text, "tables": tables})
    return pages


def build_json(pdf_path: str, out_json: str):
    pages = extract_pages(pdf_path)
    data = {"pages": pages}
    Path(out_json).write_text(json.dumps(data, indent=2))
    print(f"[SAVED] {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple PDF to JSON converter without GROBID"
    )
    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    build_json(args.pdf_path, args.output_json)
