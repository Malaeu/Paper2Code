"""Hybrid PDF to JSON conversion script.

This script demonstrates a simple pipeline that combines modern
PDF text extraction libraries with metadata extracted from a running
GROBID server. The resulting JSON contains page level text and the
TEI XML from GROBID.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import pdfplumber  # type: ignore


def extract_pages(pdf_path: str):
    """Extract text from each page using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def call_grobid(pdf_path: str, out_dir: str, grobid_url: str):
    """Call the GROBID service to obtain TEI XML."""
    cmd = [
        "grobid-client",
        "--input",
        pdf_path,
        "--output",
        out_dir,
        "--url",
        grobid_url,
        "--processFulltext",
    ]
    subprocess.run(cmd, check=True)
    tei_file = Path(out_dir) / f"{Path(pdf_path).stem}.tei.xml"
    return tei_file


def build_json(pdf_path: str, out_json: str, grobid_url: str):
    os.makedirs("temp_grobid", exist_ok=True)
    tei_file = call_grobid(pdf_path, "temp_grobid", grobid_url)
    pages = extract_pages(pdf_path)
    data = {"pages": pages}
    if tei_file.exists():
        data["tei_xml"] = tei_file.read_text()
    with open(out_json, "w") as f:
        json.dump(data, f)
    print(f"[SAVED] {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid PDF to JSON converter")
    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--grobid_url", default="http://localhost:8070")
    args = parser.parse_args()

    build_json(args.pdf_path, args.output_json, args.grobid_url)
