"""Hybrid PDF to JSON conversion script.

This script demonstrates a simple pipeline that combines modern
PDF text extraction libraries with metadata extracted from a running
GROBID server. The resulting JSON contains page level text and the
TEI XML from GROBID.
"""

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio  # type: ignore
import pdfplumber  # type: ignore


def extract_pages(pdf_path: str):
    """Extract text from each page using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


async def extract_pages_async(pdf_path: str):
    return await asyncio.to_thread(extract_pages, pdf_path)


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


async def call_grobid_async(pdf_path: str, out_dir: str, grobid_url: str):
    return await asyncio.to_thread(call_grobid, pdf_path, out_dir, grobid_url)


async def run_marker(pdf_path: str) -> str:
    """Optional: run Marker for text extraction."""
    cmd = ["marker", pdf_path]
    try:
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, check=True
        )
        return result.stdout
    except FileNotFoundError:
        return ""


async def run_nougat(pdf_path: str) -> str:
    """Optional: run Nougat to parse formulas."""
    cmd = ["nougat", pdf_path]
    try:
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, check=True
        )
        return result.stdout
    except FileNotFoundError:
        return ""


async def run_docling(pdf_path: str) -> str:
    """Optional: run Docling for table detection."""
    cmd = ["docling", pdf_path]
    try:
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, check=True
        )
        return result.stdout
    except FileNotFoundError:
        return ""


async def build_json(pdf_path: str, out_json: str, grobid_url: str):
    os.makedirs("temp_grobid", exist_ok=True)

    tasks = [
        call_grobid_async(pdf_path, "temp_grobid", grobid_url),
        extract_pages_async(pdf_path),
        run_marker(pdf_path),
        run_nougat(pdf_path),
        run_docling(pdf_path),
    ]

    grobid_file, pages, marker_out, nougat_out, docling_out = await tqdm_asyncio.gather(*tasks)

    data = {
        "pages": pages,
        "marker": marker_out,
        "nougat": nougat_out,
        "docling": docling_out,
    }
    if grobid_file.exists():
        data["tei_xml"] = grobid_file.read_text()
    with open(out_json, "w") as f:
        json.dump(data, f)
    print(f"[SAVED] {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid PDF to JSON converter")
    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--grobid_url", default="http://localhost:8070")
    args = parser.parse_args()

    asyncio.run(build_json(args.pdf_path, args.output_json, args.grobid_url))
