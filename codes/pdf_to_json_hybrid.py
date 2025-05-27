"""Hybrid PDF to JSON conversion script.

This example showcases a lightweight asynchronous pipeline for
converting a PDF into a structured JSON document. It combines page
text extracted via ``pdfplumber`` with metadata produced by a running
GROBID server.  The script is intentionally simple so that additional
parsers (e.g. ``Marker`` or ``Nougat``) can easily be plugged in.
"""

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path

from tqdm import tqdm
import pdfplumber  # type: ignore


async def extract_pages(pdf_path: str):
    """Extract text from each page using ``pdfplumber``."""

    def _run() -> list[str]:
        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return pages

    return await asyncio.to_thread(_run)


async def call_grobid(pdf_path: str, out_dir: str, grobid_url: str) -> Path:
    """Call the GROBID service to obtain TEI XML."""

    def _run() -> Path:
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
        return Path(out_dir) / f"{Path(pdf_path).stem}.tei.xml"

    return await asyncio.to_thread(_run)


async def build_json(pdf_path: str, out_json: str, grobid_url: str) -> None:
    """Run the hybrid conversion and save the resulting JSON."""

    os.makedirs("temp_grobid", exist_ok=True)

    tasks = {
        "pages": asyncio.create_task(extract_pages(pdf_path)),
        "tei": asyncio.create_task(call_grobid(pdf_path, "temp_grobid", grobid_url)),
    }

    results = {}
    with tqdm(total=len(tasks), desc="Processing", leave=False) as pbar:
        for name, task in tasks.items():
            res = await task
            results[name] = res
            pbar.update(1)

    pages: list[str] = results["pages"]
    tei_file: Path = results["tei"]

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

    asyncio.run(build_json(args.pdf_path, args.output_json, args.grobid_url))
