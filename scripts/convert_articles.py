#!/usr/bin/env python3
"""Convert article PDFs to simple Markdown files with pdftotext."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("articles"),
        help="Directory containing source PDF files.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        action="append",
        default=[],
        help="Specific PDF file to convert. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("articles/markdown_articles"),
        help="Directory for generated Markdown files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown outputs.",
    )
    return parser.parse_args()


def ensure_pdftotext() -> str:
    tool = shutil.which("pdftotext")
    if tool is None:
        raise SystemExit("Missing required tool: pdftotext")
    return tool


def markdown_for_pdf(pdf_path: Path, text: str) -> str:
    title = pdf_path.stem
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return (
        f"# {title}\n\n"
        f"- Source PDF: `{pdf_path.name}`\n"
        "- Converted with `pdftotext -nopgbrk`\n\n"
        "> Conversion note: this is plain-text extraction wrapped as Markdown. "
        "Tables, figures, and page layout may not be preserved.\n\n"
        f"{cleaned}\n"
    )


def convert_pdf(pdf_path: Path, output_dir: Path, pdftotext_bin: str, overwrite: bool) -> Path:
    output_path = output_dir / f"{pdf_path.stem}.md"
    if output_path.exists() and not overwrite:
        return output_path

    result = subprocess.run(
        [pdftotext_bin, "-nopgbrk", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    output_path.write_text(markdown_for_pdf(pdf_path, result.stdout), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    pdftotext_bin = ensure_pdftotext()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        pdf_paths = args.input_file
    else:
        pdf_paths = sorted(path for path in args.input_dir.glob("*.pdf") if path.is_file())

    invalid_paths = [path for path in pdf_paths if not path.is_file() or path.suffix.lower() != ".pdf"]
    if invalid_paths:
        raise SystemExit(f"Invalid PDF file: {invalid_paths[0]}")

    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {args.input_dir}")

    if args.input_file:
        print(f"Input files: {len(pdf_paths)}")
    else:
        print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")

    for pdf_path in pdf_paths:
        md_path = convert_pdf(pdf_path, output_dir, pdftotext_bin, args.overwrite)
        print(f"{pdf_path.name} -> {md_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
