"""Convert article PDFs into Markdown that is easier for coding agents to read.

Usage::

    python3 scripts/convert_articles.py
    python3 scripts/convert_articles.py --input-dir articles --output-dir articles/agent_readable
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


LIGATURE_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2009": " ",
        "\u200b": "",
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "--",
        "\u2212": "-",
    }
)

NOISE_LINE_PATTERNS = [
    re.compile(r"^This ICCV Workshop paper is the Open Access version.*$", re.IGNORECASE),
    re.compile(r"^Except for this watermark, it is identical to the accepted version;?$", re.IGNORECASE),
    re.compile(
        r"^the final published version of the proceedings is available on IEEE Xplore\.$",
        re.IGNORECASE,
    ),
    re.compile(r"^Preprint\..*$", re.IGNORECASE),
    re.compile(r"^Equal contribution\.$", re.IGNORECASE),
    re.compile(r"^Corresponding authors?\.$", re.IGNORECASE),
    re.compile(r"^This research was supported by.*$", re.IGNORECASE),
    re.compile(r"^\d+\s+https?://\S+$", re.IGNORECASE),
    re.compile(r"^.*\bare with the\b.*$", re.IGNORECASE),
    re.compile(r"^.*\bis with the\b.*$", re.IGNORECASE),
    re.compile(r"^Research in .*project$", re.IGNORECASE),
    re.compile(r"^UA\d+.*$", re.IGNORECASE),
]

KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "preliminaries",
    "method",
    "methods",
    "approach",
    "architecture",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "acknowledgment",
    "acknowledgements",
    "acknowledgment",
    "references",
    "appendix",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs in a directory into agent-readable Markdown."
    )
    parser.add_argument("--input-dir", default="articles", help="Directory containing PDF articles")
    parser.add_argument(
        "--output-dir",
        default="articles/agent_readable",
        help="Directory that will receive the Markdown files",
    )
    return parser.parse_args()


def ensure_tool(tool_name: str) -> None:
    if shutil.which(tool_name):
        return
    raise SystemExit(f"Required tool not found on PATH: {tool_name}")


def run_command(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Command failed ({' '.join(args)}): {stderr}")
    return result.stdout


def parse_pdfinfo(pdf_path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in run_command(["pdfinfo", str(pdf_path)]).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip()
    return metadata


def extract_pdf_text(pdf_path: Path) -> str:
    return run_command(["pdftotext", "-enc", "UTF-8", "-nopgbrk", str(pdf_path), "-"])


def normalize_line(line: str) -> str:
    cleaned = line.translate(LIGATURE_TRANSLATION).replace("\x0c", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned


def alpha_count(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def repair_spaced_heading_words(text: str) -> str:
    repaired = text
    previous = None
    while repaired != previous:
        previous = repaired
        repaired = re.sub(r"\b([A-Z]) ([A-Z]{2,})\b", r"\1\2", repaired)
    return repaired


def smart_title(word: str) -> str:
    if not word:
        return word
    if re.fullmatch(r"[IVXLCDM]+\.?", word):
        return word
    if word.isupper() and len(word) <= 4:
        return word
    if "-" in word:
        return "-".join(smart_title(part) for part in word.split("-"))
    if "/" in word:
        return "/".join(smart_title(part) for part in word.split("/"))
    return word[:1].upper() + word[1:].lower()


def normalize_heading_text(text: str) -> str:
    text = repair_spaced_heading_words(text)
    text = re.sub(r"\s+", " ", text).strip(" .:")
    match = re.match(r"^([IVXLCDM]+\.?)\s+(.+)$", text)
    if match:
        prefix, rest = match.groups()
        if rest.upper() == rest:
            rest = " ".join(smart_title(word) for word in rest.split())
        return f"{prefix} {rest}"
    match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$", text)
    if match:
        prefix, rest = match.groups()
        if rest.upper() == rest:
            rest = " ".join(smart_title(word) for word in rest.split())
        return f"{prefix}. {rest}"
    if text.upper() == text:
        return " ".join(smart_title(word) for word in text.split())
    return text


def is_noise_line(line: str) -> bool:
    if not line:
        return False
    if re.fullmatch(r"\d+", line):
        return True
    if re.fullmatch(r"[•·]", line):
        return True
    if "e-mail:" in line.lower():
        return True
    if "@" in line:
        return True
    return any(pattern.match(line) for pattern in NOISE_LINE_PATTERNS)


def split_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line:
            current.append(line)
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def is_caption_block(text: str) -> bool:
    return bool(re.match(r"^(fig(?:ure)?\.?|table)\s+\d+[:.]?", text, re.IGNORECASE))


def normalize_caption(text: str) -> str:
    text = re.sub(r"^fig\.", "Figure", text, flags=re.IGNORECASE)
    text = re.sub(r"^figure\s+", "Figure ", text, flags=re.IGNORECASE)
    text = re.sub(r"^table\s+", "Table ", text, flags=re.IGNORECASE)
    text = re.sub(r"^((?:Figure|Table)\s+\d+)\.\s*", r"\1: ", text)
    return text


def is_list_block(lines: list[str]) -> bool:
    return all(
        re.match(r"^(?:[-*]|\d+[.)])\s+", line)
        for line in lines
    )


def looks_like_artifact_block(lines: list[str]) -> bool:
    if not lines:
        return False
    if len(lines) == 1:
        return False
    total_chars = sum(len(line) for line in lines)
    long_sentence = any(re.search(r"[.!?]$", line) and len(line) > 40 for line in lines)
    short_lines = all(len(line) <= 24 for line in lines)
    uppercase_ratio = 0.0
    letters = "".join(ch for line in lines for ch in line if ch.isalpha())
    if letters:
        uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return total_chars <= 120 and short_lines and not long_sentence and uppercase_ratio >= 0.55


def is_single_line_artifact(line: str) -> bool:
    if not line:
        return False
    words = re.findall(r"[A-Za-z0-9%+-]+", line)
    if re.fullmatch(r"\(?[a-z]\)?", line, re.IGNORECASE):
        return True
    if re.fullmatch(r"[A-Z]{1,5}", line):
        return True
    if re.fullmatch(r"\d+(?:\s+\d+)*", line):
        return True
    if len(words) <= 3 and len(line) <= 24 and not re.search(r"[.!?]$", line):
        return True
    if len(line) <= 18 and alpha_count(line) <= 4 and not re.search(r"[.!?:;/]", line):
        return True
    return False


def is_heading_line(text: str) -> bool:
    text = repair_spaced_heading_words(text)
    if len(text) > 100:
        return False
    lowered = text.lower().strip(" .:")
    if lowered in KNOWN_HEADINGS:
        return True
    if re.match(r"^abstract(?:\b|[-—:]+)", text, re.IGNORECASE):
        return True
    roman_match = re.match(r"^([IVXLCDM]+\.?)\s+(.+)$", text)
    if roman_match and alpha_count(roman_match.group(2)) >= 3:
        return True
    numeric_match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$", text)
    if numeric_match and alpha_count(numeric_match.group(2)) >= 3:
        return True
    if re.match(r"^appendix(?:\s+[A-Z0-9]+)?(?:[.:]\s*.+)?$", text, re.IGNORECASE):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
    if 1 <= len(words) <= 8 and text.upper() == text and len(text) >= 8 and alpha_count(text) >= 6:
        return True
    return False


def join_wrapped_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    paragraph = lines[0]
    for line in lines[1:]:
        if paragraph.endswith("-") and not paragraph.endswith(" -"):
            paragraph = paragraph[:-1] + line
            continue
        paragraph = f"{paragraph} {line}"
    return paragraph


def normalize_heading_block(lines: list[str]) -> list[str]:
    text = repair_spaced_heading_words(" ".join(lines))
    abstract_match = re.match(r"^abstract(?:\s*[-—:]+\s*|\s*)(.*)$", text, re.IGNORECASE)
    if abstract_match:
        blocks = ["## Abstract"]
        abstract_body = abstract_match.group(1).strip()
        if abstract_body:
            blocks.append(abstract_body)
        return blocks
    return [f"## {normalize_heading_text(text)}"]


def clean_extracted_lines(raw_text: str) -> list[str]:
    cleaned_lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = normalize_line(raw_line)
        if is_noise_line(line):
            continue
        cleaned_lines.append(line)

    merged_lines: list[str] = []
    index = 0
    while index < len(cleaned_lines):
        line = cleaned_lines[index]
        if (
            line.endswith("-")
            and index + 2 < len(cleaned_lines)
            and cleaned_lines[index + 1] == ""
            and cleaned_lines[index + 2]
            and cleaned_lines[index + 2][0].islower()
        ):
            merged_lines.append(line)
            index += 2
            continue
        if (
            merged_lines
            and merged_lines[-1].endswith("-")
            and line
            and line[0].islower()
        ):
            merged_lines[-1] = merged_lines[-1][:-1] + line
        else:
            merged_lines.append(line)
        index += 1
    return merged_lines


def find_body_start(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if re.match(
            r"^abstract(?:\s*[-—:]+\s*|\s*)",
            repair_spaced_heading_words(line),
            re.IGNORECASE,
        ):
            return index
    for index, line in enumerate(lines):
        if is_heading_line(line):
            return index
    return 0


def render_non_heading_block(block_lines: list[str]) -> list[str]:
    if not block_lines:
        return []
    if len(block_lines) == 1 and is_single_line_artifact(block_lines[0]):
        return []
    if looks_like_artifact_block(block_lines):
        return []

    if is_caption_block(block_lines[0]):
        caption_lines = [block_lines[0]]
        split_index = 1
        while split_index < len(block_lines):
            caption_lines.append(block_lines[split_index])
            if re.search(r"[.!?]$", block_lines[split_index]):
                split_index += 1
                break
            split_index += 1
        rendered = [f"> {normalize_caption(join_wrapped_lines(caption_lines))}"]
        rendered.extend(render_non_heading_block(block_lines[split_index:]))
        return rendered

    joined = join_wrapped_lines(block_lines)
    if not joined:
        return []
    if is_list_block(block_lines):
        return block_lines
    return [joined]


def render_blocks(lines: list[str]) -> list[str]:
    rendered: list[str] = []
    for block_lines in split_blocks(lines):
        if not block_lines:
            continue
        first_line = block_lines[0]
        if is_heading_line(first_line):
            heading_blocks = normalize_heading_block([first_line])
            rendered.append(heading_blocks[0])
            trailing_blocks = render_non_heading_block(block_lines[1:])
            if len(heading_blocks) > 1:
                if trailing_blocks:
                    trailing_blocks[0] = f"{heading_blocks[1]} {trailing_blocks[0]}"
                else:
                    trailing_blocks = [heading_blocks[1]]
            rendered.extend(trailing_blocks)
            continue
        rendered.extend(render_non_heading_block(block_lines))
    return rendered


def extract_identifier(raw_text: str) -> str | None:
    match = re.search(r"\barXiv:\S+", raw_text)
    if match:
        return match.group(0).rstrip(",.;)")
    return None


def markdown_for_pdf(pdf_path: Path) -> str:
    metadata = parse_pdfinfo(pdf_path)
    raw_text = extract_pdf_text(pdf_path)
    cleaned_lines = clean_extracted_lines(raw_text)
    body_lines = cleaned_lines[find_body_start(cleaned_lines) :]
    rendered_blocks = render_blocks(body_lines)

    title = metadata.get("title") or pdf_path.stem
    authors = metadata.get("author", "Unknown")
    pages = metadata.get("pages", "Unknown")
    identifier = extract_identifier(raw_text)
    converted_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = "\n".join(
        [
        f"# {title}",
        "",
        f"- Source PDF: `{pdf_path.as_posix()}`",
        f"- Authors: {authors}",
        f"- Pages: {pages}",
        ]
    )
    if identifier:
        header += f"\n- Identifier: {identifier}"
    header += "\n" + "\n".join(
        [
            f"- Converted: {converted_at}",
            "- Extraction: `pdftotext -nopgbrk` with Markdown cleanup for agent reading",
            "",
            "> Conversion note: the text flow is optimized for reading, not for preserving the original PDF layout. Use the PDF for equations, tables, or figure placement that depends on page geometry.",
        ]
    )

    body = "\n\n".join(block for block in rendered_blocks if block).strip()
    return f"{header}\n\n{body}\n".strip() + "\n"


def build_index(pdf_paths: list[Path], output_dir: Path) -> str:
    lines = [
        "# Agent-Readable Articles",
        "",
        "This directory contains Markdown conversions of the PDFs in `articles/`.",
        "",
        "- Format goal: maximize readability for coding agents and local text search.",
        "- Tradeoff: layout-heavy content such as equations, tables, and figure placement may still require opening the source PDF.",
        "",
        "## Files",
        "",
    ]
    for pdf_path in pdf_paths:
        md_name = f"{pdf_path.stem}.md"
        lines.append(f"- `{md_name}` from `{pdf_path.name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_tool("pdfinfo")
    ensure_tool("pdftotext")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    pdf_paths = sorted(input_dir.glob("*.pdf"))

    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_paths:
        markdown = markdown_for_pdf(pdf_path)
        output_path = output_dir / f"{pdf_path.stem}.md"
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Converted {pdf_path.name} -> {output_path}")

    readme_path = output_dir / "README.md"
    readme_path.write_text(build_index(pdf_paths, output_dir), encoding="utf-8")
    print(f"Wrote index -> {readme_path}")


if __name__ == "__main__":
    main()
