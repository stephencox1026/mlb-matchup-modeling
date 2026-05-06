#!/usr/bin/env python3
"""Export section_7/8/10.md to PDF using fpdf2 (ASCII-safe)."""
import shutil
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "data" / "reports" / "archive"
W = 180


def ascii_fold(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def clean(s: str) -> str:
    for token in ("**", "`"):
        s = s.replace(token, "")
    return ascii_fold(s)


class Doc(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=14)


def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    pdf = Doc()
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)
    for raw in md_path.read_text(encoding="utf-8").splitlines():
        line = clean(raw.rstrip())
        if not line:
            pdf.ln(2)
            continue
        if raw.lstrip().startswith("#"):
            level = len(raw) - len(raw.lstrip("#"))
            title = clean(raw.lstrip("#").strip())
            pdf.set_font("Helvetica", "B", 14 if level <= 1 else 12 if level == 2 else 10)
            pdf.multi_cell(W, 5, title)
            pdf.set_font("Helvetica", size=9)
            continue
        if line.startswith("|") and "---" in raw:
            continue
        if raw.strip().startswith("|"):
            line = clean(" ".join(c.strip() for c in raw.split("|") if c.strip()))
        if line.startswith("---"):
            pdf.ln(2)
            continue
        pdf.multi_cell(W, 4, line if line else " ")
    pdf.output(str(pdf_path))


def main():
    if len(sys.argv) < 2:
        print("Usage: export_section_pdfs_fpdf.py YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)
    date = sys.argv[1]
    day_dir = ARCH / date
    if not day_dir.is_dir():
        print(f"Missing {day_dir}", file=sys.stderr)
        sys.exit(1)
    reports = ROOT / "data" / "reports"
    for name in ("section_7", "section_8", "section_10"):
        md = day_dir / f"{name}.md"
        out = day_dir / f"{name}_{date}.pdf"
        md_to_pdf(md, out)
        shutil.copy(out, reports / f"{name}.pdf")
        print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
