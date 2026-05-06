#!/usr/bin/env python3
"""
Build docs/beast.html — standalone page with the same content as the
Matchup dashboard's **Matchups · Beast** tab (optional Beast JSON tables,
writeups, scatter, slate picker when multiple slates exist, team filter, notes).

Runs the full dashboard HTML builder once, then extracts style, scripts, header
block (through weather + team filter), and the Beast panel so behavior stays
aligned with gen_matchup_dashboard_html.py.

Regenerate:
  PYTHONPATH=src python3 src/gen_beast_html.py
"""
from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gen_matchup_dashboard_html import build_dashboard_html  # noqa: E402

DOCS = ROOT / "docs"
DEFAULT_OUT = DOCS / "beast.html"

# When the dashboard slate switch runs, it updates the header from .main-slate-view;
# this page has no main tab — patch header counts from visible Beast views.
_BEAST_HDR_SYNC_JS = r"""
(function () {
  function syncHdrFromBeast() {
    var sel = document.getElementById("matchup-slate-select");
    var sd = sel ? sel.value : null;
    var vis = null;
    if (sd) {
      document.querySelectorAll(".beast-slate-view").forEach(function (el) {
        if (el.getAttribute("data-slate") === sd) vis = el;
      });
    }
    if (!vis) vis = document.querySelector(".beast-slate-view:not(.hidden)");
    var d = document.getElementById("hdr-slate-d");
    var n = document.getElementById("hdr-slate-n");
    if (vis && d && n) {
      d.textContent = vis.getAttribute("data-slate") || "";
      n.textContent = vis.getAttribute("data-n") || "0";
    }
  }
  function wire() {
    var sel = document.getElementById("matchup-slate-select");
    if (sel) {
      sel.addEventListener("change", function () {
        setTimeout(syncHdrFromBeast, 0);
      });
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () {
        setTimeout(syncHdrFromBeast, 0);
      });
    } else {
      setTimeout(syncHdrFromBeast, 0);
    }
  }
  wire();
})();
"""


def _extract_style(full: str) -> str:
    m = re.search(r"<style>(.*?)</style>", full, flags=re.DOTALL)
    return m.group(1) if m else ""


def _extract_main_script(full: str) -> str:
    """The dashboard ends with a single executable <script> block (after </footer>)."""
    m = re.search(
        r"(?s)<script>\s*(function tableToCsv.*?)</script>\s*</body>",
        full,
    )
    return m.group(1) if m else ""


def _extract_pre_nav_body(full: str) -> str:
    """Everything after <body> up to (but not including) the tab bar."""
    m = re.search(r"(?s)<body>\s*(.*?)\s*<nav\s+class=\"tabbar\"", full)
    return m.group(1) if m else ""


def _extract_beast_inner(full: str) -> str:
    m = re.search(
        r'(?s)<section id="panel-beast"[^>]*>(.*?)</section>\s*<section id="panel-residual"',
        full,
    )
    return m.group(1) if m else ""


def _extract_title_slate(full: str) -> str:
    m = re.search(r"<title>Matchup dashboard · ([^<]+)</title>", full)
    return (m.group(1) or "").strip() if m else ""


def _extract_footer(full: str) -> str:
    m = re.search(r"(?s)(<footer>.*?</footer>)", full)
    return m.group(1) if m else "<footer></footer>"


def build_beast_html_document(*, slate: str | None = None) -> str:
    full = build_dashboard_html(slate=slate)
    css = _extract_style(full)
    js_core = _extract_main_script(full)
    pre_nav = _extract_pre_nav_body(full)
    beast_inner = _extract_beast_inner(full)
    title_sd = _extract_title_slate(full)
    footer = _extract_footer(full)

    pre_nav = pre_nav.replace(
        "<h1>Matchup dashboard</h1>",
        "<h1>Beast · matchup model</h1>",
        1,
    )

    beast_section = (
        '<section id="panel-beast" class="tabpanel" role="tabpanel">'
        f"{beast_inner}</section>"
    )

    footer_note = (
        '<p class="intro muted" style="margin-top:0.75rem">'
        "This file is the <strong>Beast-only</strong> view. "
        "Regenerate: <code>PYTHONPATH=src python3 src/gen_beast_html.py</code>. "
        "Full UI: <code>docs/matchup_dashboard.html</code> or "
        "<code>data/reports/matchup_dashboard.html</code>.</p>"
    )
    if "</p>" in footer and "<footer>" in footer:
        footer = footer.replace("</footer>", footer_note + "</footer>", 1)

    body = f"{pre_nav}\n{beast_section}\n{footer}"
    title = f"Beast · {title_sd}" if title_sd else "Beast · matchup model"
    js = js_core + "\n" + _BEAST_HDR_SYNC_JS

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
{body}
<script>{js}</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="Slate date (YYYY-MM-DD); same as matchup dashboard")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output HTML (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the file in the default browser when done (macOS: open)",
    )
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = build_beast_html_document(slate=args.date)
    args.output.write_text(doc, encoding="utf-8")
    print(f"Wrote {args.output}")
    if not args.no_open and sys.platform == "darwin":
        subprocess.run(["open", str(args.output)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
