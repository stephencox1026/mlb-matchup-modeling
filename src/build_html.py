"""Convert SPRINT_BRIEF.md to a self-contained HTML file with embedded images."""
import base64
import re
from pathlib import Path
import markdown

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
MD_PATH = DOCS_DIR / "SPRINT_BRIEF.md"
HTML_PATH = DOCS_DIR / "SPRINT_BRIEF.html"
FIGURES_DIR = DOCS_DIR / "figures"

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LAD Hitters vs LHP — Sprint Brief</title>
<style>
  @page {{ size: letter; margin: 0.75in; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 900px; margin: 0 auto; padding: 24px;
    color: #1a1a1a; line-height: 1.55; font-size: 14px;
  }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #2176AE; padding-bottom: 6px; }}
  h2 {{ font-size: 17px; color: #2176AE; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  h3 {{ font-size: 15px; margin-top: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #ccc; padding: 5px 8px; text-align: left; }}
  th {{ background: #f0f4f8; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafbfc; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 16px auto; border: 1px solid #eee; border-radius: 4px; }}
  code {{ background: #f0f4f8; padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
  pre {{ background: #f0f4f8; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 12px; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 24px 0; }}
  strong {{ color: #1a1a1a; }}
  @media print {{
    body {{ font-size: 12px; padding: 0; }}
    h1 {{ font-size: 18px; }}
    h2 {{ font-size: 15px; break-after: avoid; }}
    table {{ font-size: 11px; }}
    img {{ max-height: 400px; }}
    pre {{ font-size: 10px; }}
  }}
</style>
</head>
<body>
{content}
</body>
</html>
"""


def embed_images(html: str) -> str:
    """Replace image src paths with base64 data URIs."""
    def replace_img(match):
        src = match.group(1)
        img_path = DOCS_DIR / src
        if img_path.exists():
            data = base64.b64encode(img_path.read_bytes()).decode()
            ext = img_path.suffix.lstrip(".")
            mime = f"image/{ext}"
            return f'src="data:{mime};base64,{data}"'
        return match.group(0)

    return re.sub(r'src="([^"]+)"', replace_img, html)


def build():
    md_text = MD_PATH.read_text()
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html_body = embed_images(html_body)
    full_html = HTML_TEMPLATE.format(content=html_body)
    HTML_PATH.write_text(full_html)
    print(f"Built {HTML_PATH} ({len(full_html):,} bytes)")
    return HTML_PATH


if __name__ == "__main__":
    path = build()
    print(f"Opening in browser...")
    import subprocess
    subprocess.run(["open", str(path)])
