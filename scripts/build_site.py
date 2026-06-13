#!/usr/bin/env python3
"""Build the GitHub Pages site for the learning materials.

Scans ``lessons/*.html`` and ``reference/*.html`` and produces:
  - ``site/index.html``  — a navigation page with links + descriptions
  - ``site/lessons/``    — copies of the lesson HTML files
  - ``site/reference/``  — copies of the reference HTML files

The output is consumed by the ``publish.yml`` GitHub Actions workflow
and deployed to GitHub Pages. The script is stdlib-only so it can run
in a fresh Actions runner with no dependency install.

Usage:
    python scripts/build_site.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
LESSONS_SRC = ROOT / "lessons"
REFERENCE_SRC = ROOT / "reference"

# Cap the description shown on the index card so the page stays scannable.
DESC_MAX = 220


def extract_meta(html: str) -> tuple[str, str]:
    """Pull ``(title, description)`` out of a lesson or reference HTML file.

    Title: from ``<title>``.
    Description: first ``<p>`` after the ``<h1>`` heading, with all HTML
    tags stripped and whitespace collapsed. Falls back to the first
    paragraph in the document if no ``<h1>`` is present.
    """
    title_match = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
    title = title_match.group(1).strip() if title_match else "Untitled"

    # First <p> after the h1 (works for both lesson and reference layouts)
    h1_end = html.find("</h1>")
    search_from = h1_end if h1_end != -1 else 0
    p_match = re.search(r"<p[^>]*>(.*?)</p>", html[search_from:], re.DOTALL)
    if not p_match:
        # Fallback: first <p> anywhere
        p_match = re.search(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)
    desc = ""
    if p_match:
        desc = re.sub(r"<[^>]+>", "", p_match.group(1)).strip()
        desc = re.sub(r"\s+", " ", desc)
    return title, desc[:DESC_MAX]


def build_index(lessons: list[dict], refs: list[dict]) -> str:
    """Render the navigation page as a self-contained HTML string."""

    def render_cards(items: list[dict], subdir: str) -> str:
        if not items:
            return '<p class="empty">No files yet.</p>'
        rows = []
        for item in items:
            desc = item["desc"] or "(no description)"
            rows.append(
                f'<li><a href="{subdir}/{item["file"]}">{item["title"]}</a>'
                f'<div class="desc">{desc}</div></li>'
            )
        return "\n".join(rows)

    lesson_cards = render_cards(lessons, "lessons")
    ref_cards = render_cards(refs, "reference")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Learning Materials — RAG, AI, LLMs</title>
<style>
  :root {{
    --ink: #1a1a1a; --ink-soft: #555; --paper: #fbfaf7;
    --accent: #b8541b; --accent-soft: #f4e1d2; --rule: #d8d2c1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0; background: var(--paper); color: var(--ink);
    font-family: "Charter", "Iowan Old Style", "Georgia", serif;
    font-size: 16px; line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }}
  .page {{ max-width: 800px; margin: 0 auto; padding: 48px 28px 64px; }}
  h1 {{
    font-size: 32px; margin: 0 0 8px; letter-spacing: -0.01em;
    border-bottom: 2px solid var(--accent); padding-bottom: 8px;
  }}
  h2 {{
    font-size: 20px; margin: 36px 0 12px; color: var(--accent);
  }}
  .meta {{ color: var(--ink-soft); font-size: 14px; margin: 0 0 24px; }}
  ul {{ list-style: none; padding: 0; margin: 0; }}
  li {{
    background: #fff; border: 1px solid var(--rule); border-radius: 6px;
    padding: 14px 18px; margin: 10px 0;
    transition: border-color 120ms ease, transform 120ms ease;
  }}
  li:hover {{
    border-color: var(--accent);
    transform: translateY(-1px);
  }}
  a {{
    color: var(--ink); text-decoration: none; font-weight: 600;
    font-size: 17px;
  }}
  a:hover {{ color: var(--accent); }}
  .desc {{
    color: var(--ink-soft); font-size: 14px; margin-top: 4px;
    line-height: 1.45;
  }}
  .empty {{ color: var(--ink-soft); font-style: italic; }}
  footer {{
    margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--rule);
    color: var(--ink-soft); font-size: 13px;
  }}
  @media (max-width: 600px) {{
    .page {{ padding: 32px 18px; }}
    h1 {{ font-size: 26px; }}
  }}
</style>
</head>
<body>
<main class="page">
<h1>Learning Materials</h1>
<p class="meta">
  RAG, AI, and LLM concepts — taught as a senior engineer with no ML background.
  {len(lessons)} lessons · {len(refs)} reference cheat sheets.
</p>

<h2>Lessons</h2>
<ul>
{lesson_cards}
</ul>

<h2>Reference cheat sheets</h2>
<ul>
{ref_cards}
</ul>

<footer>
  Built by <code>scripts/build_site.py</code> · deployed via GitHub Pages.
  Each card links to a self-contained HTML file.
</footer>
</main>
</body>
</html>
"""


def collect(src_dir: Path) -> list[dict]:
    """Return ``[{"file", "title", "desc"}, ...]`` for every HTML in ``src_dir``."""
    out: list[dict] = []
    for path in sorted(src_dir.glob("*.html")):
        html = path.read_text(encoding="utf-8")
        title, desc = extract_meta(html)
        out.append({"file": path.name, "title": title, "desc": desc})
    return out


def main() -> int:
    if not LESSONS_SRC.is_dir():
        print(f"error: {LESSONS_SRC} does not exist", file=sys.stderr)
        return 1
    if not REFERENCE_SRC.is_dir():
        print(f"error: {REFERENCE_SRC} does not exist", file=sys.stderr)
        return 1

    # Wipe and re-create site/ so we don't leave stale files from a
    # previous build (e.g. a lesson that was deleted upstream).
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "lessons").mkdir(parents=True)
    (SITE / "reference").mkdir(parents=True)

    lessons = collect(LESSONS_SRC)
    refs = collect(REFERENCE_SRC)

    for item in lessons:
        shutil.copy2(LESSONS_SRC / item["file"], SITE / "lessons" / item["file"])
    for item in refs:
        shutil.copy2(REFERENCE_SRC / item["file"], SITE / "reference" / item["file"])

    (SITE / "index.html").write_text(build_index(lessons, refs), encoding="utf-8")

    print(f"Built site with {len(lessons)} lessons and {len(refs)} references")
    print(f"  Output: {SITE}/")
    for item in lessons:
        print(f"    lessons/{item['file']:<50s} {item['title']}")
    for item in refs:
        print(f"    reference/{item['file']:<50s} {item['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
