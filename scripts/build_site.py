#!/usr/bin/env python3
"""Build the GitHub Pages site for the learning materials.

Scans ``lessons/*.html`` and ``reference/*.html`` and produces:
  - ``site/index.html``  — a navigation page with links + descriptions
  - ``site/lessons/``    — copies of the lesson HTML files
  - ``site/reference/``  — copies of the reference HTML files
  - ``site/style.css``   — copied from ``theme/style.css`` (hand-maintained
    shared stylesheet for the index, lessons, and references)

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
THEME_SRC = ROOT / "theme"

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
    # Counts go in the meta line below the page title. Kept as a
    # variable so the f-string below is pure structure (the only
    # interpolations left are the card lists).
    counts = f"{len(lessons)} lessons · {len(refs)} reference cheat sheets."

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Learning Materials — RAG, AI, LLMs</title>
<link rel="stylesheet" href="style.css">
</head>
<body class="index">
<main class="page">
<h1>Learning Materials</h1>
<p class="meta">
  RAG, AI, and LLM concepts — taught as a senior engineer with no ML background.
  {counts}
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

    # Wipe only the generated subdirs so we don't leave stale files
    # from a previous build (e.g. a lesson that was deleted upstream).
    if (SITE / "lessons").exists():
        shutil.rmtree(SITE / "lessons")
    if (SITE / "reference").exists():
        shutil.rmtree(SITE / "reference")
    (SITE / "lessons").mkdir(parents=True)
    (SITE / "reference").mkdir(parents=True)
    # Copy the hand-maintained site stylesheet into the build output.
    shutil.copy2(THEME_SRC / "style.css", SITE / "style.css")

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
