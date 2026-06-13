#!/usr/bin/env python3
"""CSS lint for the rag-blog site (stdlib-only, no extra installs).

Catches issues that the build pipeline + tests don't catch at write time:

  1. Unbalanced @media blocks in theme/style.css. A missing closing
     brace would silently apply rules to everything after the @media,
     breaking the rest of the CSS. The regex extraction itself would
     also fail to match the malformed block, so a sanity check on
     brace counts surfaces the bug.

  2. Missing body.index @media print block in theme/style.css. The
     index page needs its own @media print block (re-applying the
     global print rules at body.index specificity) so it prints
     correctly. A regression that drops the block causes the index
     to print at the screen layout (16px font, 800px max-width)
     instead of the print layout (11pt font, no max-width) -- a
     silent regression the browser would render without raising
     an error.

  3. Inline <style>...</style> blocks in the SOURCE HTML files
     (lessons/*.html, reference/*.html). The build script copies
     these verbatim into site/, and the no-inline-style test
     (tests/test_build_site.py::test_no_inline_style_blocks)
     catches the regression at build time -- but catching it at
     write time is faster feedback.

  4. Inline style="..." attributes in the source HTML files. These
     bypass the stylesheet entirely (specificity 1,0,0,0 beats any
     stylesheet rule) and are the most common "specificity
     regression" in practice. Same rationale as #3: faster feedback
     than waiting for the build.

Why a custom Python script instead of stylelint / stylefmt:
  * Stdlib-only -- matches the project's "stdlib-only by design"
    philosophy for build/test code (see chunking.py).
  * Tailored to the project's specific risks (print rules, inline
    styles) rather than a generic CSS linter.
  * No new toolchain (no Node.js + npm install for the dev or CI).
  * Fast (sub-second, no startup overhead).

The print-block + no-inline-style checks are REDUNDANT with the
regression tests in tests/test_build_site.py -- the lint runs at
write time (pre-commit), the tests run at build time (CI). The
redundancy is intentional: the lint is faster feedback for the
developer, the tests are the source of truth that ships with the
project.

Usage:
    python scripts/css_lint.py
    # exit 0 = all checks pass, exit 1 = at least one check failed
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THEME_CSS = ROOT / "theme" / "style.css"
LESSONS_SRC = ROOT / "lessons"
REFERENCE_SRC = ROOT / "reference"

# Matches `<style>` or `<style ...>` (with optional attributes). Uses
# `[\s>]` (not `>`) so it also catches `<style\n>`. Case-insensitive
# to be safe.
INLINE_STYLE_TAG = re.compile(r"<style[\s>]", re.IGNORECASE)
# Matches `style="`, `style = "`, `style='`, `style = '`, etc. (any
# whitespace around `=` + either quote style). The HTML spec allows
# both, so the lint should catch both.
INLINE_STYLE_ATTR = re.compile(r"""style\s*=\s*['"]""", re.IGNORECASE)

# Extracts every `@media { ... }` block. Handles both the simple form
# (`@media print {`) and the conditional form (`@media print and (...)`).
# Balances one level of nested braces so a future nested at-rule (e.g.
# `@media print { @supports (...) { ... } }`) still matches. CSS rules
# inside use braces, so naive `[^}]*` would fail; the alternation
# `(?:[^{}]|\{[^{}]*\})*` handles one level of nesting.
_AT_MEDIA_BLOCK = re.compile(
    r"@media[^{]*\{(?:[^{}]|\{[^{}]*\})*\}",
)
# Same regex but requires `print` in the media query.
_AT_MEDIA_PRINT_BLOCK = re.compile(
    r"@media[^{]*print[^{]*\{(?:[^{}]|\{[^{}]*\})*\}",
)


def check_theme_css() -> list[str]:
    """Check theme/style.css for balanced @media blocks + body.index print."""
    if not THEME_CSS.exists():
        return [f"missing: {THEME_CSS.relative_to(ROOT)}"]

    errors: list[str] = []
    text = THEME_CSS.read_text(encoding="utf-8")
    rel = THEME_CSS.relative_to(ROOT)

    # Check 1: every @media block is balanced. The regex requires
    # balanced braces, so an unclosed block is silently skipped by
    # the regex. Detect this with a keyword-vs-block count: if the
    # number of @media keywords in the file doesn't match the
    # number of balanced blocks the regex extracted, some block
    # is unclosed or malformed.
    all_media_blocks = _AT_MEDIA_BLOCK.findall(text)
    # Strip CSS comments before counting keywords -- a comment that
    # mentions "@media" (e.g. "the global @media print rules") would
    # falsely inflate the count. Comments don't contain `{` or `}`,
    # so the block extraction regex doesn't match them.
    text_no_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    n_keywords = len(re.findall(r"@media\b", text_no_comments))
    n_matched = len(all_media_blocks)
    if n_keywords != n_matched:
        errors.append(
            f"{rel} has {n_keywords} @media keyword(s) but only "
            f"{n_matched} balanced block(s) -- some are likely "
            "unclosed or malformed"
        )
    # Sanity check: each extracted block should itself be balanced
    # (the regex guarantees this, so a failure here means the
    # regex is buggy).
    for i, block in enumerate(all_media_blocks, 1):
        opens = block.count("{")
        closes = block.count("}")
        if opens != closes:
            errors.append(
                f"{rel} @media block {i} has unbalanced braces "
                f"({opens} '{{' vs {closes} '}}'): {block[:80]!r}..."
            )

    # Check 2: at least one @media print block contains the body.index
    # + body.index .page rules. The global @media print block (for
    # lessons + references) has only body / .page / pre / a selectors;
    # the index needs its own block because the global rules lose to
    # body.index at specificity 0,0,1 vs 0,1,1.
    print_blocks = _AT_MEDIA_PRINT_BLOCK.findall(text)
    has_index_print = any(
        "body.index" in block and "body.index .page" in block for block in print_blocks
    )
    if not has_index_print:
        errors.append(
            f"{rel} has no @media print block containing the required "
            "body.index + body.index .page rules. The index will print "
            "at the screen layout instead of the print layout."
        )

    return errors


def check_no_inline_styles() -> list[str]:
    """Check source HTML for inline <style> blocks + style= attributes."""
    errors: list[str] = []
    for src_dir in (LESSONS_SRC, REFERENCE_SRC):
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.glob("*.html")):
            text = path.read_text(encoding="utf-8")
            rel = path.relative_to(ROOT)
            if INLINE_STYLE_TAG.search(text):
                errors.append(
                    f"{rel} contains an inline <style> block (should link to ../style.css instead)"
                )
            if INLINE_STYLE_ATTR.search(text):
                errors.append(
                    f'{rel} contains an inline style="..." attribute '
                    "(bypasses the stylesheet, specificity 1,0,0,0 -- "
                    "hard to override)"
                )
    return errors


def main() -> int:
    theme_errors = check_theme_css()
    html_errors = check_no_inline_styles()
    all_errors = theme_errors + html_errors

    if all_errors:
        print("CSS lint FAILED:", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            f"\n{len(all_errors)} issue(s) found. Fix them or update the "
            "lint if the behavior is intentional.",
            file=sys.stderr,
        )
        return 1

    print("CSS lint passed:")
    print(
        f"  - {THEME_CSS.relative_to(ROOT)}: "
        f"{len(_AT_MEDIA_BLOCK.findall(THEME_CSS.read_text(encoding='utf-8')))} "
        "@media block(s) balanced, body.index print block present"
    )
    for src_dir in (LESSONS_SRC, REFERENCE_SRC):
        if src_dir.is_dir():
            n_html = sum(1 for _ in src_dir.glob("*.html"))
            print(
                f"  - {src_dir.relative_to(ROOT)}/*.html ({n_html} files): "
                "no inline <style> blocks or style= attributes"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
