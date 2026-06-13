"""Smoke tests for the static site build script (no ML deps)."""

import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = ROOT / "scripts" / "build_site.py"
SITE_INDEX = ROOT / "site" / "index.html"

# One stable substring per lesson / reference title. If the build script's
# title-extraction regex ever stops matching (tag typo, encoding change, etc.)
# the corresponding fragment disappears from the rendered card link and one
# of these assertions fires.
LESSON_TITLE_FRAGMENTS = ("Embeddings", "Vector Search", "BM25", "RRF")
REFERENCE_TITLE_FRAGMENTS = ("Embeddings", "Vector Index", "BM25", "RRF")

# Anchors the title regression guard to the rendered card link so a
# description that happens to mention the same word cannot mask a broken
# title regex. Captures (subdir, visible_link_text) for every card.
CARD_LINK = re.compile(r'<a href="(lessons|reference)/[^"]+">([^<]+)</a>')


class TestBuildSite(unittest.TestCase):
    """Smoke test: the build runs cleanly and the index lists every page.

    The build is fast (sub-second for 8 HTML files) and idempotent (it wipes
    and re-creates ``site/`` on each run), so we run it once in ``setUpClass``
    and share the parsed index across all test methods.
    """

    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        cls.build_returncode = result.returncode
        cls.build_stderr = result.stderr
        cls.index_text = SITE_INDEX.read_text(encoding="utf-8") if SITE_INDEX.exists() else ""
        # (subdir, title) tuple for every rendered card link.
        cls.card_links = CARD_LINK.findall(cls.index_text)

    def test_build_exits_cleanly(self):
        # Catches a regression where the script crashes or prints to stderr.
        self.assertEqual(
            self.build_returncode,
            0,
            f"build failed (exit {self.build_returncode}): {self.build_stderr}",
        )
        self.assertEqual(self.build_stderr, "")

    def test_index_html_was_written(self):
        # Catches a regression where the script runs but doesn't write output.
        self.assertTrue(SITE_INDEX.exists(), f"missing {SITE_INDEX}")

    def test_index_contains_every_lesson_title(self):
        titles = [t for (subdir, t) in self.card_links if subdir == "lessons"]
        # Count check: silent file drops would otherwise be invisible.
        self.assertEqual(
            len(titles),
            len(LESSON_TITLE_FRAGMENTS),
            f"expected {len(LESSON_TITLE_FRAGMENTS)} lesson cards, got {len(titles)}: {titles}",
        )
        # Substring check inside the rendered link text, not free index text.
        for needle in LESSON_TITLE_FRAGMENTS:
            self.assertTrue(
                any(needle in t for t in titles),
                f"no lesson card link contains fragment {needle!r}: {titles}",
            )

    def test_index_contains_every_reference_title(self):
        titles = [t for (subdir, t) in self.card_links if subdir == "reference"]
        self.assertEqual(
            len(titles),
            len(REFERENCE_TITLE_FRAGMENTS),
            f"expected {len(REFERENCE_TITLE_FRAGMENTS)} reference cards, "
            f"got {len(titles)}: {titles}",
        )
        for needle in REFERENCE_TITLE_FRAGMENTS:
            self.assertTrue(
                any(needle in t for t in titles),
                f"no reference card link contains fragment {needle!r}: {titles}",
            )

    def _build_succeeded(self) -> bool:
        # Precondition shared by the post-build assertions below: if the
        # build itself failed, the rendered site/ is unreliable (possibly
        # missing or stale) and any check against it would be a false
        # positive.
        return self.build_returncode == 0 and self.build_stderr == ""

    def test_no_inline_style_blocks(self):
        # Guards the CSS-extraction refactor: every page (index, lessons,
        # references) must link to the shared stylesheet instead of
        # inlining rules. A regression that reintroduces an inline
        # <style> in any of these files would cause a silent style drift
        # that the browser would render without raising an error -- this
        # test catches it at build time.
        if not self._build_succeeded():
            self.fail(
                f"build did not succeed (exit {self.build_returncode}); "
                f"cannot check for inline <style> blocks: {self.build_stderr}"
            )

        # site/ is the build output, not the source -- the source HTML
        # files in lessons/ and reference/ MAY contain inline <style>
        # before the build (they don't anymore, but the test is for the
        # build output). rglob catches every HTML under site/ including
        # any new top-level files someone might add in the future.
        site_dir = ROOT / "site"
        html_files = sorted(site_dir.rglob("*.html"))
        inline_style_files = []
        for p in html_files:
            text = p.read_text(encoding="utf-8")
            if "<style>" in text or "<style " in text:
                inline_style_files.append(str(p.relative_to(ROOT)))
        self.assertEqual(
            inline_style_files,
            [],
            "the following HTML files contain an inline <style> block "
            "(should link to the shared stylesheet instead): "
            f"{inline_style_files}",
        )

    def test_theme_css_is_copied_to_site(self):
        # Guards the build script's copy step: theme/style.css is the
        # hand-maintained source of truth; site/style.css is what gets
        # deployed. If the copy step ever breaks (file renamed, the
        # THEME_SRC constant removed, the copy skipped in a refactor),
        # the deployed site will silently fall back to whatever stale
        # CSS was there before.
        if not self._build_succeeded():
            self.fail(
                f"build did not succeed (exit {self.build_returncode}); "
                f"cannot check theme/site CSS parity: {self.build_stderr}"
            )

        theme_css = ROOT / "theme" / "style.css"
        site_css = ROOT / "site" / "style.css"
        self.assertTrue(theme_css.exists(), f"missing source: {theme_css}")
        self.assertTrue(site_css.exists(), f"missing build output: {site_css}")
        # Bytes comparison (not text) because shutil.copy2 preserves
        # the file bytes exactly; a text comparison could fail on
        # encoding normalization even when the files are functionally
        # identical.
        self.assertEqual(
            theme_css.read_bytes(),
            site_css.read_bytes(),
            f"{theme_css.relative_to(ROOT)} and {site_css.relative_to(ROOT)} "
            "differ after the build (the copy step is broken or stale)",
        )


if __name__ == "__main__":
    unittest.main()
