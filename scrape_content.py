#!/usr/bin/env python3
"""Scrape all notes/blog posts from productsway.com for RAG ingestion."""

import os
import re
import sys
from html.parser import HTMLParser

from config import CONTENT_DIR, ensure_data_dirs

ensure_data_dirs()
OUTPUT_DIR = str(CONTENT_DIR)

# All note URLs from productsway.com/notes
NOTE_SLUGS = [
    ("til-43-delete-all-remote-branches-except-main", "TIL"),
    ("til-42-cherry-pick-from-pull-request", "TIL"),
    ("til-41-how-to-deploy-old-documentation-with-mkdocs", "TIL"),
    ("til-40-how-to-set-up-folding-in-neovim", "TIL"),
    ("til-39-how-to-fix-not-a-test-file-error-with-vim-test", "TIL"),
    ("til-38-cherry-pick-git-merge-from-cli", "TIL"),
    ("til-37-how-to-fix-session-id-unknown-with-socket.io", "TIL"),
    ("til-36-github-copilot-with-nvchad-nvim", "TIL"),
    ("til-35-fix-blank-image-with-html2pdf", "TIL"),
    ("til-34-remove-some-files-from-last-commit", "TIL"),
    ("til-33-auto-merge-your-pull-request-on-github-with-renovate-bot", "TIL"),
    ("til-32-list-all-express-routes", "TIL"),
    ("til-31-fix-build-error-on-oclif-cli-for-typescript-4.8.3", "TIL"),
    ("til-30-merge-to-videos-from-cli-with-ffmpeg", "TIL"),
    ("til-29-github-action-version", "TIL"),
    ("til-28-decode-receipt-logs-with-ethers", "TIL"),
    ("til-27-install-homebrew-manually", "TIL"),
    ("til-26-how-to-connect-to-redis-on-aws-amazon-elasticache", "TIL"),
    ("til-25-yarn-global-upgrade-all-packages", "TIL"),
    ("til-24-workaround-for-trpc-fastify-adatper-cors-policy", "TIL"),
    ("til-23-restart-mac-os-x-coreaudio-daemon", "TIL"),
    ("simply-the-flutter-flow-with-makefile", "Guide"),
    ("til-22-enum-type-with-postgresql-and-sqlx", "TIL"),
    ("til-21-how-to-fix-postgresql-duplicate-key-violates-out-of-sync", "TIL"),
    ("til-20-how-to-fix-unsupported-scan-storing-driver.value-type-uint8-into-type", "TIL"),
    ("til-19-how-to-fix-android-studio-missing-essential-plugin-org.jetbrains.android", "TIL"),
    ("til-18-deploy-to-heroku-from-sub-directory", "TIL"),
    ("reload-page-from-iframe-with-cross-domain-support", "Guide"),
    ("how-to-use-custom-element-with-nextjs-react", "Guide"),
    ("til-17-call-eth_sync", "TIL"),
    ("til-14-4-simple-steps-for-backup-restore-wordpress-website", "TIL"),
    ("til-15-fix-duplicate-identifier-librarymanagedattributes", "TIL"),
    ("til-16-revert-to-1st-commit-with-git-command", "TIL"),
    ("truffle-cli-exec-with-arguments", "Guide"),
    ("new-web-app-cli", "Guide"),
    (
        "til-13-how-to-fix-refusing-to-allow-an-oauth-app-to-create-or-update-workflow-without-workflow-scope",
        "TIL",
    ),
    ("react-hook-use-wait-for-transaction-hash", "Guide"),
    ("til-11-mac-osx-open-file-from-anywhere", "TIL"),
    ("til-12-fix-the-ssh-issue-with-droplet-on-digital-ocean", "TIL"),
    ("til-10-add-user-define-function-to-typeorm-entity", "TIL"),
    ("til-9-delete-all-databases-on-mongo-db-on-local", "TIL"),
    ("til-8-jest-testing-with-absolute-import", "TIL"),
    ("til-7-use-tsconfig.json-for-ts-node", "TIL"),
    ("til-6-delete-all-users-from-aws-cognito", "TIL"),
    ("til-5-delete-all-local-branches-except-master", "TIL"),
    ("til-4-trigger-github-action-base-on-the-comment", "TIL"),
    ("til-3-rename-all-js-to-ts-files", "TIL"),
    ("create-and-apply-git-patch-from-a-commit-hash", "TIL"),
    ("git-rebase-interactive", "TIL"),
    ("welcome-to-my-experimental-world", "Guide"),
]


class ContentExtractor(HTMLParser):
    """Simple HTML content extractor that captures text and code blocks."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._in_code = False
        self._in_pre = False
        self._skip = False
        self._current_tag = ""
        self._headings = set(["h1", "h2", "h3", "h4", "h5", "h6"])

    def handle_starttag(self, tag, attrs):
        self._current_tag = tag
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True
        if tag == "code" or tag == "pre":
            if tag == "pre":
                self._in_pre = True
                self.text_parts.append("\n```\n")
            elif tag == "code" and not self._in_pre:
                self._in_code = True
                self.text_parts.append("`")
        if tag in self._headings:
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
        if tag == "pre":
            self._in_pre = False
            self.text_parts.append("\n```\n")
        elif tag == "code" and not self._in_pre:
            self._in_code = False
            self.text_parts.append("`")
        if tag in self._headings:
            self.text_parts.append("\n")
        if tag in ("p", "li", "div"):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if data.strip():
            self.text_parts.append(data.strip())

    def get_text(self):
        text = "".join(self.text_parts)
        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def fetch_page(url):
    """Fetch a page using urllib."""
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  Error fetching {url}: {e}", file=sys.stderr)
        return None


def extract_main_content(html):
    """Extract the main article content from the HTML page."""
    # Try to find the content inside Next.js data structures or main tag
    extractor = ContentExtractor()

    # Look for article/main content
    main_match = re.search(r"<main[^>]*>(.*?)</main>", html, re.DOTALL)
    if main_match:
        main_html = main_match.group(1)
        # Remove sidebar/nav elements
        main_html = re.sub(r"<nav[^>]*>.*?</nav>", "", main_html, flags=re.DOTALL)
        extractor.feed(main_html)
        text = extractor.get_text()
        if len(text) > 100:
            return text

    # Fallback: try to find article content
    article_match = re.search(r"<article[^>]*>(.*?)</article>", html, re.DOTALL)
    if article_match:
        extractor = ContentExtractor()
        extractor.feed(article_match.group(1))
        text = extractor.get_text()
        if len(text) > 100:
            return text

    # Last resort: extract everything
    body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL)
    if body_match:
        extractor.feed(body_match.group(1))
        text = extractor.get_text()
        # Filter to reasonable length
        lines = [line for line in text.split("\n") if len(line) > 2]
        return "\n".join(lines[:200])

    return extractor.get_text()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Also fetch the homepage for the about section
    print("Fetching homepage...")
    html = fetch_page("https://productsway.com/")
    if html:
        text = extract_main_content(html)
        with open(f"{OUTPUT_DIR}/homepage.md", "w", encoding="utf-8") as f:
            f.write(f"# About Dung Huynh Duc\n\n{text}")
        print(f"  Saved homepage ({len(text)} chars)")

    # Fetch each note
    for slug, _note_type in NOTE_SLUGS:
        url = f"https://productsway.com/notes/{slug}"
        filepath = f"{OUTPUT_DIR}/{slug}.md"

        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size > 200:
                print(f"  Skipping (exists): {slug} ({size} bytes)")
                continue

        print(f"  Fetching: {slug}...")
        html = fetch_page(url)
        if not html:
            continue

        text = extract_main_content(html)
        if not text or len(text) < 50:
            print(f"    Too short ({len(text) if text else 0} chars), skipping")
            continue

        # Prepend title
        title = slug.replace("-", " ").title()
        content = f"# {title}\n\n{text}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"    Saved ({len(content)} chars)")


if __name__ == "__main__":
    main()
