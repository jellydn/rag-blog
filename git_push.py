#!/usr/bin/env python3
"""Push repo contents to GitHub using the Git Data API (no git CLI needed)."""

import base64
import json
import os
import sys
import urllib.error
import urllib.request

OWNER = "jellydn"
REPO = "rag-blog"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
BASE = f"https://api.github.com/repos/{OWNER}/{REPO}"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/vnd.github+json",
}

FILES = [
    ".gitignore",
    "DAY1_NOTES.md",
    "README.md",
    "chunking.py",
    "config.py",
    "requirements.txt",
    "query.py",
    "rag_pipeline.py",
    "scrape_content.py",
    "server.py",
    "tests/test_chunker.py",
]


def api(method, path, data=None):
    url = f"{BASE}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"  ERROR {method} {path}: {e.code} - {err[:200]}")
        sys.exit(1)


def main():
    # 1. Create blobs for each file
    print("Creating blobs...")
    blob_shas = {}
    for fname in FILES:
        with open(f"/opt/data/rag-blog/{fname}", "rb") as f:
            content = f.read()
        b64 = base64.b64encode(content).decode()
        result = api(
            "POST",
            "/git/blobs",
            {
                "content": b64,
                "encoding": "base64",
            },
        )
        blob_shas[fname] = result["sha"]
        print(f"  {fname} → {result['sha'][:12]}...")

    # 2. Create tree
    print("Creating tree...")
    tree_items = [
        {
            "path": fname,
            "mode": "100644",
            "type": "blob",
            "sha": blob_shas[fname],
        }
        for fname in FILES
    ]
    tree = api("POST", "/git/trees", {"tree": tree_items})
    tree_sha = tree["sha"]
    print(f"  Tree: {tree_sha[:12]}...")

    # 3. Create commit
    print("Creating commit...")
    commit = api(
        "POST",
        "/git/commits",
        {
            "message": "Day 1: Production RAG engine for Productsway",
            "tree": tree_sha,
            "parents": [],
        },
    )
    commit_sha = commit["sha"]
    print(f"  Commit: {commit_sha[:12]}...")

    # 4. Create main branch
    print("Creating main branch...")
    api(
        "POST",
        "/git/refs",
        {
            "ref": "refs/heads/main",
            "sha": commit_sha,
        },
    )
    print("  ✅ main branch created")

    # 5. Create day-1 branch for the PR
    print("Creating day-1 branch...")
    api(
        "POST",
        "/git/refs",
        {
            "ref": "refs/heads/day-1",
            "sha": commit_sha,
        },
    )
    print("  ✅ day-1 branch created")

    # 6. Create draft PR
    print("Creating draft PR...")
    with open("/opt/data/rag-blog/DAY1_NOTES.md", encoding="utf-8") as f:
        pr_body = f.read()
    pr = api(
        "POST",
        "/pulls",
        {
            "title": "Day 1: Production RAG Systems",
            "head": "day-1",
            "base": "main",
            "body": pr_body,
            "draft": True,
        },
    )
    print(f"\n🎉 PR created: {pr['html_url']}")

    # 7. Configure repo defaults
    api(
        "PATCH",
        "",
        {
            "default_branch": "main",
            "has_issues": True,
            "has_projects": True,
        },
    )
    print("  Repo configured ✓")


if __name__ == "__main__":
    main()
