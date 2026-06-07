#!/usr/bin/env python3
"""CLI hybrid search (reuses server-style lazy-loaded HybridSearch)."""

import json
import sys
import time

from server import get_hybrid


def search(query: str, top_k: int = 5):
    t0 = time.time()
    hybrid = get_hybrid()
    results, timing = hybrid.search(query, top_k=top_k)
    timing["total_ms"] = round((time.time() - t0) * 1000, 1)
    return {
        "query": query,
        "results": results,
        "timing": timing,
        "total_chunks": hybrid.bm25.total_docs,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question"')
        print('       python query.py --json "your question"')
        sys.exit(1)

    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv
    query_text = " ".join(args)

    result = search(query_text)
    if as_json:
        print(json.dumps(result, indent=2))
        return

    print(f"\n{'=' * 60}")
    print(f"🔍 Query: {result['query']}")
    print(f"📊 {result['total_chunks']} chunks searched in {result['timing']['total_ms']}ms")
    print(f"{'=' * 60}\n")

    for i, r in enumerate(result["results"], 1):
        print(f"─── Result #{i} ───")
        print(f"📌 {r['title']}")
        print(f"🏷️  {r['category']}")
        print(f"🔗 {r['source_url']}")
        vs = r.get("vector_score")
        if vs is not None:
            print(f"📊 Vector: {vs:.3f} | RRF: {r['rrf_score']:.3f}")
        else:
            print(f"📊 RRF: {r['rrf_score']:.3f}")
        print(f"📝 {r['content'][:400]}...")
        print()


if __name__ == "__main__":
    main()
