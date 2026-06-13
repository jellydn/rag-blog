"""Unit tests for markdown chunking (no ML deps)."""

import unittest

from chunking import CHUNK_SIZE, Document, MarkdownChunker
from rag_pipeline import BM25Index


class TestMarkdownChunker(unittest.TestCase):
    def test_skips_headings_inside_code_fence(self):
        text = """# Title

```python
## not a real heading
print("ok")
```

## Real section

Body here with enough text to keep the chunk valid for ingestion rules.
"""
        doc = Document(
            id="test-doc",
            title="Title",
            content=text,
            source_url="https://example.com/notes/test-doc",
            category="TIL",
        )
        chunks = MarkdownChunker().chunk(doc)
        self.assertGreaterEqual(len(chunks), 1)
        combined = "\n".join(c["content"] for c in chunks)
        self.assertIn("```python", combined)
        self.assertIn("## not a real heading", combined)
        self.assertIn("## Real section", combined)

    def test_chunk_id_on_records(self):
        doc = Document(
            id="slug-a",
            title="T",
            content="# Title\n\nEnough text here to pass the minimum chunk length gate easily.",
            source_url="https://example.com/notes/slug-a",
            category="TIL",
        )
        chunks = MarkdownChunker().chunk(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["id"], "slug-a:0")

    def test_total_chunks_matches_kept_segments(self):
        doc = Document(
            id="multi",
            title="Multi",
            content="# A\n\n" + ("word " * 80) + "\n\n## B\n\n" + ("other " * 80),
            source_url="https://example.com/notes/multi",
            category="Guide",
        )
        chunks = MarkdownChunker().chunk(doc)
        if chunks:
            self.assertEqual(chunks[0]["total_chunks"], len(chunks))
            for i, c in enumerate(chunks):
                self.assertEqual(c["chunk_index"], i)

    def test_long_section_splits_under_size(self):
        body = "paragraph.\n\n" * 200
        doc = Document(
            id="long",
            title="Long",
            content=f"# Long\n\n## Section\n\n{body}",
            source_url="https://example.com/notes/long",
            category="TIL",
        )
        chunks = MarkdownChunker().chunk(doc)
        for c in chunks:
            self.assertGreaterEqual(len(c["content"]), 20)
            if len(c["content"]) > CHUNK_SIZE * 2:
                self.fail(f"chunk unexpectedly large: {len(c['content'])}")


class TestBM25MaxTf(unittest.TestCase):
    """The new BM25Index.max_tf(token) helper used by the suppression playground."""

    def _make_index(self, docs):
        bm25 = BM25Index()
        bm25.add_documents(doc_ids=[f"d{i}" for i in range(len(docs))], texts=docs)
        return bm25

    def test_max_tf_zero_for_missing_token(self):
        bm25 = self._make_index(["hello world", "foo bar"])
        self.assertEqual(bm25.max_tf("xyzzy"), 0)

    def test_max_tf_returns_highest_occurrence_count(self):
        # 'neovim' appears 1x in doc0, 3x in doc1, 2x in doc2 -> max = 3
        bm25 = self._make_index([
            "neovim is cool",
            "neovim neovim neovim folding",
            "two neovim mentions here",
        ])
        self.assertEqual(bm25.max_tf("neovim"), 3)

    def test_max_tf_handles_empty_index(self):
        bm25 = BM25Index()  # never add_documents
        self.assertEqual(bm25.max_tf("anything"), 0)


if __name__ == "__main__":
    unittest.main()
