"""Unit tests for markdown chunking (no ML deps)."""

import unittest

from chunking import CHUNK_SIZE, Document, MarkdownChunker


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


if __name__ == "__main__":
    unittest.main()