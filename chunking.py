"""Markdown-aware chunking (stdlib only — safe to import in unit tests)."""

import re
from dataclasses import dataclass, field
from typing import List

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}:{chunk_index}"


def parse_chunk_id(chunk_id_str: str) -> tuple[str, int]:
    if ":" in chunk_id_str:
        doc, idx = chunk_id_str.rsplit(":", 1)
        return doc, int(idx)
    return chunk_id_str, 0


@dataclass
class Document:
    """A raw document scraped from the site."""

    id: str
    title: str
    content: str
    source_url: str
    category: str
    tags: List[str] = field(default_factory=list)


class MarkdownChunker:
    """
    Smart chunker that respects markdown structure.
    - Keeps headings with their content
    - Splits on ## or ### boundaries when content is too long
    - Never splits mid-code-block
    - Preserves context overlap
    """

    SECTION_HEADING_RE = re.compile(r"^##+\s+", re.MULTILINE)
    CODE_BLOCK_RE = re.compile(r"^```", re.MULTILINE)

    def chunk(self, doc: Document) -> List[dict]:
        raw = doc.content
        chunks_raw = self._split_by_headings(raw)

        kept = []
        for chunk_text in chunks_raw:
            chunk_text = chunk_text.strip()
            if chunk_text and len(chunk_text) >= 20:
                kept.append(chunk_text)

        result = []
        total = len(kept)
        for i, chunk_text in enumerate(kept):
            first_line = chunk_text.split("\n")[0]
            sub_title = (
                first_line.lstrip("#").strip()
                if first_line.startswith("#")
                else doc.title
            )
            row = {
                "title": sub_title,
                "content": chunk_text,
                "doc_id": doc.id,
                "source_url": doc.source_url,
                "category": doc.category,
                "chunk_index": i,
                "total_chunks": total,
            }
            row["id"] = chunk_id(doc.id, i)
            result.append(row)

        return result

    def _split_by_headings(self, text: str) -> List[str]:
        code_blocks = [m.span() for m in self.CODE_BLOCK_RE.finditer(text)]

        def is_in_code_block(pos: int) -> bool:
            return sum(1 for start, end in code_blocks if start < pos) % 2 == 1

        heading_matches = [
            m
            for m in self.SECTION_HEADING_RE.finditer(text)
            if not is_in_code_block(m.start())
        ]

        if not heading_matches:
            return self._split_by_size(text)

        sections = []
        for i, match in enumerate(heading_matches):
            start = match.start()
            end = (
                heading_matches[i + 1].start()
                if i + 1 < len(heading_matches)
                else len(text)
            )
            section = text[start:end].strip()
            if section:
                if len(section) > CHUNK_SIZE * 1.5:
                    sections.extend(self._split_by_size(section))
                else:
                    sections.append(section)

        if heading_matches and heading_matches[0].start() > 0:
            pre = text[: heading_matches[0].start()].strip()
            if pre:
                sections.insert(0, pre)

        return sections if sections else [text]

    def _split_by_size(self, text: str) -> List[str]:
        chunks = []
        lines = text.split("\n")
        current = []
        current_len = 0
        in_code = False

        for line in lines:
            if self.CODE_BLOCK_RE.match(line.strip()):
                in_code = not in_code

            current.append(line)
            current_len += len(line) + 1

            if current_len >= CHUNK_SIZE and not in_code:
                chunks.append("\n".join(current))

                overlap_chars = 0
                overlap_lines = []
                for ln in reversed(current):
                    if overlap_chars >= CHUNK_OVERLAP:
                        break
                    overlap_lines.insert(0, ln)
                    overlap_chars += len(ln) + 1

                current = overlap_lines
                current_len = overlap_chars
                in_code = False

        if current:
            remaining = "\n".join(current).strip()
            if remaining:
                chunks.append(remaining)

        return chunks