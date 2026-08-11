"""Pages to chunks — the unit that gets embedded, retrieved, and cited.

A chunk is the smallest thing an answer can point at, so it carries everything
needed to prove where a claim came from: the document, the page, and the span
within that page.
"""

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from llama_index.core.node_parser import SentenceSplitter

from app.ingest.parser import Page

# A fixed namespace makes chunk ids deterministic: re-ingesting the same bytes
# regenerates the same ids, so an upsert overwrites in place instead of
# accumulating duplicate copies of the document.
_CHUNK_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    # Position within the document, so neighbouring chunks can be stitched back
    # together when an answer needs more context than one chunk holds.
    index: int
    page: int
    text: str
    # Offsets into the *normalized* page text (what this service stores and
    # displays), letting a UI highlight the exact span behind a citation.
    char_start: int
    char_end: int


def _word_tokenizer(text: str) -> list[str]:
    """Whitespace words, standing in for BPE tokens.

    The splitter's default tokenizer fetches its vocabulary over the network on
    first use. Ingestion should not depend on that, and the exact token count
    does not matter here — only that chunks land comfortably under the embedding
    model's truncation limit. See `Settings.chunk_size_words`.
    """
    return text.split()


def build_splitter(chunk_size_words: int, chunk_overlap_words: int) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=chunk_size_words,
        chunk_overlap=chunk_overlap_words,
        tokenizer=_word_tokenizer,
    )


def chunk_id(doc_id: str, index: int) -> str:
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{doc_id}:{index}"))


def chunk_pages(
    pages: Iterable[Page],
    doc_id: str,
    splitter: SentenceSplitter,
    *,
    id_factory: Callable[[str, int], str] = chunk_id,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for page in pages:
        # A cursor rather than a plain `find`: a boilerplate line repeated on the
        # page (a header, a clause number) would otherwise match its first
        # occurrence every time and report the same span for several chunks.
        cursor = 0
        for text in splitter.split_text(page.text):
            if not (text := text.strip()):
                continue

            start = page.text.find(text, cursor)
            if start == -1:
                # The splitter normalized whitespace somewhere inside the chunk,
                # so it no longer matches the page verbatim. The chunk is still
                # perfectly citable by page — only the span highlight is lost.
                start, end = -1, -1
            else:
                end = start + len(text)
                cursor = start + 1

            chunks.append(
                Chunk(
                    id=id_factory(doc_id, len(chunks)),
                    doc_id=doc_id,
                    index=len(chunks),
                    page=page.number,
                    text=text,
                    char_start=start,
                    char_end=end,
                )
            )

    return chunks
