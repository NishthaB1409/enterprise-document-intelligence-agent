"""Ingestion: PDF bytes in, citable chunks in the store."""

import pytest

from app.ingest.parser import UnreadableDocument
from app.ingest.pipeline import document_id, ingest_pdf
from tests.fakes import InMemoryVectorStore, StubEmbedder
from tests.pdfs import build_pdf

CONTRACT = [
    "Section 1. Term.\nThis agreement begins on 1 January 2026.",
    "Section 2. Termination.\nEither party may terminate on thirty days notice.",
]


def _ingest(data: bytes, store: InMemoryVectorStore, **overrides):
    params = {"chunk_size_words": 40, "chunk_overlap_words": 5} | overrides
    return ingest_pdf(
        data, filename="contract.pdf", embedder=StubEmbedder(), store=store, **params
    )


def test_indexes_every_page_and_keeps_its_number():
    store = InMemoryVectorStore()

    result = _ingest(build_pdf(CONTRACT), store)

    assert result.pages == 2
    assert result.chunks == len(store.points)
    assert result.source == "contract.pdf"

    pages = {point.chunk.page for point in store.points.values()}
    assert pages == {1, 2}

    # The citation has to survive the round trip, not just the parse.
    termination = next(
        point.chunk for point in store.points.values() if "thirty days" in point.chunk.text
    )
    assert termination.page == 2
    assert termination.doc_id == result.doc_id


def test_collection_is_created_before_anything_is_written():
    store = InMemoryVectorStore()

    _ingest(build_pdf(CONTRACT), store)

    # Sized from the embedder rather than a constant: getting this wrong makes
    # every upsert fail against a real Qdrant.
    assert store.collection_dimension == StubEmbedder().dimension


def test_same_bytes_produce_the_same_document_id():
    data = build_pdf(CONTRACT)

    # Different filename, same content — one document, not two.
    assert document_id(data) == document_id(build_pdf(CONTRACT))
    assert document_id(data) != document_id(build_pdf(["Something else entirely."]))


def test_reingesting_replaces_the_previous_chunks():
    store = InMemoryVectorStore()
    # One long page, so the chunk size actually decides how many chunks there
    # are. Chunks never span a page, so short pages would give the same count
    # at any size.
    data = build_pdf(
        [" ".join(f"Clause {n} states an obligation of the parties." for n in range(1, 21))]
    )

    first = _ingest(data, store)
    # Bigger chunks means fewer of them. Without the delete-then-upsert, the
    # leftovers from the first run would stay searchable and the same passage
    # would be citable twice under two ids.
    second = _ingest(data, store, chunk_size_words=400)

    assert second.doc_id == first.doc_id
    assert second.chunks < first.chunks
    assert len(store.points) == second.chunks


def test_reingesting_leaves_other_documents_alone():
    store = InMemoryVectorStore()
    other = ingest_pdf(
        build_pdf(["An unrelated policy document."]),
        filename="policy.pdf",
        embedder=StubEmbedder(),
        store=store,
        chunk_size_words=40,
        chunk_overlap_words=5,
    )

    _ingest(build_pdf(CONTRACT), store)
    _ingest(build_pdf(CONTRACT), store)

    surviving = {point.chunk.doc_id for point in store.points.values()}
    assert other.doc_id in surviving


def test_a_scan_is_rejected_rather_than_silently_indexed():
    store = InMemoryVectorStore()

    # A PDF whose pages carry no text layer — what a scanned contract looks like.
    with pytest.raises(UnreadableDocument, match="OCR"):
        _ingest(build_pdf(["", ""]), store)

    assert store.points == {}


def test_a_non_pdf_upload_is_rejected():
    with pytest.raises(UnreadableDocument):
        _ingest(b"this is not a PDF at all", InMemoryVectorStore())
