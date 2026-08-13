"""The real Qdrant adapter, against a real Qdrant engine.

Embedded mode runs the engine in-process against a directory, so this exercises
`QdrantVectorStore` itself — payload round-trip, filtered delete, ranking, the
missing-collection path — with no container and no network. The in-memory fake
used everywhere else proves the pipeline; this proves the adapter under it.
"""

import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.ingest.chunking import Chunk
from app.services import build_qdrant_client
from app.vectorstore.qdrant_store import QdrantVectorStore

DIMENSION = 4


@pytest.fixture
def store(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    yield QdrantVectorStore(client=client, collection="test-documents")
    # Embedded mode holds an exclusive lock on the directory; without this the
    # next test in the same process cannot open its own.
    client.close()


def _chunk(index: int, doc_id: str = "doc-1", page: int = 1) -> Chunk:
    return Chunk(
        id=f"0000000{index}-0000-0000-0000-00000000000{index}",
        doc_id=doc_id,
        index=index,
        page=page,
        text=f"Clause {index} of the agreement.",
        char_start=index * 10,
        char_end=index * 10 + 5,
    )


def _vector(index: int) -> list[float]:
    vector = [0.0] * DIMENSION
    vector[index % DIMENSION] = 1.0
    return vector


def test_a_hit_carries_everything_a_citation_needs(store: QdrantVectorStore):
    store.ensure_collection(DIMENSION)
    store.upsert([_chunk(1, page=7)], [_vector(1)], source="contract.pdf")

    (hit,) = store.search(_vector(1), top_k=3)

    # The payload has to survive the round trip intact — a citation assembled
    # from a partial payload would point at the wrong place, silently.
    assert hit.source == "contract.pdf"
    assert hit.chunk.page == 7
    assert hit.chunk.doc_id == "doc-1"
    assert hit.chunk.text == "Clause 1 of the agreement."
    assert (hit.chunk.char_start, hit.chunk.char_end) == (10, 15)
    assert hit.score == pytest.approx(1.0)


def test_results_come_back_ranked(store: QdrantVectorStore):
    store.ensure_collection(DIMENSION)
    store.upsert([_chunk(1), _chunk(2)], [_vector(1), _vector(2)], source="contract.pdf")

    hits = store.search(_vector(2), top_k=2)

    assert hits[0].chunk.index == 2
    assert hits[0].score > hits[1].score


def test_top_k_limits_the_result_set(store: QdrantVectorStore):
    store.ensure_collection(DIMENSION)
    chunks = [_chunk(i) for i in range(1, 4)]
    store.upsert(chunks, [_vector(i) for i in range(1, 4)], source="contract.pdf")

    assert len(store.search(_vector(1), top_k=1)) == 1


def test_upserting_the_same_id_replaces_rather_than_duplicates(store: QdrantVectorStore):
    store.ensure_collection(DIMENSION)
    store.upsert([_chunk(1)], [_vector(1)], source="contract.pdf")
    # Same chunk id, new filename — what re-ingesting a renamed file does.
    store.upsert([_chunk(1)], [_vector(1)], source="contract-v2.pdf")

    hits = store.search(_vector(1), top_k=10)

    assert len(hits) == 1
    assert hits[0].source == "contract-v2.pdf"


def test_deleting_a_document_leaves_the_others(store: QdrantVectorStore):
    store.ensure_collection(DIMENSION)
    store.upsert([_chunk(1, doc_id="doc-1")], [_vector(1)], source="a.pdf")
    store.upsert([_chunk(2, doc_id="doc-2")], [_vector(2)], source="b.pdf")

    store.delete_document("doc-1")

    remaining = store.search(_vector(2), top_k=10)
    assert [hit.chunk.doc_id for hit in remaining] == ["doc-2"]


def test_deleting_an_unknown_document_is_not_an_error(store: QdrantVectorStore):
    store.ensure_collection(DIMENSION)

    store.delete_document("never-ingested")


def test_ensure_collection_is_idempotent(store: QdrantVectorStore):
    store.ensure_collection(DIMENSION)
    store.upsert([_chunk(1)], [_vector(1)], source="contract.pdf")

    # Runs again on every ingest, and must not wipe what is already indexed.
    store.ensure_collection(DIMENSION)

    assert len(store.search(_vector(1), top_k=10)) == 1


def test_searching_before_anything_is_ingested_returns_nothing(store: QdrantVectorStore):
    # No ensure_collection: this is a /query that arrives before the first
    # /ingest, which is a legitimate state and not a 500.
    assert store.search(_vector(1), top_k=5) == []


def test_a_mismatched_vector_count_is_caught_before_it_reaches_the_engine(
    store: QdrantVectorStore,
):
    store.ensure_collection(DIMENSION)

    with pytest.raises(ValueError, match="correspond"):
        store.upsert([_chunk(1), _chunk(2)], [_vector(1)], source="contract.pdf")


def test_a_configured_path_selects_embedded_mode(tmp_path):
    settings = Settings(_env_file=None, qdrant_path=str(tmp_path / "embedded"))

    client = build_qdrant_client(settings)
    try:
        # Reaching a real collection listing proves the engine started against
        # the directory rather than trying to reach a server.
        assert client.get_collections().collections == []
    finally:
        client.close()
