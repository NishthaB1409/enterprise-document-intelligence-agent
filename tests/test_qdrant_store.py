"""The real Qdrant adapter, against a real Qdrant engine.

Embedded mode runs the engine in-process against a directory, so this exercises
`QdrantVectorStore` itself — payload round-trip, filtered delete, ranking, the
missing-collection path — with no container and no network. The in-memory fake
used everywhere else proves the pipeline; this proves the adapter under it.
"""

import pytest
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.ingest.chunking import Chunk
from app.ingest.sparse import SparseVector
from app.services import build_qdrant_client
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.vectorstore.store import CollectionSchemaError

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


def _orthogonal_to(vector: list[float]) -> list[float]:
    """A unit vector with zero cosine similarity to `vector`.

    Used to prove a hit came from the sparse arm: if the dense arm scores the
    chunk at zero and it still comes back, BM25 is what found it.
    """
    axis = next(i for i, value in enumerate(vector) if not value)
    orthogonal = [0.0] * DIMENSION
    orthogonal[axis] = 1.0
    return orthogonal


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


class TestHybridSearch:
    """Fusion against the real engine.

    The in-memory fake re-implements RRF so the pipeline can be tested without
    Qdrant; these hold the real thing to the same *ordering* guarantees, so the
    two cannot drift apart unnoticed. Scores are not compared — Qdrant's fused
    scores are on its own scale and asserting them would pin an implementation
    detail of the engine.
    """

    def test_a_sparse_only_match_is_retrievable(self, store: QdrantVectorStore):
        """The BM25 half round-trips: a chunk whose dense vector is orthogonal
        to the query still comes back when its terms match."""
        store.ensure_collection(DIMENSION)
        store.upsert(
            [_chunk(1)],
            [_vector(1)],
            [SparseVector(indices=[42], values=[3.0])],
            source="contract.pdf",
        )

        hits = store.hybrid_search(
            _orthogonal_to(_vector(1)),
            SparseVector(indices=[42], values=[1.0]),
            top_k=5,
            candidates=10,
        )

        assert [hit.chunk.index for hit in hits] == [1]

    def test_fusion_promotes_the_chunk_both_arms_rank(self, store: QdrantVectorStore):
        store.ensure_collection(DIMENSION)
        # chunk 1: strong on both arms. chunk 2: wins dense outright, no terms.
        store.upsert(
            [_chunk(1), _chunk(2)],
            [_vector(1), _vector(2)],
            [
                SparseVector(indices=[42], values=[5.0]),
                SparseVector(indices=[99], values=[5.0]),
            ],
            source="contract.pdf",
        )

        hits = store.hybrid_search(
            _vector(2),
            SparseVector(indices=[42], values=[1.0]),
            top_k=2,
            candidates=10,
        )

        assert [hit.chunk.index for hit in hits] == [1, 2]

    def test_a_hybrid_hit_carries_the_same_payload_a_dense_hit_does(
        self, store: QdrantVectorStore
    ):
        """A citation assembled from a fused result must be as complete as one
        from a dense result — the fusion path reads the payload separately."""
        store.ensure_collection(DIMENSION)
        store.upsert(
            [_chunk(1, page=7)],
            [_vector(1)],
            [SparseVector(indices=[42], values=[1.0])],
            source="contract.pdf",
        )

        (hit,) = store.hybrid_search(
            _vector(1), SparseVector(indices=[42], values=[1.0]), top_k=5, candidates=10
        )

        assert hit.source == "contract.pdf"
        assert hit.chunk.page == 7
        assert hit.chunk.text == "Clause 1 of the agreement."
        assert (hit.chunk.char_start, hit.chunk.char_end) == (10, 15)

    def test_an_empty_query_vector_degrades_to_dense_rather_than_failing(
        self, store: QdrantVectorStore
    ):
        store.ensure_collection(DIMENSION)
        store.upsert(
            [_chunk(1)],
            [_vector(1)],
            [SparseVector(indices=[42], values=[1.0])],
            source="contract.pdf",
        )

        hits = store.hybrid_search(
            _vector(1), SparseVector(indices=[], values=[]), top_k=5, candidates=10
        )

        assert [hit.chunk.index for hit in hits] == [1]

    def test_hybrid_search_before_anything_is_ingested_returns_nothing(
        self, store: QdrantVectorStore
    ):
        assert (
            store.hybrid_search(
                _vector(1), SparseVector(indices=[1], values=[1.0]), top_k=5, candidates=10
            )
            == []
        )

    def test_a_mismatched_sparse_vector_count_is_caught(self, store: QdrantVectorStore):
        store.ensure_collection(DIMENSION)

        with pytest.raises(ValueError, match="correspond"):
            store.upsert(
                [_chunk(1), _chunk(2)],
                [_vector(1), _vector(2)],
                [SparseVector(indices=[1], values=[1.0])],
                source="contract.pdf",
            )


def test_a_collection_predating_hybrid_retrieval_is_reported_with_the_remedy(tmp_path):
    """Qdrant will not add a sparse index to an existing collection, and a
    fusion query against one fails with a message that says nothing about how
    it got that way. Ingestion has to catch it and say what to do."""
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        # Exactly what the pre-phase-2 code created: dense vectors, no sparse.
        client.create_collection(
            collection_name="legacy",
            vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE),
        )

        store = QdrantVectorStore(client=client, collection="legacy")

        with pytest.raises(CollectionSchemaError, match="re-ingest"):
            store.ensure_collection(DIMENSION)
    finally:
        client.close()


def test_a_dense_only_deployment_can_serve_a_pre_hybrid_collection(tmp_path):
    """The sparse index is only required by the half of the system that queries
    it. A deployment running RETRIEVAL_MODE=dense against a collection created
    before phase 2 must keep working — forcing a re-ingest for an index it will
    never read would be a migration with nothing on the other side of it."""
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        client.create_collection(
            collection_name="legacy",
            vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE),
        )

        store = QdrantVectorStore(
            client=client, collection="legacy", requires_sparse=False
        )
        store.ensure_collection(DIMENSION)
        store.upsert([_chunk(1)], [_vector(1)], source="contract.pdf")

        assert [hit.chunk.index for hit in store.search(_vector(1), top_k=5)] == [1]
    finally:
        client.close()
