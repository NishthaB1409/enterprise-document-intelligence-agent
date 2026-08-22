"""Qdrant-backed implementation of `VectorStore`.

The chunk is stored whole in the point payload rather than kept in a second
database keyed by id. One round trip returns everything a citation needs, and
there is no way for the text an answer quotes to drift out of sync with the
vector that retrieved it.

Each point carries two vectors: the default (unnamed) dense vector, and a named
`bm25` sparse one. Keeping the dense vector unnamed is what lets the dense-only
`search` path stay exactly as it was — naming it would have been tidier and
would have silently invalidated every collection already on disk.
"""

import logging
import re
from collections.abc import Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.ingest.chunking import Chunk
from app.ingest.sparse import SparseVector
from app.vectorstore.store import CollectionSchemaError, ScoredChunk

logger = logging.getLogger(__name__)

# Cosine, because the embedding models this service uses (bge, e5) are trained
# with a cosine objective and their vectors are normalized.
_DISTANCE = models.Distance.COSINE

# The sparse vector's name in the collection schema. Not configurable: it is an
# internal detail of this adapter, and changing it would orphan every point
# already indexed under the old name.
SPARSE_VECTOR = "bm25"


class QdrantVectorStore:
    def __init__(
        self, client: QdrantClient, collection: str, *, requires_sparse: bool = True
    ) -> None:
        self._client = client
        self._collection = collection
        # Whether a missing BM25 index is a fault. A dense-only deployment can
        # serve a collection that predates hybrid retrieval perfectly well, and
        # blocking its ingests over an index it will never query would be a
        # migration it has no reason to perform.
        self._requires_sparse = requires_sparse
        # Set once the collection is known to exist and to be correctly shaped,
        # so the common path (every ingest after the first) costs no round trip.
        self._ready = False

    def ensure_collection(self, dimension: int) -> None:
        if self._ready:
            return

        if self._client.collection_exists(self._collection):
            self._verify_schema()
        else:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=dimension, distance=_DISTANCE),
                sparse_vectors_config={
                    # IDF is computed by the engine across the whole collection.
                    # See the note in `app.ingest.sparse` about why it cannot be
                    # baked into the vectors at ingest time.
                    SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
            )
            # Re-ingesting a document deletes its old chunks by doc_id. Without
            # an index that delete is a full scan of the collection.
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(
                "Created Qdrant collection %r (dim=%d, distance=%s, sparse=%s)",
                self._collection,
                dimension,
                _DISTANCE,
                SPARSE_VECTOR,
            )

        self._ready = True

    def _verify_schema(self) -> None:
        """Catch a pre-hybrid collection at ingest time, with the fix in hand.

        Qdrant refuses to add a sparse vector to an existing collection, and a
        fusion query against one fails deep in the engine with a message that
        says nothing about how it got that way. Better to say so here, while the
        operator is running an ingest and can act on it, than at query time.
        """
        if not self._requires_sparse:
            return

        sparse = self._client.get_collection(self._collection).config.params.sparse_vectors
        if sparse and SPARSE_VECTOR in sparse:
            return

        raise CollectionSchemaError(
            f"Qdrant collection {self._collection!r} has no {SPARSE_VECTOR!r} sparse vector, "
            "so it predates hybrid retrieval. Qdrant cannot add one to an existing "
            "collection, and the documents in it were never BM25-indexed, so they would "
            "be invisible to the lexical half of the search. Delete the collection and "
            "re-ingest, or point QDRANT_COLLECTION at a new name."
        )

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVector] | None = None,
        *,
        source: str,
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(vectors)} vectors; they must correspond"
            )
        if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(sparse_vectors)} sparse vectors; "
                "they must correspond"
            )

        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=chunk.id,
                    vector=self._vector_payload(vector, index, sparse_vectors),
                    payload={
                        "doc_id": chunk.doc_id,
                        "index": chunk.index,
                        "page": chunk.page,
                        "text": chunk.text,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "source": source,
                    },
                )
                for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )

    @staticmethod
    def _vector_payload(
        vector: Sequence[float],
        index: int,
        sparse_vectors: Sequence[SparseVector] | None,
    ) -> dict[str, object]:
        # The empty-string key is Qdrant's name for the default unnamed vector.
        # Addressing it explicitly is what allows a named sparse vector to ride
        # alongside it on the same point.
        payload: dict[str, object] = {"": list(vector)}
        if sparse_vectors is not None:
            sparse = sparse_vectors[index]
            payload[SPARSE_VECTOR] = models.SparseVector(
                indices=sparse.indices, values=sparse.values
            )
        return payload

    def delete_document(self, doc_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            ),
        )

    def search(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]:
        return self._query(query=list(vector), limit=top_k)

    def hybrid_search(
        self,
        vector: Sequence[float],
        sparse_vector: SparseVector,
        top_k: int,
        *,
        candidates: int,
    ) -> list[ScoredChunk]:
        if not sparse_vector.indices:
            # Every query term was a stopword or unknown to the tokenizer, so
            # the BM25 arm would match nothing and fusion would just re-rank the
            # dense list at extra cost.
            return self.search(vector, top_k)

        return self._query(
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            prefetch=[
                models.Prefetch(query=list(vector), limit=candidates),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices, values=sparse_vector.values
                    ),
                    using=SPARSE_VECTOR,
                    limit=candidates,
                ),
            ],
        )

    def _query(
        self,
        *,
        query: object,
        limit: int,
        prefetch: list[models.Prefetch] | None = None,
    ) -> list[ScoredChunk]:
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query,
                prefetch=prefetch,
                limit=limit,
                with_payload=True,
            )
        except (UnexpectedResponse, ValueError) as exc:
            if not _is_missing_collection(exc):
                raise
            # Querying before anything has been ingested. An empty corpus is a
            # legitimate state, not a server fault — and checking existence on
            # every query to avoid this would cost a round trip forever to
            # handle a condition that stops occurring after the first upload.
            logger.info("Collection %r does not exist yet; returning no hits", self._collection)
            return []

        return [_to_scored_chunk(point) for point in response.points]


# Embedded Qdrant reports a missing collection as `Collection <name> not found`.
# Anchored, because it is not the only ValueError phrased that way: querying a
# collection that has no sparse index raises `Sparse vector bm25 is not found in
# the collection`, and a bare "not found" substring test would swallow it — the
# schema fault of `_verify_schema` would come back as an empty result set, which
# reads to a caller as "nothing matched" rather than "half the search is dead".
_MISSING_COLLECTION = re.compile(r"^collection .* not found\.?$", re.IGNORECASE)


def _is_missing_collection(exc: Exception) -> bool:
    """Querying a collection that was never created.

    The two client modes report it differently — a server returns 404, while
    the embedded engine raises a plain ValueError — and the ValueError has to
    be matched on its message, so anything else it might mean still propagates.
    """
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code == 404
    return bool(_MISSING_COLLECTION.match(str(exc).strip()))


def _to_scored_chunk(point: models.ScoredPoint) -> ScoredChunk:
    payload = point.payload or {}
    return ScoredChunk(
        chunk=Chunk(
            id=str(point.id),
            doc_id=payload["doc_id"],
            index=payload["index"],
            page=payload["page"],
            text=payload["text"],
            char_start=payload["char_start"],
            char_end=payload["char_end"],
        ),
        source=payload.get("source", ""),
        score=point.score,
    )
