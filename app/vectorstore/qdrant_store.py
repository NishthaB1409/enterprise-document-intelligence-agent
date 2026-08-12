"""Qdrant-backed implementation of `VectorStore`.

The chunk is stored whole in the point payload rather than kept in a second
database keyed by id. One round trip returns everything a citation needs, and
there is no way for the text an answer quotes to drift out of sync with the
vector that retrieved it.
"""

import logging
from collections.abc import Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.ingest.chunking import Chunk
from app.vectorstore.store import ScoredChunk

logger = logging.getLogger(__name__)

# Cosine, because the embedding models this service uses (bge, e5) are trained
# with a cosine objective and their vectors are normalized.
_DISTANCE = models.Distance.COSINE


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection
        # Set once the collection is known to exist, so the common path (every
        # ingest after the first) costs no round trip.
        self._ready = False

    def ensure_collection(self, dimension: int) -> None:
        if self._ready:
            return

        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=dimension, distance=_DISTANCE),
            )
            # Re-ingesting a document deletes its old chunks by doc_id. Without
            # an index that delete is a full scan of the collection.
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(
                "Created Qdrant collection %r (dim=%d, distance=%s)",
                self._collection,
                dimension,
                _DISTANCE,
            )

        self._ready = True

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, source: str
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(vectors)} vectors; they must correspond"
            )

        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=chunk.id,
                    vector=list(vector),
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
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

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
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                limit=top_k,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
            # Querying before anything has been ingested. An empty corpus is a
            # legitimate state, not a server fault — and checking existence on
            # every query to avoid this would cost a round trip forever to
            # handle a condition that stops occurring after the first upload.
            logger.info("Collection %r does not exist yet; returning no hits", self._collection)
            return []

        return [_to_scored_chunk(point) for point in response.points]


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
