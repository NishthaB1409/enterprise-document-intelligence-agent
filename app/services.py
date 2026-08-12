"""The composition root: the one place that picks concrete implementations.

Everything below the API layer is written against Protocols (`Embedder`,
`VectorStore`, `Answerer`). This module is where those become a FastEmbed
embedder, a Qdrant client, and an Anthropic call — and it is the only module a
test has to replace to run the whole pipeline in-process.

Construction here is deliberately cheap and offline: the embedder defers its
weight download, the answerer defers its client, and the Qdrant client does not
connect until asked. Building the app must not require the world to be up.
"""

from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.config import Settings
from app.generation.answerer import Answerer, AnthropicAnswerer
from app.ingest.embedding import Embedder, FastEmbedEmbedder
from app.retrieval.retriever import DenseRetriever
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.vectorstore.store import VectorStore


@dataclass(frozen=True, slots=True)
class Services:
    embedder: Embedder
    store: VectorStore
    retriever: DenseRetriever
    answerer: Answerer


def build_services(settings: Settings) -> Services:
    embedder = FastEmbedEmbedder(
        model_name=settings.embedding_model, cache_dir=settings.embedding_cache_dir
    )
    store = QdrantVectorStore(
        client=QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        collection=settings.qdrant_collection,
    )
    return Services(
        embedder=embedder,
        store=store,
        retriever=DenseRetriever(embedder, store, settings.retrieval_top_k),
        answerer=AnthropicAnswerer(
            model=settings.answer_model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.answer_max_tokens,
            effort=settings.answer_effort,
        ),
    )
