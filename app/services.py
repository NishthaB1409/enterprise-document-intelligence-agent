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
from app.generation.answerer import Answerer, LLMAnswerer
from app.graph.nodes import Critic, Grader, Rewriter, Router
from app.graph.pipeline import AgentPipeline, QueryPipeline, SimplePipeline
from app.ingest.embedding import Embedder, FastEmbedEmbedder
from app.ingest.sparse import FastEmbedSparseEmbedder, SparseEmbedder
from app.llm import AnthropicLLM, OpenAILLM, StructuredLLM
from app.retrieval.reranking import CrossEncoderReranker
from app.retrieval.retriever import (
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
    Retriever,
)
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.vectorstore.store import VectorStore


@dataclass(frozen=True, slots=True)
class Services:
    embedder: Embedder
    store: VectorStore
    retriever: Retriever
    answerer: Answerer
    # What `/query` actually calls. Either the single retrieve-then-generate
    # pass or the agentic graph; the route cannot tell which.
    pipeline: QueryPipeline
    # None in dense-only mode. Ingestion checks it rather than the settings, so
    # what gets indexed always matches what the retriever can search.
    sparse_embedder: SparseEmbedder | None = None


def build_services(settings: Settings) -> Services:
    embedder = FastEmbedEmbedder(
        model_name=settings.embedding_model, cache_dir=settings.embedding_cache_dir
    )
    hybrid = settings.retrieval_mode == "hybrid"
    store = QdrantVectorStore(
        client=build_qdrant_client(settings),
        collection=settings.qdrant_collection,
        requires_sparse=hybrid,
    )
    sparse_embedder = (
        FastEmbedSparseEmbedder(
            model_name=settings.sparse_model, cache_dir=settings.embedding_cache_dir
        )
        if hybrid
        else None
    )

    retriever = build_retriever(settings, embedder, sparse_embedder, store)
    llm = build_llm(settings)
    answerer = LLMAnswerer(llm, max_tokens=settings.answer_max_tokens)

    return Services(
        embedder=embedder,
        store=store,
        sparse_embedder=sparse_embedder,
        retriever=retriever,
        answerer=answerer,
        pipeline=build_pipeline(settings, retriever, answerer, llm),
    )


def build_pipeline(
    settings: Settings, retriever: Retriever, answerer: Answerer, llm: StructuredLLM
) -> QueryPipeline:
    """The agentic graph, or the single pass it has to justify itself against.

    Each node is constructed only if it is enabled, and the graph is handed
    `None` for the rest — so a disabled node is an absent object rather than a
    branch that runs and returns early.
    """
    if not settings.agent_enabled:
        return SimplePipeline(retriever, answerer)

    effort = settings.agent_effort
    return AgentPipeline(
        retriever=retriever,
        answerer=answerer,
        router=Router(llm, effort=effort) if settings.agent_route else None,
        grader=Grader(llm, effort=effort) if settings.agent_grade else None,
        # The rewriter is only ever reached from a grade that found nothing, so
        # without grading there is no path to it.
        rewriter=Rewriter(llm, effort=effort) if settings.agent_grade else None,
        critic=Critic(llm) if settings.agent_critique else None,
        max_rewrites=settings.agent_max_rewrites,
    )


def build_retriever(
    settings: Settings,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder | None,
    store: VectorStore,
) -> Retriever:
    """Assemble the configured retrieval stack.

    Reranking composes over either base retriever, so the two settings are
    independent — which they need to be, because on the corpus in `eval/` they
    do not move the score in the same direction.
    """
    base: Retriever = (
        DenseRetriever(embedder, store, settings.retrieval_top_k)
        if sparse_embedder is None
        else HybridRetriever(
            embedder,
            sparse_embedder,
            store,
            settings.retrieval_top_k,
            candidates=settings.retrieval_candidates,
        )
    )

    if not settings.rerank_enabled:
        return base

    return RerankingRetriever(
        base,
        CrossEncoderReranker(
            model_name=settings.reranker_model, cache_dir=settings.embedding_cache_dir
        ),
        settings.retrieval_top_k,
        candidates=settings.retrieval_candidates,
    )


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Embedded if `qdrant_path` is set, otherwise a server at `qdrant_url`.

    Same client class either way, so `QdrantVectorStore` cannot tell the
    difference and neither can anything above it.
    """
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        # Left on, this client probes the server for its version during
        # construction — which would make merely importing this module do
        # network I/O, and emit a warning on every test run.
        check_compatibility=False,
    )


def build_llm(settings: Settings) -> StructuredLLM:
    """The provider switch, and the whole of the difference between vendors.

    Every model call in the service — answering, routing, grading, rewriting,
    critique — goes through the object this returns, so switching provider is
    one branch rather than one branch per call site.
    """
    if settings.llm_provider == "openai":
        return OpenAILLM(
            model=settings.resolved_answer_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_tokens=settings.answer_max_tokens,
        )

    return AnthropicLLM(
        model=settings.resolved_answer_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.answer_max_tokens,
        effort=settings.answer_effort,
    )
