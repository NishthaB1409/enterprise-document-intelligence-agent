import pytest
from fastapi.testclient import TestClient
from langfuse import get_client

from app.config import Settings
from app.main import create_app
from app.retrieval.retriever import HybridRetriever
from app.services import Services
from tests.fakes import (
    InMemoryVectorStore,
    StubAnswerer,
    StubEmbedder,
    StubSparseEmbedder,
)
from tests.fake_langfuse import FakeLangfuseServer

# The Langfuse client is a process-wide singleton, so the server and app are built
# once per session rather than per test.


@pytest.fixture(scope="session")
def fake_langfuse():
    with FakeLangfuseServer() as server:
        yield server


@pytest.fixture(scope="session")
def settings(fake_langfuse: FakeLangfuseServer) -> Settings:
    return Settings(
        # Ignore the developer's own .env. Without this, whichever provider and
        # keys happen to be configured locally would leak into the suite, and
        # tests that assert on an *unconfigured* deployment would pass or fail
        # depending on whose machine they ran on.
        _env_file=None,
        llm_provider="anthropic",
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_host=fake_langfuse.host,
        environment="test",
        release="test-release",
        # Present so /query gets past its configuration check; nothing reaches
        # Anthropic, because the answerer is a stub.
        anthropic_api_key="sk-ant-test",
        retrieval_top_k=3,
        chunk_size_words=40,
        chunk_overlap_words=5,
        # Mirrors the production default, so the end-to-end tests exercise the
        # retriever that actually ships rather than a simpler stand-in.
        retrieval_mode="hybrid",
        retrieval_candidates=10,
        # The real cross-encoder would download ~80MB. Reranking has its own
        # tests against a stub; here the point is the rest of the pipeline.
        rerank_enabled=False,
    )


@pytest.fixture(scope="session")
def services(settings: Settings) -> Services:
    embedder = StubEmbedder()
    sparse_embedder = StubSparseEmbedder()
    store = InMemoryVectorStore()
    return Services(
        embedder=embedder,
        store=store,
        sparse_embedder=sparse_embedder,
        retriever=HybridRetriever(
            embedder,
            sparse_embedder,
            store,
            settings.retrieval_top_k,
            candidates=settings.retrieval_candidates,
        ),
        answerer=StubAnswerer(),
    )


@pytest.fixture(scope="session")
def client(settings: Settings, services: Services):
    app = create_app(settings, services)
    # Entering TestClient runs the lifespan, which initialises the Langfuse client.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _isolate_services(services: Services):
    """The app is built once per session, so anything a test indexes or stubs
    would otherwise leak into the next one."""
    yield
    services.store.clear()
    services.answerer.reset()


@pytest.fixture
def store(services: Services) -> InMemoryVectorStore:
    return services.store


@pytest.fixture
def answerer(services: Services) -> StubAnswerer:
    return services.answerer


@pytest.fixture
def spans(fake_langfuse: FakeLangfuseServer):
    """Clears captured spans before a test, and flushes the exporter after the
    request so assertions run against everything the SDK actually sent."""
    fake_langfuse.collector.spans.clear()
    yield fake_langfuse.collector


@pytest.fixture
def flush():
    def _flush() -> None:
        get_client().flush()

    return _flush
