import pytest
from fastapi.testclient import TestClient
from langfuse import get_client

from app.config import Settings
from app.main import create_app
from tests.fake_langfuse import FakeLangfuseServer

# The Langfuse client is a process-wide singleton, so the server and app are built
# once per session rather than per test.


@pytest.fixture(scope="session")
def fake_langfuse():
    with FakeLangfuseServer() as server:
        yield server


@pytest.fixture(scope="session")
def client(fake_langfuse: FakeLangfuseServer):
    settings = Settings(
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_host=fake_langfuse.host,
        environment="test",
        release="test-release",
    )
    app = create_app(settings)
    # Entering TestClient runs the lifespan, which initialises the Langfuse client.
    with TestClient(app) as test_client:
        yield test_client


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
