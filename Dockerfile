# uv's own image, so the lockfile is installed by the tool that wrote it —
# `uv sync --frozen` fails loudly if uv.lock and pyproject.toml have drifted,
# which is the point of checking a lockfile in.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Compile to .pyc at install time (slower build, faster start) and copy rather
# than symlink, so nothing points outside the image layer.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Dependencies first, in their own layer: they change far less often than the
# application code, so editing a route does not reinstall torch-sized wheels.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
RUN uv sync --frozen --no-dev

ENV PATH="/srv/.venv/bin:$PATH"

# The embedding weights (~130MB) download on first use. Pointing this at a
# volume keeps a container restart from re-downloading them.
ENV EMBEDDING_CACHE_DIR=/var/cache/fastembed

# Not root. The service parses uploaded PDFs, which is exactly the code path a
# hostile file would aim at, so the process gets nothing it does not need.
# The two directories it writes to are created here and owned by it: a named
# volume mounted over a directory copies that directory's ownership when the
# volume is first created, so the volumes start out writable too.
#
# The user gets a real home directory. The Hugging Face downloader that fetches
# the embedding model writes logs and state under $HOME/.cache, and without a
# home every download fails with "Permission denied", even though the model
# cache itself is writable. HF_HOME then moves that state onto the cache volume,
# so it persists alongside the model it describes.
RUN useradd --system --uid 10001 --create-home --home-dir /home/edia edia     && mkdir -p /srv/data /var/cache/fastembed     && chown edia /srv/data /var/cache/fastembed
ENV HF_HOME=/var/cache/fastembed/huggingface
USER edia

EXPOSE 8000

# Python rather than curl: the slim base image ships no curl, and installing it
# would add a network tool to the image for the sake of one probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3     CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
