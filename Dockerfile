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

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
