FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /uvx /usr/local/bin/

RUN uv sync --locked --no-dev --no-install-project

COPY src/ ./src/

COPY models/ ./models/

COPY README.md ./

RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "iris.api.main:app", "--host", "0.0.0.0", "--port", "8000"]