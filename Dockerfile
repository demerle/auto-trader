FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

RUN useradd -u 1000 -m appuser && mkdir -p /app/data && chown -R appuser:appuser /app
WORKDIR /app
USER appuser

ENV UV_PROJECT_ENVIRONMENT=/home/appuser/venv \
    PATH="/home/appuser/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=appuser:appuser . .
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uvicorn", "autotrader.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
