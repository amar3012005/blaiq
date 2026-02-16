# Dockerfile for Optimized GraphRAG Service (Fast Build Optimized)
FROM python:3.11-slim AS base

# Install uv for lightning-fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvbin/uv
ENV PATH="/uvbin:${PATH}"
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements or individual packages for installation
# We'll use a specific index for torch to keep it under 200MB (CPU only)
ENV UV_HTTP_TIMEOUT=120
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
    redis \
    qdrant-client \
    neo4j \
    langchain \
    langchain-openai \
    langchain-core \
    fastapi \
    uvicorn \
    python-dotenv \
    pyyaml \
    sentence-transformers \
    pydantic \
    groq \
    torch \
    "optimum[onnxruntime]" \
    onnx \
    onnxruntime-tools \
    requests

# ============================================================================
# Runtime Stage
# ============================================================================
FROM base AS runtime

WORKDIR /app

# Copy application code (This layer changes often, but the heavy deps above are cached)
COPY src/ ./src/
COPY static/ ./static/
COPY config.yaml ./

# Create directories and set permissions
RUN mkdir -p data/_pipeline logs && touch logs/llm_error.log && chmod -R 777 logs data

# Expose API port
EXPOSE 8001

# Set python path
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

# Run optimized API
CMD ["python", "-m", "uvicorn", "src.retriever.retriever_api_optimized:app", "--host", "0.0.0.0", "--port", "8001"]
