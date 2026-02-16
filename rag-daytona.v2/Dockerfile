# ============================================================================
# RAG Service Dockerfile - Daytona V2 Production Build (Fast Build Optimized)
# ============================================================================

# Step 1: Base image with uv installed
FROM python:3.11-slim AS base

# Install uv for lightning-fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvbin/uv
ENV PATH="/uvbin:${PATH}"
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files FIRST
COPY requirements.txt ./requirements.txt

# Install ALL dependencies
ENV UV_HTTP_TIMEOUT=120
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# ============================================================================
# Runtime Stage - Final production image
# ============================================================================
FROM base AS runtime

# Create package structure
RUN mkdir -p /app/daytona_agent/services && \
    echo "" > /app/daytona_agent/__init__.py && \
    echo "" > /app/daytona_agent/services/__init__.py

# Copy service code
COPY . /app/daytona_agent/services/rag
COPY shared /app/daytona_agent/services/shared
COPY data /app/daytona_knowledge_base

# Set Python environment
ENV PYTHONPATH=/app:/app/daytona_agent/services/rag
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Threading Optimization for ONNX
ENV OMP_NUM_THREADS=4
ENV ONNXRUNTIME_EXECUTION_MODE=parallel

# Configuration
ENV DAYTONA_RAG_EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENV DAYTONA_RAG_VECTOR_STORE_PATH="/app/index"
ENV DAYTONA_RAG_KNOWLEDGE_BASE_PATH="/app/daytona_knowledge_base"
ENV PORT="8000"
ENV LOG_LEVEL="INFO"

# Expose service port
EXPOSE 8000

# Run service
CMD ["python", "-m", "uvicorn", "daytona_agent.services.rag.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
