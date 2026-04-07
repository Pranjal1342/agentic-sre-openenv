# =============================================================================
# Agentic SRE OpenEnv Environment — Multi-Stage Dockerfile
#
# Stage 1 (builder): Runs the offline RAG index build (unstructured + FAISS).
#                    Heavy deps are installed here; only the resulting
#                    assets/ directory is copied into the runtime image.
#
# Stage 2 (runtime): Lean production image. Loads the pre-built FAISS index
#                    at startup — no external DB or internet required.
# =============================================================================

# ---- Stage 1: Offline RAG Index Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps separately to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files needed for the offline index build.
COPY rag/ ./rag/
COPY openenv.yaml .
COPY knowledge_base/ ./knowledge_base/

# Run the offline FAISS build script.
# Output: /build/assets/faiss_index/index.faiss + metadata.pkl
# These pre-built assets are baked into the runtime image, satisfying the
# OpenEnv requirement for a clean `docker build && docker run` startup
# with no external database dependencies.
RUN python rag/offline_index.py


# ---- Stage 2: Runtime Image ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install curl for the HEALTHCHECK probe before dropping to non-root.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Add a non-root user for HF Spaces security compliance.
RUN adduser --disabled-password --gecos "" --uid 1000 appuser

# Install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-built FAISS index from Stage 1.
COPY --from=builder /build/assets/faiss_index ./assets/faiss_index

# Copy the full application source.
COPY . .

# Transfer ownership to the non-root user.
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Liveness probe — Docker and HF Spaces use this to determine if the
# container is healthy before routing traffic to it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uvicorn ASGI server — module path matches openenv.yaml `app` field.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]