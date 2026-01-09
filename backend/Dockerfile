# =============================================================================
# CRM AI Orchestrator - Backend Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
# Features: Non-root user, health check, minimal attack surface
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS production

# Labels for container metadata
LABEL maintainer="CRM AI Orchestrator Team" \
      version="0.1.0" \
      description="Backend API for CRM AI Orchestrator"

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # App configuration
    APP_HOME=/app \
    APP_USER=appuser \
    APP_GROUP=appgroup \
    # Default runtime settings
    PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=1

WORKDIR $APP_HOME

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 $APP_GROUP && \
    useradd --uid 1000 --gid $APP_GROUP --shell /bin/bash --create-home $APP_USER

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=$APP_USER:$APP_GROUP . .

# Switch to non-root user
USER $APP_USER

# Expose port
EXPOSE $PORT

# Health check - verify API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/health || exit 1

# Default command - production server with uvicorn
CMD ["sh", "-c", "uvicorn backend.app.main:app --host $HOST --port $PORT --workers $WORKERS"]

# -----------------------------------------------------------------------------
# Stage 3: Development - With dev tools
# -----------------------------------------------------------------------------
FROM production AS development

# Switch back to root for installing dev dependencies
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install dev Python packages
RUN pip install \
    black \
    ruff \
    mypy \
    pytest \
    pytest-asyncio \
    pytest-cov

# Switch back to non-root user
USER $APP_USER

# Development command with hot reload
CMD ["sh", "-c", "uvicorn backend.app.main:app --host $HOST --port $PORT --reload"]
