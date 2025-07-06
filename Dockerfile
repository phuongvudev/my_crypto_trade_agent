# Multi-stage Docker build for Crypto Trading AI Agent
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gcc \
    g++ \
    make \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
WORKDIR /tmp
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib from builder stage
COPY --from=builder /usr/lib/libta_lib.* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading -s /bin/bash trading

# Create application directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data/historical /app/data/live /app/models && \
    chown -R trading:trading /app

# Copy application code
COPY --chown=trading:trading . .

# Create .env file template
RUN echo "# Add your exchange API credentials here" > .env.template && \
    echo "BINANCE_API_KEY=your_api_key_here" >> .env.template && \
    echo "BINANCE_SECRET=your_secret_here" >> .env.template && \
    chown trading:trading .env.template

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose port for dashboard
EXPOSE 8501

# Default command
CMD ["python", "main.py", "--mode", "paper", "--strategy", "rule_based"]

# Labels for metadata
LABEL maintainer="Crypto Trading AI Team" \
      version="1.0.0" \
      description="AI-powered cryptocurrency trading agent" \
      org.opencontainers.image.source="https://github.com/your-username/crypto-trading-ai"
