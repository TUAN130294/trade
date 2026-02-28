# ===========================================
# VN-QUANT Trading System - Docker Image
# ===========================================

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt ./

# Install Python dependencies (Use CPU version for torch to speed up build)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000
EXPOSE 8003
EXPOSE 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "run_autonomous_paper_trading.py"]
