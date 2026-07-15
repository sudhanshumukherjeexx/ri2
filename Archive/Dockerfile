# Stage 1: Build wheels
FROM python:3.12-slim as builder
WORKDIR /wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.12-slim
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create user
RUN useradd -m streamlituser


# Install curl and necessary dependencies including libgomp1 for LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy only the wheel files (not requirements.txt)
COPY --from=builder /wheels/*.whl /wheels/

# Install Python packages
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels && \
    rm -rf ~/.cache/pip

# Copy application code
COPY . .
RUN mkdir -p /app/data && \
    chown -R streamlituser:streamlituser /app

# Set permissions and user
USER streamlituser

# Expose port and define health check
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "Hello.py", "--server.port=8501", "--server.enableCORS=false"]