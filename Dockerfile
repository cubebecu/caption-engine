FROM nvidia/cuda:13.2.1-runtime-ubuntu24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set library path so llama-server finds bundled .so files
ENV LD_LIBRARY_PATH=/app${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python deps first (layer caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY server/ ./server/
COPY system_prompt_default.md .
COPY tools/llama.cpp/build/bin/llama-server .
RUN chmod +x llama-server

# Copy shared libraries (llama.cpp) — must be in /app for LD_LIBRARY_PATH
COPY tools/llama.cpp/build/bin/libggml*.so* .
COPY tools/llama.cpp/build/bin/libllama*.so* .
COPY tools/llama.cpp/build/bin/libmtmd*.so* .

# Copy models into the image
COPY models/ ./models/

# Create dirs for jobs + logs
RUN mkdir -p /app/jobs /app/logs

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["python3", "-m", "server.main"]
