# 1. Base image: lightweight Python 3.12
FROM python:3.12-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Install system tools (gcc) and UV (your Python env manager)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install uv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency specs (pyproject.toml & lockfile)
COPY pyproject.toml uv.lock* ./

# 5. Install Python dependencies via UV
RUN uv pip install --system --no-cache-dir .


# 6. Copy your pipeline code
COPY src/ src/

# 7. Default command: run your ML pipeline
CMD ["python", "src/run_pipeline.py"]
