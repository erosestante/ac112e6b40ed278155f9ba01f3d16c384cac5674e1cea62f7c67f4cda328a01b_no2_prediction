# deploy/docker/airflow.Dockerfile

FROM apache/airflow:2.9.3

USER airflow

ENV PYTHONPATH=/opt/airflow
ENV PATH="/home/airflow/.local/bin:$PATH"
WORKDIR /opt/airflow

# Copy dependency specs and source code
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

# Install project + airflow deps, pinned & consolidated
RUN pip install --no-cache-dir uv==0.1.37 && \
    pip install --no-cache-dir apache-airflow-providers-docker==1.1.0 && \
    uv pip install --no-cache-dir --python /usr/local/bin/python --prefix=/home/airflow/.local .
