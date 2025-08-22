FROM apache/airflow:3.0.3

USER airflow

ENV PYTHONPATH=/opt/airflow
ENV PATH="/home/airflow/.local/bin:$PATH"
ENV PYTHONUSERBASE=/home/airflow/.local
#this is what uv respects by default

WORKDIR /opt/airflow

# Copy lockfiles and code
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

# Install uv and dependencies into airflow's home
RUN pip install --no-cache-dir uv==0.1.40 && \
    pip install --no-cache-dir apache-airflow-providers-docker==2.7.0 && \
    uv pip install --no-cache-dir --system .
