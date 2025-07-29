from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime

# Absolute path to the root of your repo on the host:
REPO_DIR = "/Users/erosestante/Documents/mlops-no2/ac112e6b40ed278155f9ba01f3d16c384cac5674e1cea62f7c67f4cda328a01b_no2_prediction-1"

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 7, 29),
}

with DAG(
    dag_id="ml_pipeline_docker",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    run_entire_pipeline = DockerOperator(
        task_id="run_pipeline",
        image="ac112e6b40ed278155f9ba01f3d16c384cac5674e1cea62f7c67f4cda328a01b-ml-pipeline",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(
                source=f"{REPO_DIR}/data",
                target="/app/data",
                type="bind"
            ),
            Mount(
                source=f"{REPO_DIR}/models",
                target="/app/models",
                type="bind"
            ),
        ],
        # no volumes= here!
    )
