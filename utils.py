import os
import tarfile
import pandas as pd
from pathlib import Path
import boto3
import time
import json
import datetime

region = boto3.Session().region_name
sm_client = boto3.client("sagemaker", region)
cw = boto3.Session().client("cloudwatch")
sm_runtime = boto3.client("sagemaker-runtime")

def create_tar(tarfile_name: str, local_path: Path):
    """
    Create a tar.gz archive with the content of `local_path`.
    """
    with tarfile.open(tarfile_name, mode="w:gz") as archive:
        [
            archive.add(k, arcname=f"{k.relative_to(local_path)}")
            for k in local_path.glob("**/*.*")
            if f"{k.relative_to(local_path)}"[0] != "."
        ]
    tar_size = Path(tarfile_name).stat().st_size / 10**6
    return tar_size

def list_model_metadata_df():
    """
    List the domain, framework, task, and model name of standard machine learning models found in 
    common model zoos.
    """
    list_model_metadata_response = sm_client.list_model_metadata()

    domains = []
    frameworks = []
    framework_versions = []
    tasks = []
    models = []

    for model_summary in list_model_metadata_response["ModelMetadataSummaries"]:
        domains.append(model_summary["Domain"])
        tasks.append(model_summary["Task"])
        models.append(model_summary["Model"])
        frameworks.append(model_summary["Framework"])
        framework_versions.append(model_summary["FrameworkVersion"])

    data = {
        "Domain": domains,
        "Task": tasks,
        "Framework": frameworks,
        "FrameworkVersion": framework_versions,
        "Model": models,
    }

    df = pd.DataFrame(data)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.colheader_justify", "center")
    pd.set_option("display.precision", 3)
    return df

def endpoint_creation_wait(endpoint_name: str):
    """
    Waiting for the endpoint to finish creation
    """
    describe_endpoint_response = sm_client.describe_endpoint(EndpointName=endpoint_name)

    while describe_endpoint_response["EndpointStatus"] == "Creating":
        describe_endpoint_response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(describe_endpoint_response["EndpointStatus"])
        time.sleep(15)

    return describe_endpoint_response

def get_invocation_metrics_for_endpoint_variant(endpoint_name, variant_name, start_time, end_time):
    metrics = cw.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="Invocations",
        StartTime=start_time,
        EndTime=end_time,
        Period=60,
        Statistics=["Sum"],
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
            {"Name": "VariantName", "Value": variant_name},
        ],
    )
    return (
        pd.DataFrame(metrics["Datapoints"])
        .sort_values("Timestamp")
        .set_index("Timestamp")
        .drop("Unit", axis=1)
        .rename(columns={"Sum": variant_name})
    )


def plot_endpoint_metrics(endpoint_name=None, variant1=None, variant2=None, start_time=None):
    start_time = start_time or datetime.now() - timedelta(minutes=60)
    end_time = datetime.datetime.now()
    metrics_variant1 = get_invocation_metrics_for_endpoint_variant(
        endpoint_name, variant1["VariantName"], start_time, end_time
    )
    metrics_variant2 = get_invocation_metrics_for_endpoint_variant(
        endpoint_name, variant2["VariantName"], start_time, end_time
    )
    metrics_variants = metrics_variant1.join(metrics_variant2, how="outer")
    metrics_variants.plot()
    return metrics_variants

def invoke_with_single_sentence(list_data, endpoint_name, variant_name):
    print(f"Sending test traffic to the endpoint {endpoint_name}. \nPlease wait...")
    predictions = []
    for payload in list_data:
        print(".", end="", flush=True)
        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
            TargetVariant=variant_name,
        )
        predictions.append(response["Body"].read().decode("utf-8"))
        time.sleep(0.5)
    print('\nDone!')
    return predictions


def invoke_endpoint_for_two_minutes(endpoint_name=None):
    with open("../sample_payload/batch_data.csv", "r") as f:
        for row in f:
            print(".", end="", flush=True)
            payload = row.rstrip("\n")
            response = sm_runtime.invoke_endpoint(
                EndpointName=endpoint_name, ContentType="text/csv", Body=payload
            )
            response["Body"].read().decode("utf-8")
            time.sleep(1)
            
    return

def endpoint_update_wait(endpoint_name: str):
    """
    Waiting for the endpoint to finish update
    """
    while True:
        status = sm_client.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        if status in ["InService", "Failed"]:
            print("Done")
            break
        print(".", end="", flush=True)
        time.sleep(5)

    return 