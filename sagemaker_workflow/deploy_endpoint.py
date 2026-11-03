import argparse

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig

ROLE_NAME = "SageMakerExecutionRole-nanoporous"
ENDPOINT_NAME = "nanoporous-api"
TRAINING_JOB = "nanoporous-densenet-201-2026-10-30-09-26-38-974"


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_job", type=str, default=TRAINING_JOB, help="Job whose model is deployed")
    parser.add_argument("--image_tag", type=str, default="latest", help="Tag of the nanoporous-api image in ECR")
    parser.add_argument("--delete", action="store_true", help="Delete the endpoint instead")
    return vars(parser.parse_args())


def delete_endpoint(client):
    # an endpoint is made of three resources: endpoint, endpoint config and model
    config_name = client.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointConfigName"]
    model_name = client.describe_endpoint_config(EndpointConfigName=config_name)["ProductionVariants"][0]["ModelName"]
    client.delete_endpoint(EndpointName=ENDPOINT_NAME)
    client.delete_endpoint_config(EndpointConfigName=config_name)
    client.delete_model(ModelName=model_name)
    print(f"Deleted {ENDPOINT_NAME}")


def main():
    args = parse_command_line_args()
    client = boto3.client("sagemaker")
    if args["delete"]:
        delete_endpoint(client)
        return

    account_id = boto3.client("sts").get_caller_identity()["Account"]
    region = boto3.session.Session().region_name
    model_data = client.describe_training_job(TrainingJobName=args["training_job"])["ModelArtifacts"]["S3ModelArtifacts"]

    model = Model(
        name=sagemaker.utils.name_from_base(ENDPOINT_NAME),
        image_uri=f"{account_id}.dkr.ecr.{region}.amazonaws.com/nanoporous-api:{args['image_tag']}",
        model_data=model_data,
        role=f"arn:aws:iam::{account_id}:role/{ROLE_NAME}",
        # SageMaker unpacks model.tar.gz into /opt/ml/model
        env={"MODEL_PATH": "/opt/ml/model/fold_0/best_model.pth"},
    )
    print(f"Deploying {model_data}")
    model.deploy(
        endpoint_name=ENDPOINT_NAME,
        # the API uses ~700 MB, 3072 MB is the default limit of new accounts
        serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=3072, max_concurrency=1),
        tags=[{"Key": "project", "Value": "nanoporous"}],
    )
    print(f"Endpoint {ENDPOINT_NAME} is ready")


if __name__ == "__main__":
    main()
