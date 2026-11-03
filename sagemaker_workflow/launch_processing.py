import argparse
import os
import shutil
import tempfile

import boto3
import sagemaker
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKET = "nanoporous-sagemaker-sz-euw1"
ROLE_NAME = "SageMakerExecutionRole-nanoporous"
DATASET = "Dataset1_gold_0.25.xlsx"
INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"
# generate_grids.py, plus the original scripts that verify_generate_grids.py compares it with
SOURCE_FILES = [
    "sagemaker_workflow/generate_grids.py",
    "sagemaker_workflow/verify_generate_grids.py",
    "datasets/grid_generator.py",
    "database_preparation/npy_rotations.py",
    "database_preparation/prepare_database.py",
]
# the scikit-learn image has pandas 1.1, which can't read .xlsx with openpyxl; numpy must stay 1.x for its scipy
REQUIREMENTS = "pandas==1.5.3\nnumpy<2\nopenpyxl\ntqdm\n"


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["verify", "generate"], required=True)
    parser.add_argument("--max_structures", type=int, default=0, help="Only the first N structures, 0 = all")
    parser.add_argument("--instance_type", type=str, default="ml.t3.xlarge")
    parser.add_argument("--instance_count", type=int, default=1)
    parser.add_argument("--output_prefix", type=str, default="database/dataset1", help="S3 folder for the output")
    parser.add_argument("--no_wait", action="store_true", help="Don't wait for the job to finish")
    return vars(parser.parse_args())


def copy_source_files():
    source_dir = tempfile.mkdtemp()
    for path in SOURCE_FILES:
        os.makedirs(os.path.dirname(os.path.join(source_dir, path)), exist_ok=True)
        shutil.copy(os.path.join(REPO_ROOT, path), os.path.join(source_dir, path))
    with open(os.path.join(source_dir, "requirements.txt"), "w") as f:
        f.write(REQUIREMENTS)
    return source_dir


def main():
    args = parse_command_line_args()
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    dataset_path = f"{INPUT_DIR}/{DATASET}"

    processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.2-1",
        py_version="py3",
        role=f"arn:aws:iam::{account_id}:role/{ROLE_NAME}",
        instance_type=args["instance_type"],
        instance_count=args["instance_count"],
        base_job_name=f"nanoporous-{args['mode']}",
        code_location=f"s3://{BUCKET}/code",
        max_runtime_in_seconds=12 * 3600,
        tags=[{"Key": "project", "Value": "nanoporous"}],
    )

    if args["mode"] == "verify":
        code = "sagemaker_workflow/verify_generate_grids.py"
        arguments = ["--dataset", dataset_path]
        outputs = []
    else:
        code = "sagemaker_workflow/generate_grids.py"
        arguments = ["--dataset", dataset_path, "--output_dir", OUTPUT_DIR, "--include_descriptors",
                     "--max_structures", str(args["max_structures"])]
        # everything written to OUTPUT_DIR is uploaded to S3 at the end of the job
        outputs = [ProcessingOutput(source=OUTPUT_DIR, destination=f"s3://{BUCKET}/{args['output_prefix']}")]

    processor.run(
        code=code,
        source_dir=copy_source_files(),
        arguments=arguments,
        inputs=[ProcessingInput(source=f"s3://{BUCKET}/raw/{DATASET}", destination=INPUT_DIR)],
        outputs=outputs,
        wait=not args["no_wait"],
    )
    print(f"Job: {processor.latest_job.job_name}")


if __name__ == "__main__":
    main()
