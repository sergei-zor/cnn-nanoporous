import argparse
import os
import shutil
import tempfile

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

from metrics import METRIC_DEFINITIONS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKET = "nanoporous-sagemaker-sz-euw1"
ROLE_NAME = "SageMakerExecutionRole-nanoporous"
CHECKPOINT_DIR = "/opt/ml/checkpoints"
# train.py and the files it imports, uploaded as the job's code
SOURCE_FILES = [
    "train.py",
    "training.py",
    "data.py",
    "utils.py",
    "validation.py",
    "inputoutput.py",
    "generate_model.py",
    "TRIDCNNPyTorch",
    "sagemaker_workflow/train_entry.py",
]
# settings of run_training.sh, plus skip_train_eval and checkpoint_dir added for SageMaker
HYPERPARAMETERS = {
    "model_name": "densenet-201",
    "optimizer": "Adam",
    "batch_size": 40,
    "size": 80,
    "p_flip": 0.3,
    "aug_roll_ratio": 0.3,
    "cv_folds": 8,
    "use_stratification": True,
    "skip_train_eval": True,
    "checkpoint_dir": CHECKPOINT_DIR,
}


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=str, default="0", help='Folds to train, e.g. "0" or "0 1 2"')
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--instance_type", type=str, default="ml.g5.xlarge")
    parser.add_argument("--spot", action="store_true", help="Use a spot instance (cheaper, can be interrupted)")
    parser.add_argument("--resume_job", type=str, default="", help="Continue from the checkpoints of this job")
    parser.add_argument("--no_wait", action="store_true", help="Don't wait for the job to finish")
    return vars(parser.parse_args())


def copy_source_files():
    # only the needed files, not the whole repo with datasets and models
    source_dir = tempfile.mkdtemp()
    for path in SOURCE_FILES:
        destination = os.path.join(source_dir, path)
        if os.path.isdir(os.path.join(REPO_ROOT, path)):
            shutil.copytree(os.path.join(REPO_ROOT, path), destination, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy(os.path.join(REPO_ROOT, path), destination)
    # the container installs requirements.txt before training
    shutil.copy(os.path.join(REPO_ROOT, "sagemaker_workflow", "requirements-train.txt"), os.path.join(source_dir, "requirements.txt"))
    return source_dir


def main():
    args = parse_command_line_args()
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    hyperparameters = dict(HYPERPARAMETERS, folds=args["folds"], n_epochs=args["n_epochs"], lr=args["lr"])

    # the job name is chosen here, so the checkpoint folder can be named after it
    job_name = sagemaker.utils.name_from_base("nanoporous-densenet-201")
    checkpoint_s3_uri = f"s3://{BUCKET}/checkpoints/{args['resume_job'] or job_name}"
    max_run = 24 * 3600

    estimator = PyTorch(
        entry_point="sagemaker_workflow/train_entry.py",
        source_dir=copy_source_files(),
        framework_version="2.6",
        py_version="py312",
        role=f"arn:aws:iam::{account_id}:role/{ROLE_NAME}",
        instance_type=args["instance_type"],
        instance_count=1,
        volume_size=50,  # GB, the grids are 7.7 GB
        hyperparameters=hyperparameters,
        metric_definitions=METRIC_DEFINITIONS,
        output_path=f"s3://{BUCKET}/training",
        code_location=f"s3://{BUCKET}/code",
        max_run=max_run,
        use_spot_instances=args["spot"],
        max_wait=2 * max_run if args["spot"] else None,  # time limit including waiting for a spot instance
        # SageMaker uploads CHECKPOINT_DIR to S3 during training and downloads it again when the job restarts
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=CHECKPOINT_DIR,
        tags=[{"Key": "project", "Value": "nanoporous"}],
    )

    print("Hyperparameters:", hyperparameters)
    estimator.fit(
        {"training": TrainingInput(f"s3://{BUCKET}/database/dataset1")},
        job_name=job_name,
        wait=not args["no_wait"],
    )
    print(f"Job: {job_name}")
    print(f"Model: s3://{BUCKET}/training/{job_name}/output/model.tar.gz")
    print(f"Checkpoints: {checkpoint_s3_uri}")


if __name__ == "__main__":
    main()
