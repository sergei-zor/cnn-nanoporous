import os
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from metrics import METRIC_DEFINITIONS

# Runs train_entry.py locally the way SageMaker would, on the 40 exemplary structures (2 epochs, resnet-18)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


def make_data_dir(data_dir):
    # same layout as the training data in S3: database.csv + grids/*.npy
    os.makedirs(os.path.join(data_dir, "grids"))
    expected = pd.read_csv(os.path.join(REPO_ROOT, "expected_inference_results.csv"))
    for filename in expected["filename"]:
        shutil.copy(os.path.join(REPO_ROOT, "exemplary_structures", filename), os.path.join(data_dir, "grids"))
    database = pd.DataFrame({"npy_path": "grids/" + expected["filename"], "cii": expected["ExactMDValue"]})
    database.to_csv(os.path.join(data_dir, "database.csv"), index=False)


def main():
    root = tempfile.mkdtemp()
    make_data_dir(os.path.join(root, "data"))
    env = dict(
        os.environ,
        SM_CHANNEL_TRAINING=os.path.join(root, "data"),
        SM_MODEL_DIR=os.path.join(root, "model"),
        SM_OUTPUT_DATA_DIR=os.path.join(root, "output"),
    )
    # hyperparameters in the "--name value" form SageMaker uses
    command = [sys.executable, os.path.join(HERE, "train_entry.py"),
               "--model_name", "resnet-18", "--size", "80", "--n_epochs", "2", "--batch_size", "4",
               "--cv_folds", "3", "--folds", "0 1", "--use_stratification", "True", "--use_descriptors", "False",
               "--skip_train_eval", "True", "--checkpoint_dir", os.path.join(root, "checkpoints")]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    log = result.stdout + result.stderr
    if result.returncode != 0:
        print(log[-3000:])
        raise SystemExit("train_entry.py failed")

    for path in ["model/fold_0/best_model.pth", "model/fold_1/best_model.pth", "output/fold_0/history.pickle",
                 "output/fold_0/tensorboard", "checkpoints/fold_0_checkpoint.pth"]:
        assert os.path.exists(os.path.join(root, path)), f"Missing {path}"
        print(f"found {path}")
    for metric in METRIC_DEFINITIONS:
        values = re.findall(metric["Regex"], log)
        assert values, f"Metric {metric['Name']} not found in the log"
        print(f"metric {metric['Name']}: {values[-1]}")
    print(f"Smoke test passed, outputs in {root}")


if __name__ == "__main__":
    main()
