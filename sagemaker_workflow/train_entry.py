import argparse
import json
import os
import shutil
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train

# store_true flags of train.py, SageMaker passes them as "--flag True"
BOOLEAN_FLAGS = [
    "smoke_test",
    "use_descriptors",
    "no_structure",
    "use_stratification",
    "no_testset",
    "remove_model",
    "use_pca",
    "scale_only",
    "skip_train_eval",
]
# nargs="+" arguments of train.py, SageMaker passes them as one string "0 1 2"
LIST_ARGS = ["folds", "fc_first_size", "fc_second_size"]


def parse_command_line_args():
    # SageMaker sets these environment variables in the container
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    args, hyperparameters = parser.parse_known_args()
    return vars(args), hyperparameters


def to_train_args(hyperparameters):
    train_args = []
    for name, value in zip(hyperparameters[::2], hyperparameters[1::2]):
        if name[2:] in BOOLEAN_FLAGS:
            if value == "True":
                train_args.append(name)
        elif name[2:] in LIST_ARGS:
            train_args += [name] + value.split()
        else:
            train_args += [name, value]
    return train_args


def localize_database(data_dir, workdir):
    # database.csv has paths like "grids/0_rot-x.npy", train.py needs the full path inside the container
    database = pd.read_csv(os.path.join(data_dir, "database.csv"))
    grids_dir = os.path.join(data_dir, "grids").replace("\\", "/")  # data.py splits paths on "/"
    database["npy_path"] = [f"{grids_dir}/{os.path.basename(path)}" for path in database["npy_path"]]
    database_path = os.path.join(workdir, "database.csv")
    database.to_csv(database_path, index=False)
    return database_path


def save_outputs(workdir, model_dir, output_data_dir):
    # train.py writes everything to <workdir>/experiment/fold_<k>/
    experiment_dir = os.path.join(workdir, "experiment")
    for fold in sorted(os.listdir(experiment_dir)):
        fold_dir = os.path.join(experiment_dir, fold)
        # best model -> model.tar.gz, everything else (metrics, history, tensorboard) -> output.tar.gz
        os.makedirs(os.path.join(model_dir, fold))
        shutil.move(os.path.join(fold_dir, "best_model.pth"), os.path.join(model_dir, fold, "best_model.pth"))
        shutil.copytree(fold_dir, os.path.join(output_data_dir, fold))

        r2 = {}
        for split in ["train", "val", "test"]:
            with open(os.path.join(fold_dir, f"summary_{split}_best_model.json")) as f:
                r2[split] = json.load(f)["r2"]
        print(f"{fold} best model: train_r2={r2['train']:.6f} val_r2={r2['val']:.6f} test_r2={r2['test']:.6f}")


def main():
    args, hyperparameters = parse_command_line_args()
    workdir = tempfile.mkdtemp()
    database_path = localize_database(args["data_dir"], workdir)

    # fixed folder name instead of a timestamp, which also has ":" that Windows doesn't allow (local test)
    train.get_experiment_name = lambda args: "experiment"
    # run train.py in this process, as if it was called from the command line
    sys.argv = ["train.py"] + to_train_args(hyperparameters) + ["--database", database_path, "--outputdir", workdir]
    print("Running:", " ".join(sys.argv))
    train.main()

    save_outputs(workdir, args["model_dir"], args["output_data_dir"])


if __name__ == "__main__":
    main()
