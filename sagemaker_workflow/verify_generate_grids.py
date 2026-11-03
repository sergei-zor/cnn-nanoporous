import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

# Runs the original pipeline (grid_generator.py -> npy_rotations.py -> prepare_database.py) and generate_grids.py
# on the same structures, and checks that the grids and database.csv are identical
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRECTIONS = ["x", "y", "z"]


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=os.path.join(REPO_ROOT, "datasets", "Dataset1_gold_0.25.xlsx"))
    parser.add_argument("--numbers", nargs="+", type=int, default=[0, 155, 2599, 4999])
    return vars(parser.parse_args())


def run(script, *args):
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, script), *map(str, args)], check=True, capture_output=True)


def run_original_pipeline(dataset, numbers, workdir):
    # grid_generator.py processes every structure in the file, so give it only the selected ones
    subset = dataset[dataset["Number"].isin(numbers)].reset_index(drop=True)
    subset_path = os.path.join(workdir, "subset.xlsx")
    subset.to_excel(subset_path, index=False)
    run("datasets/grid_generator.py", "--dataset", subset_path, "--output_dir", os.path.join(workdir, "generated"),
        "--grid_resolution", 80, "--n_cells", 125, "--crystal_system", "FCC")

    # grid_generator.py names grids grid_0, grid_1, ..., rename them to the structure number
    grids_dir = os.path.join(workdir, "original_grids")
    os.makedirs(grids_dir)
    for counter, number in enumerate(subset.drop_duplicates("Number")["Number"]):
        shutil.move(os.path.join(workdir, "generated", "grids_80", f"grid_{counter}.npy"),
                    os.path.join(grids_dir, f"grid_{number}.npy"))

    rotated_dir = os.path.join(workdir, "original_rotated")
    for direction in DIRECTIONS:
        run("database_preparation/npy_rotations.py", "--input_dir", grids_dir, "--output_dir", rotated_dir,
            "--rotation", direction)
    run("database_preparation/prepare_database.py", "--dataset", subset_path, "--input_dir", rotated_dir,
        "--output_dir", workdir, "--output_csv", "original_database.csv", "--include_descriptors")
    return rotated_dir, os.path.join(workdir, "original_database.csv")


def main():
    args = parse_command_line_args()
    dataset = pd.read_excel(args["dataset"])
    numbers = args["numbers"]
    workdir = tempfile.mkdtemp()

    print(f"Running the original pipeline on structures {numbers}")
    original_grids, original_csv = run_original_pipeline(dataset, numbers, workdir)
    print("Running generate_grids.py")
    new_dir = os.path.join(workdir, "new")
    run("sagemaker_workflow/generate_grids.py", "--dataset", args["dataset"], "--output_dir", new_dir,
        "--include_descriptors", "--numbers", *numbers)

    identical = True
    for number in numbers:
        for direction in DIRECTIONS:
            filename = f"{number}_rot-{direction}.npy"
            same = np.array_equal(np.load(os.path.join(original_grids, filename)),
                                  np.load(os.path.join(new_dir, "grids", filename)))
            print(f"{filename}: {'identical' if same else 'DIFFERENT'}")
            identical = identical and same

    # the paths differ (absolute vs relative), the file names must match
    original = pd.read_csv(original_csv)
    new = pd.read_csv(os.path.join(new_dir, "database.csv"))
    original["npy_path"] = [os.path.basename(p.replace("\\", "/")) for p in original["npy_path"]]
    new["npy_path"] = [os.path.basename(p) for p in new["npy_path"]]
    same = original.equals(new)
    print(f"database.csv: {'identical' if same else 'DIFFERENT'} ({len(new)} rows)")
    identical = identical and same

    if not identical:
        raise SystemExit(f"Outputs differ, files kept in {workdir}")
    shutil.rmtree(workdir)
    print("generate_grids.py output is identical to the original pipeline")


if __name__ == "__main__":
    main()
