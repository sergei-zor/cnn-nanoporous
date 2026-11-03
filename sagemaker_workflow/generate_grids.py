import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import special

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database_preparation.npy_rotations import rotate_array

# Does the same as grid_generator.py + npy_rotations.py + prepare_database.py (checked by verify_generate_grids.py),
# but keeps the atoms in memory instead of writing .lmp files, uses numpy instead of loops over atoms,
# and runs on all CPU cores
PHASE_COLUMNS = [f"Phase {i}" for i in range(1, 31)]
DIRECTIONS = ["x", "y", "z"]
WAVE_NUMBER_SQUARED = 9
FCC_BASIS = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset .xlsx file")
    parser.add_argument("--output_dir", type=str, required=True, help="grids/ and database.csv are saved here")
    parser.add_argument("--grid_resolution", type=int, default=80)
    parser.add_argument("--cell_constant", type=float, default=4.0701105)
    parser.add_argument("--n_cells", type=int, default=125)
    parser.add_argument("--solid_fraction", type=float, default=0.25)
    parser.add_argument("--include_descriptors", action="store_true")
    parser.add_argument("--numbers", nargs="+", type=int, default=None, help="Only these structure numbers")
    parser.add_argument("--max_structures", type=int, default=0, help="Only the first N structures, 0 = all")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    return vars(parser.parse_args())


def wave_directions():
    # all integer (h, k, l) with h^2 + k^2 + l^2 = 9, in the same order as grid_generator.py
    d = round(math.sqrt(WAVE_NUMBER_SQUARED))
    r = range(-d, d + 1)
    return np.array([[h, k, l] for h in r for k in r for l in r if h * h + k * k + l * l == WAVE_NUMBER_SQUARED])


def solid_atoms(phases, a, n_cells, solid_fraction):
    # FCC lattice sites, cell (i, j, k) starts at ((i+1)a, (j+1)a, (k+1)a) as in grid_generator.py
    cell_origins = (np.indices((n_cells, n_cells, n_cells)).reshape(3, -1).T + 1) * a
    sites = np.concatenate([cell_origins + a * offset for offset in FCC_BASIS])

    # a site is solid where the sum of standing waves is below epsilon
    directions = wave_directions()
    wave_vectors = 2 * math.pi * directions / (n_cells * a)
    field = np.zeros(len(sites))
    for wave_vector, phase in zip(wave_vectors, phases):
        field += np.cos(sites @ wave_vector + phase)
    field *= math.sqrt(2 / len(directions))
    epsilon = math.sqrt(2) * special.erfinv(2 * solid_fraction - 1)
    return sites[field < epsilon]


def voxelize(atoms, box_length, resolution):
    # a voxel is True if it contains at least one atom
    indices = (atoms * (resolution / box_length)).astype(int)
    indices = indices[np.all(indices < resolution, axis=1)]
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return grid


def generate_structure(task):
    number, phases, args = task
    atoms = solid_atoms(phases, args["cell_constant"], args["n_cells"], args["solid_fraction"])
    grid = voxelize(atoms, args["n_cells"] * args["cell_constant"], args["grid_resolution"])
    for direction in DIRECTIONS:
        path = os.path.join(args["output_dir"], "grids", f"{number}_rot-{direction}.npy")
        np.save(path, rotate_array(grid, direction))


def get_host_index():
    # with several Processing instances, each one takes every n-th structure
    config_path = "/opt/ml/config/resourceconfig.json"
    if not os.path.exists(config_path):
        return 0, 1
    with open(config_path) as f:
        config = json.load(f)
    return config["hosts"].index(config["current_host"]), len(config["hosts"])


def write_database(dataset, numbers, output_dir, include_descriptors):
    # same rows and order as prepare_database.py, but with paths relative to output_dir
    rows = dataset[dataset["Number"].isin(numbers)].copy()
    rows["direction"] = rows["Direction"].str.lower()
    rows["order"] = rows["direction"].map({"x": 0, "y": 1, "z": 2})
    rows = rows.sort_values(["order", "Number"])
    database = pd.DataFrame({"npy_path": "grids/" + rows["Number"].astype(str) + "_rot-" + rows["direction"] + ".npy",
                             "cii": rows["Cii"]})
    if include_descriptors:
        descriptor_columns = dataset.columns[32:-1]
        database = pd.concat([database, rows[descriptor_columns]], axis=1)
    database.to_csv(os.path.join(output_dir, "database.csv"), index=False)
    print(f"database.csv with {len(database)} rows")


def main():
    args = parse_command_line_args()
    dataset = pd.read_excel(args["dataset"])
    structures = dataset.drop_duplicates("Number").sort_values("Number")
    if args["numbers"]:
        structures = structures[structures["Number"].isin(args["numbers"])]
    if args["max_structures"] > 0:
        structures = structures.head(args["max_structures"])
    numbers = structures["Number"].tolist()
    os.makedirs(os.path.join(args["output_dir"], "grids"), exist_ok=True)

    host_index, n_hosts = get_host_index()
    if host_index == 0:
        write_database(dataset, numbers, args["output_dir"], args["include_descriptors"])
    tasks = [(number, phases, args) for number, phases in zip(numbers, structures[PHASE_COLUMNS].values)]
    tasks = tasks[host_index::n_hosts]

    print(f"Host {host_index + 1}/{n_hosts}: {len(tasks)} structures on {args['workers']} workers")
    start = time.time()
    with Pool(args["workers"]) as pool:
        for i, _ in enumerate(pool.imap_unordered(generate_structure, tasks), start=1):
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - start
                print(f"{i}/{len(tasks)} structures, {elapsed:.0f}s, {elapsed / i:.2f}s per structure")


if __name__ == "__main__":
    main()
