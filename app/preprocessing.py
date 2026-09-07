import pandas as pd
from tqdm import tqdm
from data import get_npy_shape


def check_npy_cubic_and_equalsize(df: pd.DataFrame) -> int:
    """
    Checks the shape and size of the candidate structure.
    """
    required_npy_shape = None
    for row_id, row in tqdm(df.iterrows(), total=len(df), desc="Checking dataset"):
        npy_path = row["npy_path"]
        npy_shapes = get_npy_shape(npy_path)
        assert (
            npy_shapes[0] == npy_shapes[1] == npy_shapes[2]
        ), f"Structures must be cubic. Non cubic structure found in {npy_path}"
        if row_id == 0:
            required_npy_shape = npy_shapes[0]
        else:
            assert (
                required_npy_shape == npy_shapes[0]
            ), f"Structure of different resolutions found. Expected {required_npy_shape} but {npy_shapes[0]} found for {npy_path}"
    return required_npy_shape