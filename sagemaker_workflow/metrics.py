# SageMaker reads these values from the training log and shows them in the console (Algorithm metrics).
# They match the epoch line printed by train.py and the best model line printed by train_entry.py.
NUMBER = r"([-0-9.e+]+|nan)"

METRIC_DEFINITIONS = [
    {"Name": "train:loss", "Regex": rf"train runningloss=\S+ loss={NUMBER}"},
    {"Name": "train:r2", "Regex": rf"train runningloss=\S+ loss=\S+ mse=\S+ r2={NUMBER}"},
    {"Name": "val:loss", "Regex": rf"val loss={NUMBER}"},
    {"Name": "val:r2", "Regex": rf"val loss=\S+ r2={NUMBER}"},
    {"Name": "test:loss", "Regex": rf"test loss={NUMBER}"},
    {"Name": "test:r2", "Regex": rf"test loss=\S+ r2={NUMBER}"},
    {"Name": "best:val_r2", "Regex": rf"best model: .*val_r2={NUMBER}"},
    {"Name": "best:test_r2", "Regex": rf"best model: .*test_r2={NUMBER}"},
]
