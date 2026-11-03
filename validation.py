import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from tqdm import tqdm


def validate_one_epoch(
    model, dataloader, loss_function, device, use_descriptors=False, writer=None, dataloadertype="train", epoch=0
):
    model.eval()
    if len(dataloader) == 0:
        output = {
            "loss": np.nan,
            "mse": np.nan,
            "mape": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "target": np.nan,
            "pred": np.nan,
            "paths": np.nan,
        }
        return output
    running_loss = 0
    running_count = 0
    allpredictions = []
    allpaths = []
    targets = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Val"):
            inputs = data[0].to(device)
            target = data[1].to(device)
            if use_descriptors:
                desc = data[3].to(device)
                y_pred = model(inputs, desc)
            else:
                y_pred = model(inputs)
            allpaths.append(data[2])
            running_loss += loss_function(y_pred, target.unsqueeze(1))
            running_count += len(target)
            allpredictions += y_pred.cpu().detach().tolist()
            targets += target.cpu().detach().tolist()
    model.train()
    allpredictions = np.array(allpredictions)[:, 0]
    loss = (running_loss / running_count).detach().cpu().item()
    return summarize_predictions(targets, allpredictions, loss, allpaths, writer, dataloadertype, epoch)


def summarize_predictions(targets, allpredictions, loss, allpaths=None, writer=None, dataloadertype="train", epoch=0):
    targets = np.array(targets)
    allpredictions = np.array(allpredictions)
    allpaths = allpaths if allpaths is not None else []
    try:
        _r2 = r2_score(targets, allpredictions)
    except:
        _r2 = np.nan
    try:
        _mse = mean_squared_error(targets, allpredictions)
        _mape = mean_absolute_percentage_error(targets, allpredictions)
        _mae = mean_absolute_error(targets, allpredictions)
    except:
        _mse = np.nan
        _mape = np.nan
        _mae = np.nan
    output = {
        "loss": loss,
        "mse": _mse,
        "mape": _mape,
        "mae": _mae,
        "r2": _r2,
        "target": targets.tolist(),
        "pred": allpredictions.tolist(),
        "paths": allpaths,
    }
    if writer:
        writer.add_scalar(f"Loss/{dataloadertype}", loss, epoch)
        writer.add_scalar(f"MAPE/{dataloadertype}", _mape, epoch)
        writer.add_scalar(f"MAE/{dataloadertype}", _mae, epoch)
        writer.add_scalar(f"R2/{dataloadertype}", _r2, epoch)
    return output
