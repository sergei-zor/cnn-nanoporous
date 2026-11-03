import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
from training import get_optimizer, CustomLoss

from generate_model import generate_model, ALLOWED_MODELS
from inputoutput import get_experiment_name, save_to_pickle, save_to_json
from utils import DownSample, EarlyStopper, RandomRoll, RandomFlip
from data import (
    MDDataset,
    DescDimReducer,
    split_into_folds_stratification,
    split_into_folds_nostratification,
    get_npy_shape,
)
from torch.utils.tensorboard import SummaryWriter
from validation import validate_one_epoch, summarize_predictions


def check_npy_cubic_and_equalsize(df: pd.DataFrame) -> int:
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


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="md_database.csv", help="Path to database .csv file")
    parser.add_argument("--outputdir", type=str, default="CNN/results", help="Path where results are stored")
    parser.add_argument("--model_name", choices=ALLOWED_MODELS, default=ALLOWED_MODELS[0], help="Model type")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--size", type=int, default=125)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--loss", choices=["mse", "huber", "l1"], default="mse", help="Which loss function to use")
    parser.add_argument("--optimizer", choices=["SGD", "Adam"], default="Adam", help="Which optimier to use")
    parser.add_argument("--aug_roll_ratio", type=float, default=0.0, help="RandomRoll augmentation")
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test only")
    parser.add_argument("--suffix", type=str, default="", help="Suffix appended to filename")
    parser.add_argument("--use_descriptors", action="store_true")
    parser.add_argument("--no_structure", action="store_true")
    parser.add_argument("--use_stratification", action="store_true")
    parser.add_argument("--fc_first_size", nargs="+", type=int, default=[33, 33, 21])
    parser.add_argument("--fc_second_size", nargs="+", type=int, default=[32, 32, 32])
    parser.add_argument("--p_dropout", type=float, default=0.3, help="Dropout for fullconnected")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--folds", nargs="+", type=int, default=[0], help="Which folds to use")
    parser.add_argument("--p_flip", type=float, default=0.0, help="Flip probability")
    parser.add_argument("--gradient_clip", type=float, default=0.0, help="Gradient clip value")
    parser.add_argument(
        "--negative_prediction_penalty",
        type=float,
        default=0.0,
        help="Prefactor for additional L2 penalty of negative predictions",
    )
    parser.add_argument("--no_testset", action="store_true")
    parser.add_argument("--evaluationset", type=str, default="")
    parser.add_argument("--remove_model", action="store_true")
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--scale_only", action="store_true")
    parser.add_argument("--n_limit", type=int, default=0)
    parser.add_argument(
        "--skip_train_eval",
        action="store_true",
        help="Report train metrics from the training pass instead of re-evaluating the whole train set every epoch",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="", help="Save a checkpoint every epoch here and resume from it if present"
    )
    return vars(parser.parse_args())


def save_checkpoint(state, path):
    """Writes to a temporary file first, so an interruption never leaves a half-written checkpoint behind."""
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    args = parse_command_line_args()
    seed = args["seed"]
    batch_size = args["batch_size"]
    n_epochs = args["n_epochs"]
    outputdir = args["outputdir"]
    model_name = args["model_name"]
    loss_name = args["loss"]
    roll_ratio = args["aug_roll_ratio"]
    smoke_test = args["smoke_test"]
    use_descriptors = args["use_descriptors"]
    no_structure = args["no_structure"]
    use_stratification = args["use_stratification"]
    optimizer_type = args["optimizer"]
    fc_first_size = args["fc_first_size"]
    fc_second_size = args["fc_second_size"]
    p_dropout = args["p_dropout"]
    cv_folds = args["cv_folds"]
    which_folds = args["folds"]
    learning_rate = args["lr"]
    p_flip = args["p_flip"]
    target_size = args["size"]
    negative_prediction_penalty = args["negative_prediction_penalty"]
    no_testset = args["no_testset"]
    gradient_clip = args["gradient_clip"]
    remove_model = args["remove_model"]
    evaluationset = args["evaluationset"] if args["evaluationset"] != "" else None
    use_pca = args["use_pca"]
    scale_only = args["scale_only"]
    n_limit = args["n_limit"]
    skip_train_eval = args["skip_train_eval"]
    checkpoint_dir = args["checkpoint_dir"]
    pca_n_components = fc_first_size[0]

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Will run on device={device}")
    database = pd.read_csv(args["database"])

    if n_limit > 0:
        database = database.sample(n_limit).reset_index(drop=True)

    npy_size = check_npy_cubic_and_equalsize(df=database)
    print(f"Structures are cubic: {npy_size}x{npy_size}x{npy_size}")
    print(f"Structures will be rescaled to: {target_size}x{target_size}x{target_size}")

    if evaluationset:
        df_eval = pd.read_csv(evaluationset)
        npy_size_eval = check_npy_cubic_and_equalsize(df=df_eval)
        assert (
            npy_size == npy_size_eval
        ), f"Evaluation dataset have structures of different size. {npy_size} expected. {npy_size_eval} found"

    ### shuffle the database
    database = database.sample(frac=1)

    if smoke_test:
        database = database.head(200)
        n_epochs = 2
        batch_size = 10
        cv_folds = 3
        which_folds = [0, 1]

    experiment_name = get_experiment_name(args)

    downsampler = DownSample(scale_factor=target_size / npy_size)
    randomroll = RandomRoll(ratio_y=roll_ratio, ratio_z=roll_ratio)
    randomflip = RandomFlip(p_flip=p_flip)
    transform_train = transforms.Compose([transforms.ToTensor(), downsampler, randomflip, randomroll])
    transform_test = transforms.Compose([transforms.ToTensor(), downsampler])

    split_into_folds = split_into_folds_stratification if use_stratification else split_into_folds_nostratification

    for df_train, df_val, df_test, fold_id in split_into_folds(df=database, cv_folds=cv_folds, which_folds=which_folds):
        if no_testset:
            df_train = pd.concat([df_train, df_test])
            df_test = df_test.iloc[0:0]
        print(
            f"Fold: {fold_id} Dataset size. Train={len(df_train)} Val={len(df_val)} Test={len(df_test)}  Total={len(database)}"
        )
        datadimreducer = None
        if use_descriptors and (use_pca or scale_only):
            datadimreducer = DescDimReducer(df=df_train, pca_n_components=pca_n_components, scale_only=scale_only)

        trainset = MDDataset(
            df=df_train, transform=transform_train, use_descriptors=use_descriptors, descriptor_transform=datadimreducer
        )
        valset = MDDataset(
            df=df_val, transform=transform_test, use_descriptors=use_descriptors, descriptor_transform=datadimreducer
        )
        testset = MDDataset(
            df=df_test, transform=transform_test, use_descriptors=use_descriptors, descriptor_transform=datadimreducer
        )
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        if evaluationset:
            evalset = MDDataset(
                df_eval, transform=transform_test, use_descriptors=use_descriptors, descriptor_transform=datadimreducer
            )
            evalloader = DataLoader(evalset, batch_size=batch_size, shuffle=False)

        model = generate_model(
            model_name=model_name,
            use_descriptors=use_descriptors,
            no_structure=no_structure,
            fc_first_size=fc_first_size,
            fc_second_size=fc_second_size,
            p_dropout=p_dropout,
        )
        model.to(device)

        loss_fn = CustomLoss(loss_name=loss_name, penalty=negative_prediction_penalty)
        optimizer = get_optimizer(model=model, optimizer_type=optimizer_type, lr=learning_rate)
        experiment_path = os.path.join(outputdir, experiment_name, f"fold_{fold_id}")
        model_path = os.path.join(experiment_path, "best_model.pth")
        os.makedirs(experiment_path, exist_ok=True)
        save_to_json(args, os.path.join(experiment_path, "parameters.json"))

        history = []
        writer = SummaryWriter(os.path.join(experiment_path, "tensorboard"))
        early_stopper = EarlyStopper(patience=10, model_path=model_path)

        start_epoch = 0
        finished = False
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold_id}_checkpoint.pth")
            best_model_checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold_id}_best_model.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                early_stopper.counter = checkpoint["early_stopper_counter"]
                early_stopper.min_validation_loss = checkpoint["min_validation_loss"]
                # load_state_dict restored the old learning rate; a new --lr wins and gets the full patience
                if optimizer.param_groups[0]["lr"] != learning_rate:
                    print(f"Fold {fold_id}: learning rate {optimizer.param_groups[0]['lr']} -> {learning_rate}")
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate
                    early_stopper.counter = 0
                history = checkpoint["history"]
                trainloss, valloss, testloss = checkpoint["last_losses"]
                epoch = checkpoint["epoch"]
                finished = checkpoint["finished"]
                start_epoch = epoch + 1
                shutil.copy(best_model_checkpoint_path, model_path)
                print(f"Fold {fold_id}: resumed from checkpoint after epoch {epoch}, finished={finished}")

        for epoch in range(start_epoch, start_epoch if finished else n_epochs):
            running_loss = 0
            running_count = 0
            train_predictions = []
            train_targets = []
            for data in tqdm(trainloader, desc="Train"):
                inputs = data[0].to(device)
                targets = data[1].to(device)
                if use_descriptors:
                    descriptors = data[3].to(device)
                    y_pred = model(inputs, descriptors)
                else:
                    y_pred = model(inputs)
                # forward, backward, and then weight update
                loss = loss_fn(y_pred, targets.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                if gradient_clip > 1e-6:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()
                running_loss += loss.item()
                running_count += len(data)
                train_predictions += y_pred.detach().cpu()[:, 0].tolist()
                train_targets += targets.cpu().tolist()
            totalloss = running_loss / running_count

            if skip_train_eval:
                # metrics of the training pass itself: weights change during the epoch, augmentation and dropout are on
                trainloss = summarize_predictions(
                    train_targets,
                    train_predictions,
                    running_loss / len(train_targets),
                    writer=writer,
                    dataloadertype="train",
                    epoch=epoch,
                )
            else:
                trainloss = validate_one_epoch(
                    model=model,
                    dataloader=trainloader,
                    loss_function=loss_fn,
                    device=device,
                    use_descriptors=use_descriptors,
                    writer=writer,
                    dataloadertype="train",
                    epoch=epoch,
                )
            valloss = validate_one_epoch(
                model=model,
                dataloader=valloader,
                loss_function=loss_fn,
                device=device,
                use_descriptors=use_descriptors,
                writer=writer,
                dataloadertype="val",
                epoch=epoch,
            )
            testloss = validate_one_epoch(
                model=model,
                dataloader=testloader,
                loss_function=loss_fn,
                device=device,
                use_descriptors=use_descriptors,
                writer=writer,
                dataloadertype="test",
                epoch=epoch,
            )
            history.append({"epoch": epoch, "trainloss": trainloss, "valloss": valloss, "testloss": testloss})
            print(
                f'Fold {fold_id} Ep {epoch}: train runningloss={totalloss} loss={trainloss["loss"]:.4f} mse={trainloss["mse"]:.4f} r2={trainloss["r2"]:.4f} val loss={valloss["loss"]:.4f} r2={valloss["r2"]:.4f} bestloss: {early_stopper.min_validation_loss:4f} counter={early_stopper.counter} test loss={testloss["loss"]:.4f} r2={testloss["r2"]:.4f}'
            )
            stop = early_stopper.early_stop(valloss["loss"], model=model)
            if checkpoint_dir:
                if early_stopper.counter == 0:  # early_stop just saved a new best model
                    shutil.copy(model_path, best_model_checkpoint_path)
                checkpoint = {
                    "epoch": epoch,
                    "finished": stop,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "early_stopper_counter": early_stopper.counter,
                    "min_validation_loss": early_stopper.min_validation_loss,
                    "history": history,
                    "last_losses": (trainloss, valloss, testloss),
                }
                save_checkpoint(checkpoint, checkpoint_path)
            if stop:
                print("Break due to early stopping")
                break
        save_to_pickle(history, os.path.join(experiment_path, "history.pickle"))
        save_to_json(trainloss, os.path.join(experiment_path, "summary_train.json"))
        save_to_json(valloss, os.path.join(experiment_path, "summary_val.json"))
        save_to_json(testloss, os.path.join(experiment_path, "summary_test.json"))

        print(f"Restoring best model from: {model_path}")
        model.load_state_dict(torch.load(model_path))
        trainloss = validate_one_epoch(
            model=model,
            dataloader=trainloader,
            loss_function=loss_fn,
            device=device,
            use_descriptors=use_descriptors,
            writer=None,
            dataloadertype="train",
            epoch=epoch,
        )
        valloss = validate_one_epoch(
            model=model,
            dataloader=valloader,
            loss_function=loss_fn,
            device=device,
            use_descriptors=use_descriptors,
            writer=None,
            dataloadertype="val",
            epoch=epoch,
        )
        testloss = validate_one_epoch(
            model=model,
            dataloader=testloader,
            loss_function=loss_fn,
            device=device,
            use_descriptors=use_descriptors,
            writer=None,
            dataloadertype="test",
            epoch=epoch,
        )

        save_to_json(trainloss, os.path.join(experiment_path, "summary_train_best_model.json"))
        save_to_json(valloss, os.path.join(experiment_path, "summary_val_best_model.json"))
        save_to_json(testloss, os.path.join(experiment_path, "summary_test_best_model.json"))
        if evaluationset:
            evalloss = validate_one_epoch(
                model=model,
                dataloader=evalloader,
                loss_function=loss_fn,
                device=device,
                use_descriptors=use_descriptors,
                writer=None,
                dataloadertype="eval",
                epoch=epoch,
            )
            save_to_json(evalloss, os.path.join(experiment_path, "summary_evaluation_best_model.json"))
        if remove_model:
            os.remove(os.path.join(experiment_path, "best_model.pth"))


if __name__ == "__main__":
    main()
