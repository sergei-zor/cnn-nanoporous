import os
import tempfile
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, HTTPException
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import DownSample
from data import MDDataset

from app.model import load_inference_model
from app.preprocessing import check_npy_cubic_and_equalsize 

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_DenseNet201.pth")
MODEL_NAME = os.environ.get("MODEL_NAME", "densenet-201")
TARGET_SIZE = int(os.environ.get("TARGET_SIZE", 80))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = None
fully_connected = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, fully_connected
    print(f"Loading {MODEL_NAME} on device {device}")
    model, fully_connected = load_inference_model(MODEL_PATH, MODEL_NAME, device)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(device)}

@app.post("/predict")
async def predict(file: UploadFile):
    if not file.filename.endswith(".npy"):
        raise HTTPException(400, "Expected a .npy structure file")

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        database = pd.DataFrame({"npy_path": [tmp_path], "cii": [np.nan]})

        try:
            npy_size = check_npy_cubic_and_equalsize(df=database)
        except Exception as e:
            raise HTTPException(400, f"Unsupported structure file: {e}")

        downsampler = DownSample(scale_factor=TARGET_SIZE / npy_size)
        transform = transforms.Compose([transforms.ToTensor(), downsampler])

        dataset = MDDataset(df=database, transform=transform, use_descriptors=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        data = next(iter(loader))

        inputs = data[0].to(device)
        with torch.no_grad():
            embedding = model(inputs)
            y_pred = torch.squeeze(fully_connected(embedding))

        return {
            "prediction": float(y_pred.item()),  
            "model_name": MODEL_NAME,
            "native_size": npy_size,
            "target_size": TARGET_SIZE,
        }
    finally:
        os.remove(tmp_path)  