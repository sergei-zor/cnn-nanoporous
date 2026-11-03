from io import BytesIO

from fastapi import HTTPException, Request
from starlette.datastructures import UploadFile

from app.main import app, health, predict

# A SageMaker endpoint calls GET /ping and POST /invocations on port 8080.
# Both reuse the existing /health and /predict, which keep working as before.


@app.get("/ping")
async def ping():
    status = await health()
    if not status["model_loaded"]:
        raise HTTPException(503, "Model not loaded")
    return status


@app.post("/invocations")
async def invocations(request: Request):
    # the request body is the raw .npy file, /predict expects it as an uploaded file
    body = await request.body()
    if not body:
        raise HTTPException(400, "Expected a .npy structure file as the request body")
    return await predict(UploadFile(file=BytesIO(body), filename="structure.npy"))
