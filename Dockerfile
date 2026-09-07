FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu


COPY app/ app/
COPY TRIDCNNPyTorch/ TRIDCNNPyTorch/
COPY generate_model.py .
COPY utils.py .
COPY data.py .
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]