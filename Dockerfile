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
COPY sagemaker_workflow/serving.py sagemaker_workflow/serving.py

# SageMaker endpoints 
RUN printf '#!/bin/sh\nexec uvicorn sagemaker_workflow.serving:app --host 0.0.0.0 --port 8080\n' > /usr/local/bin/serve \
    && chmod +x /usr/local/bin/serve

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]