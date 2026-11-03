# SageMaker workflow

This folder runs the workflow of the repository on Amazon SageMaker for Dataset 1:

1. Generating the training grids in a Processing job
2. Training DenseNet-201 on a GPU instance
3. Serving the trained model 

The scripts below are started locally, AWS only runs the jobs.

- `generate_grids.py` - faster version of the grid generation and database preparation scripts
- `verify_generate_grids.py` - checks that `generate_grids.py` gives exactly the same grids as the original scripts
- `launch_processing.py` - starts the grid generation job
- `train_entry.py` - runs `train.py` inside the SageMaker training container
- `metrics.py` - values in the SageMaker console
- `local_smoke_test.py` - runs `train_entry.py` locally 
- `launch_training.py` - starts the training job
- `serving.py` - adds the routes SageMaker needs to the FastAPI 
- `deploy_endpoint.py` - deploys the trained model as an endpoint
- `invoke_endpoint.py` - sends structures to the endpoint

## AWS setup

The region is (`eu-west-1`):

- An IAM user for the local machine, with access keys saved under the profile `nanoporous` (`aws configure --profile nanoporous`)
- A monthly budget with e-mail alerts
- Service quotas for `ml.g5.xlarge for training job usage` and `ml.g5.xlarge for spot training job usage` (new accounts have 0)
- A private S3 bucket `nanoporous-sagemaker-sz-euw1`, with the dataset uploaded to `raw/Dataset1_gold_0.25.xlsx`
- A role `SageMakerExecutionRole-nanoporous` with `AmazonSageMakerFullAccess` and access to the bucket. This is the identity for the jobs.
- A private ECR repository `nanoporous-api` for the API image

A Python environment for the SageMaker SDK outside the repo:
```
python -m venv $HOME\venvs\sagemaker
& $HOME\venvs\sagemaker\Scripts\python -m pip install "sagemaker>=2.230,<3"
```
In every new terminal:
```
$env:AWS_PROFILE = "nanoporous"
$env:SAGEMAKER_SUPPRESS_V2_WARNING = "1"
$PY = "$HOME\venvs\sagemaker\Scripts\python"
```

## 1. Training data generation

The original scripts write a LAMMPS file with about 2M atoms for every structure and read it back.
`generate_grids.py` keeps the atoms in memory, uses numpy for the voxelization and runs on all CPU cores.
The output is identical, which can be checked with:
```
python sagemaker_workflow/verify_generate_grids.py
```

A Processing job downloads the dataset from S3, runs `generate_grids.py` and uploads the result to
`s3://nanoporous-sagemaker-sz-euw1/database/dataset1/`:
```
grids/<number>_rot-<x|y|z>.npy
database.csv
```
Commands:
```
& $PY sagemaker_workflow/launch_processing.py --mode verify
& $PY sagemaker_workflow/launch_processing.py --mode generate --max_structures 20 --output_prefix database/test20
& $PY sagemaker_workflow/launch_processing.py --mode generate --instance_count 2 --no_wait
```
The first one runs the identity check on AWS, the second is a small test, the last one generates all 5000 structures
(15000 grids). With 2 `ml.t3.xlarge` instances it takes about 1 hour.

## 2. Training

A training job starts a GPU instance with the AWS PyTorch image, downloads the grids, runs `train_entry.py`
and uploads the results. `train_entry.py` converts the SageMaker inputs into the arguments `train.py` expects and copies
the results into the folders SageMaker uploads:
```
/opt/ml/input/data/training/  <- s3://.../database/dataset1/
/opt/ml/model/                -> s3://.../training/<job>/output/model.tar.gz   (best_model.pth of each fold)
/opt/ml/output/data/          -> s3://.../training/<job>/output/output.tar.gz  (metrics, history, tensorboard)
/opt/ml/checkpoints/          <-> s3://.../checkpoints/<job>/                  (synced during training)
```
Local test (CPU):
```
python sagemaker_workflow/local_smoke_test.py
```
Test on AWS:
```
& $PY sagemaker_workflow/launch_training.py --n_epochs 2
& $PY sagemaker_workflow/launch_training.py --spot --no_wait
```
The first is a short test, the second trains fold 0 for up to 100 epochs on a spot instance.
Other folds are trained with `--folds 1` and so on.

To continue a stopped job on a normal instance instead of a spot:
```
& $PY sagemaker_workflow/launch_training.py --resume_job <job-name> --lr 0.0001 --no_wait
```
If `--lr` is different from the checkpoint, the new learning rate is used and the early stopping counter starts again.

After job is done, the model can be downloaded with:
```
aws s3 cp s3://nanoporous-sagemaker-sz-euw1/training/<job-name>/output/model.tar.gz sagemaker_artifacts/
tar -xzf sagemaker_artifacts/model.tar.gz -C sagemaker_artifacts
```
to delete checkpoints: `aws s3 rm s3://nanoporous-sagemaker-sz-euw1/checkpoints/ --recursive`.

### Results

Fold 0 achieved the best model at epoch 99, ~6 minutes per epoch on `ml.g5.xlarge`.

| Set | R2 | MAE |
|---|---|---|
| Train | 0.976 | 0.074 |
| Validation | 0.961 | 0.095 |
| Test | 0.961 | 0.095 |

## 3. Serving

The endpoint runs the same Docker image as the API. `serving.py` adds two routes SageMaker calls:

- `GET /ping` - same as `/health`
- `POST /invocations` - same as `/predict`, with the `.npy` file as the request body

The model is not part of the image, SageMaker downloads the `model.tar.gz` of the training job when the endpoint starts.

Local test:
```
$env:MODEL_PATH = "sagemaker_artifacts/fold_0/best_model.pth"
uvicorn sagemaker_workflow.serving:app --port 8080
curl -X POST --data-binary "@exemplary_structures/155_rot-z.npy" http://localhost:8080/invocations
```

### Pushing the image to ECR

The image is built and pushed by GitHub Actions (`.github/workflows/push-image.yml`) on every push to `master`.
It logs in to AWS with OpenID Connect, no AWS keys are stored:

1. IAM - Identity providers - Add provider - OpenID Connect
2. IAM - Roles - Create role - Web identity 
3. Add an inline policy to the role that allows pushing to the repository 
4. GitHub - Settings - Secrets and variables - Actions - new secret `AWS_ACCOUNT_ID`

### Deploying

```
& $PY sagemaker_workflow/deploy_endpoint.py
& $PY sagemaker_workflow/invoke_endpoint.py exemplary_structures/155_rot-z.npy exemplary_structures/419_rot-x.npy
& $PY sagemaker_workflow/deploy_endpoint.py --delete
```