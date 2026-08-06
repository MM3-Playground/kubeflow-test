# DataLad + MLflow + Prefect local ProcessWorker demo

This repository demonstrates four workflows on a CPU-only test VM:

1. upload the user-prepared `train_datalad.txt`, `val_datalad.txt`, and `test_datalad.txt` files to MLflow;
2. start a new training run and automatically evaluate it;
3. reproduce a prior MLflow training run using its recorded code/data commits and conditioned manifests;
4. retrain with changed settings or a changed DataLad version, evaluate, register the model, and promote it when the accuracy threshold passes.

The default `minimum_accuracy` is `0.0`, so every successfully evaluated demo model is accepted and the promotion alias is moved.

## Architecture

The worker VM is intentionally generic. It initially contains only:

```text
compose.yaml
.env
persistent workspace directory
```

It does **not** need this repository checked out.

```text
Developer machine                         Test VM
-----------------                         -------
repository checkout                       Prefect ProcessWorker container
prefect deploy --all  ────────────────►    pulls this repository per run
                                          installs requirements-worker.txt
Prefect UI Run       ────────────────►     executes flow in subprocess
                                          clones training code + DataLad data
                                          trains/evaluates on CPU
                                          logs/registers/promotes in MLflow
```

Prefect deployment pull steps clone this repository for each run and install its Python dependencies. The generic worker only installs `git`, `git-annex`, and CA certificates before starting.

---

## One-time setup on the developer/user machine

The developer machine contains the repository checkout and performs the first deployment registration.

### 1. Install only the deployment tooling

A full training environment is not needed locally. Use a temporary virtual environment, pipx, or uv. For example:

```bash
python -m venv .deploy-venv
source .deploy-venv/bin/activate
python -m pip install "prefect>=3,<4"
```

Configure the local Prefect client:

```bash
export PREFECT_API_URL="http://PREFECT_HOST:4200/api"
export PREFECT_API_AUTH_STRING="admin:replace-me"
```

### 2. Create the Prefect GitHub Secret block

```bash
export GITHUB_READ_TOKEN="github_pat_..."
python scripts/create_git_secret.py github-read-token --env GITHUB_READ_TOKEN
unset GITHUB_READ_TOKEN
```

This stores the token in the Prefect backend as the Secret block expected by `prefect.yaml`:

```text
github-read-token
```

The worker still has `GITHUB_READ_TOKEN` in its `.env` because pipeline tasks may clone a separate private training-code repository and a private DataLad repository.

### 3. Push this repository first

The worker pulls the pipeline from Git, so commit and push the final repository before deploying:

```bash
git add .
git commit -m "Add Prefect DataLad MLflow demo"
git push
```

### 4. Register deployments from the local checkout

Set the URL and branch of **this pipeline repository**:

```bash
export PIPELINE_REPOSITORY_URL="https://github.com/ORG/mlflow-prefect-test.git"
export PIPELINE_REPOSITORY_BRANCH="main"

prefect deploy --all
```

`prefect deploy` registers these pull instructions; it does not copy the local repository to the VM. On each flow run, the ProcessWorker:

1. clones `PIPELINE_REPOSITORY_URL`;
2. installs `requirements-worker.txt` from that clone;
3. changes into the cloned directory;
4. imports and executes the selected flow.

After deployment, the UI should show:

```text
upload-manifest-bundle/upload-manifest-bundle
new-training-and-evaluation/new-local-training
reproduce-training-and-evaluation/reproduce-local-training
retrain-and-evaluate/retrain-local-model
```

---

## D. Upload the initial manifests

The three original manifest files are prepared by the user before the first run:

```text
train_datalad.txt
val_datalad.txt
test_datalad.txt
```

They contain one tab-separated path/label pair per line:

```text
/path/to/image.png<TAB>0
```

Because files on a laptop cannot be passed as filesystem paths to a VM worker, upload them to MLflow first from the user machine:

```bash
cp configs/manifests.example.json configs/manifests.local.json
# Edit dataset_root and the three local manifest paths.

export MLFLOW_TRACKING_URI="http://<host>:<port>/"
export MLFLOW_TRACKING_USERNAME="<username>"
export MLFLOW_TRACKING_PASSWORD="<password>"

python run_pipeline.py manifests --config configs/manifests.local.json
```

This creates a small MLflow run and prints `manifest_source_run_id`. The manifests are converted to portable dataset-relative paths before upload.

Use that run ID in the new/retrain flow parameters:

```json
{
  "manifest_source_run_id": "MLFLOW_RUN_ID"
}
```

Training itself still starts from the Prefect UI and runs on the VM.

---

## E. Start workflows from the Prefect UI

The deployment asks for:
- `code_repo_url`
- `dataset_repo_url`
- `settings`
- optional `code_commit`
- optional `dataset_commit`

For a manifest bundle, put `manifest_source_run_id` inside `settings` and omit
`train_paths_file`, `val_paths_file`, and `test_paths_file`.

Example `settings`:

```json
{
  "save_dir": "/workspace/runs/demo",
  "manifest_source_run_id": "YOUR_MANIFEST_RUN_ID",
  "mlflow_workspace": "YOUR_WORKSPACE",
  "mlflow_experiment": "YOUR_EXPERIMENT",
  "run_name": "cpu-demo",
  "model": "ours",
  "image_size": 512,
  "batch_size": 1,
  "workers": 0,
  "n_epochs": 2,
  "lr": 0.001,
  "factor": 0.9,
  "patience": 5,
  "device": "cpu",
  "minimum_accuracy": 0.0,
  "registered_model_name": "anime-attributor",
  "candidate_alias": "candidate",
  "promotion_alias": "champion",
  "promote_on_pass": true
}
```

The new and retrain workflows use original `*_datalad.txt` manifests. Reproduction downloads and reuses the conditioned train/validation manifests logged by the source training run.

---

## F. Dependency model

The generic Compose worker does not know a project’s Python dependencies in advance. The repository owns them in:

```text
requirements-worker.txt
pyproject.toml
```

For this demo, `requirements-worker.txt` is the runtime source used by the Prefect pull step. It explicitly includes CPU PyTorch and all training/evaluation dependencies.

Every flow run currently invokes `pip_install_requirements`. The Docker pip cache reduces repeated downloads, but packages are installed into the shared worker container environment. This is acceptable for a single-project demo but not ideal for concurrent heterogeneous projects. Later, use a Docker/Kubernetes worker with a project-specific image per flow run.

---

## G. Model registration and promotion

After evaluation, the pipeline:

1. registers the MLflow model version;
2. assigns the `candidate` alias;
3. compares evaluation accuracy with `minimum_accuracy`;
4. moves the promotion alias, default `champion`, when accepted.

For the demo:

```json
"minimum_accuracy": 0.0
```

accepts every successful evaluation. Increase it later, for example to `0.90`.

---

## H. Important files

```text
compose.worker-demo.yaml    generic worker VM service
.env.demo.example           worker-side environment template
prefect.yaml                deployments and runtime pull/install steps
requirements-worker.txt     runtime dependencies, including CPU PyTorch
pipeline/flows.py           new/reproduce/retrain/evaluate/register flows
pipeline/helpers.py         private HTTPS Git/DataLad helpers
train_local.py              CPU/single-process training entry point
eval.py                     local evaluation entry point
configs/                    example parameter and manifest configurations
```

## Redeploy after changing flow parameters

From a developer checkout, not from the worker VM:

```bash
git add .
git commit -m "Simplify Prefect worker parameters"
git push

export PREFECT_API_URL="http://PREFECT_HOST:4200/api"
export PREFECT_API_AUTH_STRING="username:password"
export PIPELINE_REPOSITORY_URL="https://github.com/ORG/mlflow-prefect-test.git"
export PIPELINE_REPOSITORY_BRANCH="main"

prefect deploy --all
```

Deployments with the same names are updated in place. Refresh the Prefect UI before opening the run form.
The generic worker does not need to be restarted because it pulls the repository for each run.
