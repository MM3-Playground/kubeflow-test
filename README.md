# DataLad + Kubeflow Pipelines demo

## Install locally to compile pipeline YAML

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. First-time manifest upload

The `*_datalad.txt` files remain local and are ignored by Git.

```bash
cp configs/manifests.example.json configs/manifests.local.json
# edit the paths
python run_pipeline.py manifests --config configs/manifests.local.json
```

This creates:

```text
compiled/upload-manifest-bundle.yaml
```

Upload and run it in KFP. It creates a `Dataset` artifact containing a ZIP with the three portable manifests. Copy that artifact's URI and use it as `manifest_bundle_uri`.

## 2. Compile pipelines

```bash
python run_pipeline.py new
python run_pipeline.py reproduce
python run_pipeline.py retrain
```

The config files are examples/reference values. The generated KFP pipeline exposes the corresponding values directly as run parameters in the UI.

## New training graph

```text
resolve_sources
  ├── code_source Artifact
  └── DataLad Dataset artifact
           +
     manifest Dataset
           |
           v
       train_local
       ├── Model
       ├── checkpoints
       ├── train/val Dataset artifacts
       ├── Metrics
       ├── training history
       ├── metadata
       └── system metrics
           |
           v
      evaluate_local
       ├── test Dataset
       ├── accuracy Metrics
       ├── ClassificationMetrics
       ├── result archive
       ├── metadata
       └── system metrics
           |
           v
    Kubeflow Model Registry
```

## Reproduction

For exact reproduction, supply the exact source `code_commit`, `dataset_commit`, and the source run's conditioned-manifest bundle URI. KFP also supports cloning/rerunning a prior run from the UI.

## Important differences from MLflow

KFP does not have one generic `mlflow.autolog()` equivalent, and KFP scalar metrics do not provide MLflow's `step=epoch` metric-history interface. Therefore this project logs scalar summaries using native `Metrics` and preserves the complete per-epoch series as a tracked JSON artifact.

Kubeflow Model Registry does not provide the same registered-model alias API used by MLflow (`candidate`, `champion`). The POC records `accepted` and `promoted` metadata instead.

KFP itself also does not provide MLflow's built-in system-metric logging API. The components sample CPU/RSS every five seconds and persist the samples as KFP artifacts.


## GitHub credentials for private code and DataLad repositories

Create this Secret in every Kubeflow Profile namespace that runs the pipeline:

```bash
kubectl create secret generic github-credentials \
  -n my-profile \
  --from-literal=username=x-access-token \
  --from-literal=token="$GITHUB_PAT"
```

The reusable pipelines expect exactly:

```text
Secret: github-credentials
keys:
  username
  token
```

The PAT is not a KFP parameter. KFP injects it only into `resolve_sources`,
`train_local`, and `evaluate_local`. Those components use a temporary
`GIT_ASKPASS` script, so the PAT is not embedded in the repository URL or Git
command line. The same credential environment is used by `git`, `datalad clone`,
and `datalad get`.

For an organization-managed token, use the narrowest read-only GitHub
fine-grained PAT that can access the code and DataLad repositories.

## Simplified compilation

Only manifest upload needs a local JSON configuration:

```bash
python run_pipeline.py manifests --config configs/manifests.example.json
```

The reusable pipelines need no config file:

```bash
python run_pipeline.py new
python run_pipeline.py reproduce
python run_pipeline.py retrain
```

Experiment-specific values are KFP run parameters entered/stored with each run.

## DataLad manifest paths and runtime download location

The portable manifests should identify files relative to the root of the
DataLad dataset whenever possible. Example:

```text
train/real/image001.png	0
train/fake/image002.png	1
```

`configs/manifests.example.json` has `dataset_root`, which is the local DataLad
checkout used only while preparing the portable manifest bundle. If an input
manifest contains absolute paths underneath `dataset_root`, the upload helper
converts them to relative paths automatically.

At runtime, this POC materializes the DataLad repository inside each KFP task:

```text
resolve_sources Pod: /tmp/data
train_local Pod:     /tmp/data
evaluate_local Pod:  /tmp/data
```

Therefore `train/real/image001.png` becomes
`/tmp/data/train/real/image001.png` inside a training/evaluation Pod.

These `/tmp/data` directories are Pod-local and are not persistent/shared.
`train_local` and `evaluate_local` each perform their own `datalad clone` and
`datalad get` for now. A future `persistent-datasets` PVC can replace this
without changing the portable manifest format.


## Runtime image

The KFP source/train/evaluation tasks use the image specified when the pipeline is compiled:

```bash
export KUBEFLOW_RUNTIME_IMAGE=YOUR_REGISTRY/kubeflow-runtime:TAG
python run_pipeline.py new
python run_pipeline.py reproduce
python run_pipeline.py retrain
```

On PowerShell:

```powershell
$env:KUBEFLOW_RUNTIME_IMAGE = "YOUR_REGISTRY/kubeflow-runtime:TAG"
uv run run_pipeline.py new
```

The runtime image is expected to already contain Python 3.11, Git, git-annex, and DataLad. The pipeline does **not** run `apt-get` or install DataLad. It still clones the selected code repository and exact commit. Training/evaluation create `/tmp/venv`, install that checked-out repository's `requirements.txt`, and execute its `train_local.py` / `eval.py` with the venv Python.

Private image pulls use `Secret/regcred`; private Git/DataLad access uses `Secret/github-credentials`.
