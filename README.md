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
python run_pipeline.py new --config configs/new.example.json
python run_pipeline.py reproduce --config configs/reproduce.example.json
python run_pipeline.py retrain --config configs/retrain.example.json
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