from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from prefect import flow, get_run_logger, task
from prefect.blocks.system import Secret

from .helpers import (
    datalad_clone_or_update,
    git_clone_or_update,
    load_json,
    materialize_manifest,
    run,
    write_portable_manifest,
)


def _execution_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d%H%M%S')}"


def _worker_git_credentials() -> tuple[str | None, str | None]:
    """Read infrastructure-managed Git credentials from the worker environment."""
    token = Secret.load(os.environ.get("GITHUB_SECRET_NAME")).get()
    if not token:
        return None, None
    return os.environ.get("GITHUB_READ_USERNAME", "x-access-token"), token


def _repo_name(repo_url: str) -> str:
    name = Path(repo_url.rstrip("/").removesuffix(".git")).name
    if not name:
        raise ValueError(f"Cannot derive repository name from URL: {repo_url}")
    return name


def _workspace_path(kind: str, repo_url: str) -> str:
    root = Path(os.environ.get("PIPELINE_WORK_ROOT", "/workspace")).expanduser()
    return str((root / kind / _repo_name(repo_url)).resolve())


def _safe_model_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return value or "prefect-model"


@task(name="prepare-codebase", retries=2, retry_delay_seconds=5)
def prepare_code(repo_url: str, commit: str | None = None) -> dict[str, str]:
    username, token = _worker_git_credentials()
    return git_clone_or_update(
        repo_url,
        _workspace_path("code", repo_url),
        commit,
        username=username,
        token=token,
    )


@task(name="prepare-datalad-dataset", retries=2, retry_delay_seconds=10)
def prepare_dataset(repo_url: str, commit: str | None = None) -> dict[str, str]:
    username, token = _worker_git_credentials()
    return datalad_clone_or_update(
        repo_url,
        _workspace_path("data", repo_url),
        commit,
        username=username,
        token=token,
    )


def _download_artifact(run_id: str, artifact_path: str, destination: Path, workspace: str | None = None) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    if workspace:
        mlflow.set_workspace(workspace)
    downloaded = Path(MlflowClient().download_artifacts(run_id, artifact_path, str(destination)))
    return downloaded.resolve()


@task(name="prepare-original-manifests")
def prepare_original_manifests(dataset: dict[str, str], settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve user manifests from worker-visible files or an MLflow manifest-bundle run."""
    updated = dict(settings)
    save_dir = Path(updated["save_dir"]).expanduser().resolve()
    materialized_dir = save_dir / "input-manifests"
    source_run_id = updated.get("manifest_source_run_id")

    names = {
        "train_paths_file": "train_datalad.txt",
        "val_paths_file": "val_datalad.txt",
        "test_paths_file": "test_datalad.txt",
    }
    for key, filename in names.items():
        configured = updated.get(key)
        if source_run_id:
            portable = _download_artifact(str(source_run_id), f"manifests/{filename}", materialized_dir / "portable", workspace=updated.get("mlflow_workspace"))
            updated[key] = str(materialize_manifest(
                portable, materialized_dir / filename, dataset["path"]
            ))
        else:
            if not configured:
                raise ValueError(f"{key} is required when manifest_source_run_id is not supplied")
            configured_path = Path(configured).expanduser().resolve()
            if not configured_path.exists():
                raise FileNotFoundError(f"Manifest not found on worker VM: {configured_path}")
            updated[key] = str(configured_path)
    return updated


@task(name="train-model")
def train_local(code: dict[str, str], dataset: dict[str, str], settings: dict[str, Any],
                *, use_conditioned_files: bool = False) -> dict[str, Any]:
    logger = get_run_logger()
    execution_id = _execution_id("train")
    save_dir = str(Path(settings["save_dir"]).expanduser().resolve())
    train_file = str(Path(settings["train_paths_file"]).expanduser().resolve())
    val_file = str(Path(settings["val_paths_file"]).expanduser().resolve()) if settings.get("val_paths_file") else None

    command = [
        sys.executable, "-u", "train_local.py",
        "--id", execution_id,
        "--run_name", str(settings.get("run_name", "freq")),
        "--save_dir", save_dir,
        "--batch_size", str(settings.get("batch_size", 1)),
        "--workers", str(settings.get("workers", 0)),
        "--model", str(settings.get("model", "ours")),
        "--image_size", str(settings.get("image_size", 512)),
        "--factor", str(settings.get("factor", 0.9)),
        "--patience", str(settings.get("patience", 5)),
        "--paths_file", train_file,
        "--test_paths_file", str(Path(settings["test_paths_file"]).expanduser().resolve()),
        "--n_epochs", str(settings.get("n_epochs", 2)),
        "--lr", str(settings.get("lr", 1e-3)),
        "--device", str(settings.get("device", "cpu")),
        "--repo", dataset["repo"],
        "--commit", dataset["commit"],
        "--name", dataset["name"],
        "--dataset_root", dataset["path"],
        "--workspace", settings["mlflow_workspace"],
        "--experiment", settings["mlflow_experiment"],
    ]
    if val_file:
        command.extend(["--val_paths_file", val_file])
    if settings.get("load_path"):
        command.extend(["--load_path", str(settings["load_path"])])
    if settings.get("n_c_samples") is not None:
        command.extend(["--n_c_samples", str(settings["n_c_samples"])])
    if settings.get("val_n_c_samples") is not None:
        command.extend(["--val_n_c_samples", str(settings["val_n_c_samples"])])

    encoded_settings = base64.b64encode(json.dumps(settings, separators=(",", ":")).encode()).decode()
    env = {
        "CODE_REPO": code["repo"],
        "CODE_COMMIT": code["commit"],
        "PIPELINE_SETTINGS_B64": encoded_settings,
        "PIPELINE_KIND": str(settings.get("pipeline_kind", "new")),
    }
    logger.info("Running local training %s", execution_id)
    run(command, cwd=code["path"], env=env)
    result_path = Path(save_dir) / "pipeline-results" / f"train-{execution_id}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Training did not produce result contract: {result_path}")
    return load_json(result_path)


@task(name="evaluate-model")
def evaluate_local(code: dict[str, str], dataset: dict[str, str], training: dict[str, Any],
                   settings: dict[str, Any]) -> dict[str, Any]:
    logger = get_run_logger()
    execution_id = _execution_id("eval")
    save_dir = str(Path(settings["save_dir"]).expanduser().resolve())
    test_file = str(Path(settings["test_paths_file"]).expanduser().resolve())
    output_dir = Path(save_dir) / "evaluations" / execution_id
    command = [
        sys.executable, "-u", "eval.py",
        "--id", execution_id,
        "--iut_paths_file", test_file,
        "--image_size", str(settings.get("image_size", 512)),
        "--out_dir", str(output_dir),
        "--model", str(settings.get("model", "ours")),
        "--load_path", training["best_checkpoint"],
        "--repo", dataset["repo"],
        "--commit", dataset["commit"],
        "--name", dataset["name"],
        "--dataset_root", dataset["path"],
        "--workspace", settings["mlflow_workspace"],
        "--experiment", settings["mlflow_experiment"],
    ]
    env = {"PARENT_MLFLOW_RUN_ID": training["mlflow_run_id"], "SAVE_DIR": save_dir}
    logger.info("Running local evaluation %s", execution_id)
    run(command, cwd=code["path"], env=env)
    result_path = Path(save_dir) / "pipeline-results" / f"eval-{execution_id}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Evaluation did not produce result contract: {result_path}")
    result = load_json(result_path)
    threshold = float(settings.get("minimum_accuracy", 0.0))
    result["minimum_accuracy"] = threshold
    result["accepted"] = float(result["accuracy"]) >= threshold
    return result


@task(name="register-and-promote-model")
def register_and_promote_model(dataset: dict[str, str], training: dict[str, Any],
                               evaluation: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    model_name = _safe_model_name(str(settings.get("registered_model_name") or
                                      f"{dataset['name']}-{settings.get('model', 'model')}"))
    client = MlflowClient()
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)

    model_version = mlflow.register_model(training["model_uri"], model_name)
    version = str(model_version.version)
    client.set_model_version_tag(model_name, version, "training_run_id", training["mlflow_run_id"])
    client.set_model_version_tag(model_name, version, "evaluation_run_id", evaluation["mlflow_run_id"])
    client.set_model_version_tag(model_name, version, "accuracy", str(evaluation["accuracy"]))
    client.set_model_version_tag(model_name, version, "dataset_commit", dataset["commit"])

    candidate_alias = str(settings.get("candidate_alias", "candidate"))
    promotion_alias = str(settings.get("promotion_alias", "champion"))
    client.set_registered_model_alias(model_name, candidate_alias, version)
    promoted = bool(evaluation["accepted"] and settings.get("promote_on_pass", True))
    if promoted:
        client.set_registered_model_alias(model_name, promotion_alias, version)

    return {
        "registered_model_name": model_name,
        "model_version": version,
        "candidate_alias": candidate_alias,
        "promotion_alias": promotion_alias if promoted else None,
        "promoted": promoted,
        "accuracy": evaluation["accuracy"],
        "minimum_accuracy": evaluation["minimum_accuracy"],
    }


def _download_conditioned_manifests(source_run_id: str, destination: Path,
                                    dataset_root: str) -> tuple[str, str | None]:
    client = MlflowClient()
    run_info = client.get_run(source_run_id)
    source_execution_id = run_info.data.params.get("run_id")
    if not source_execution_id:
        raise RuntimeError("Source MLflow run does not contain the run_id parameter")
    destination.mkdir(parents=True, exist_ok=True)

    def download(kind: str, required: bool) -> str | None:
        filename = f"cond_paths_file_{source_execution_id}_{kind}.txt"
        candidates = [f"datasets/portable/{filename}", f"datasets/{filename}"]
        error: Exception | None = None
        for artifact_path in candidates:
            try:
                downloaded = Path(client.download_artifacts(source_run_id, artifact_path, str(destination / "download")))
                return str(materialize_manifest(downloaded, destination / filename, dataset_root))
            except Exception as exc:
                error = exc
        if required:
            raise RuntimeError(f"Could not download conditioned manifest for {kind}") from error
        return None

    return download("train", True), download("val", False)


@flow(name="upload-manifest-bundle", log_prints=True)
def upload_manifest_bundle_flow(train_paths_file: str, val_paths_file: str, test_paths_file: str,
                                dataset_root: str, mlflow_workspace: str,
                                mlflow_experiment: str = "pipeline-manifests") -> dict[str, str]:
    """Run on the machine that can see the user-prepared manifests and upload portable copies."""
    mlflow.set_workspace(mlflow_workspace)
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=f"manifest-bundle-{_execution_id('manifest')}") as active:
        temp = Path.cwd() / ".manifest-upload" / active.info.run_id
        for source, name in [
            (train_paths_file, "train_datalad.txt"),
            (val_paths_file, "val_datalad.txt"),
            (test_paths_file, "test_datalad.txt"),
        ]:
            portable = write_portable_manifest(source, temp / name, dataset_root)
            mlflow.log_artifact(str(portable), "manifests")
        mlflow.log_params({"dataset_root_at_upload": str(Path(dataset_root).resolve())})
        return {"manifest_source_run_id": active.info.run_id}


def _run_pipeline(code: dict[str, str], dataset: dict[str, str], settings: dict[str, Any],
                  use_conditioned_files: bool = False) -> dict[str, Any]:
    training = train_local(code, dataset, settings, use_conditioned_files=use_conditioned_files)
    evaluation = evaluate_local(code, dataset, training, settings)
    registration = register_and_promote_model(dataset, training, evaluation, settings)
    return {"training": training, "evaluation": evaluation, "registration": registration}


@flow(name="new-training-and-evaluation", log_prints=True)
def new_training_flow(
    code_repo_url: str,
    dataset_repo_url: str,
    settings: dict[str, Any],
    code_commit: str | None = None,
    dataset_commit: str | None = None,
) -> dict[str, Any]:
    """Run a new experiment. Paths and Git credentials are worker infrastructure settings."""
    settings = {"minimum_accuracy": 0.0, "promote_on_pass": True, **settings, "pipeline_kind": "new"}
    code = prepare_code(code_repo_url, code_commit)
    dataset = prepare_dataset(dataset_repo_url, dataset_commit)
    settings = prepare_original_manifests(dataset, settings)
    return _run_pipeline(code, dataset, settings)


@flow(name="reproduce-training-and-evaluation", log_prints=True)
def reproduce_training_flow(
    source_mlflow_run_id: str,
    save_dir: str | None = None,
    settings_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reproduce a run from MLflow-recorded code, data, settings, and conditioned manifests."""
    source = MlflowClient().get_run(source_mlflow_run_id)
    params, tags = dict(source.data.params), dict(source.data.tags)
    if "pipeline.settings_json" not in tags:
        raise RuntimeError("Source run does not contain pipeline.settings_json")
    settings = json.loads(tags["pipeline.settings_json"])
    settings.update(settings_overrides or {})
    if save_dir:
        settings["save_dir"] = save_dir
    settings["pipeline_kind"] = "reproduce"
    settings.setdefault("minimum_accuracy", 0.0)
    settings.setdefault("promote_on_pass", True)

    code = prepare_code(tags["code.repo"], tags["code.commit"])
    dataset = prepare_dataset(params["repo"], params["commit"])
    manifests_dir = Path(settings["save_dir"]).expanduser().resolve() / "reproduction-manifests" / source_mlflow_run_id
    train_manifest, val_manifest = _download_conditioned_manifests(
        source_mlflow_run_id, manifests_dir, dataset["path"]
    )
    settings["train_paths_file"], settings["val_paths_file"] = train_manifest, val_manifest
    try:
        portable_test = _download_artifact(
            source_mlflow_run_id,
            "datasets/portable/test_datalad.txt",
            manifests_dir / "test",
            workspace=settings.get("mlflow_workspace"),
        )
        settings["test_paths_file"] = str(
            materialize_manifest(
                portable_test,
                manifests_dir / "test_datalad.txt",
                dataset["path"],
            )
        )
    except Exception:
        if not settings.get("test_paths_file"):
            raise RuntimeError(
                "Source run has no portable test manifest and no test_paths_file override"
            )
    return _run_pipeline(code, dataset, settings, use_conditioned_files=True)


@flow(name="retrain-and-evaluate", log_prints=True)
def retrain_flow(
    source_mlflow_run_id: str,
    dataset_repo_url: str,
    settings_overrides: dict[str, Any],
    dataset_commit: str | None = None,
) -> dict[str, Any]:
    """Retrain source code with another dataset version and/or changed settings."""
    source = MlflowClient().get_run(source_mlflow_run_id)
    tags = dict(source.data.tags)
    if "pipeline.settings_json" not in tags:
        raise RuntimeError("Source run does not contain pipeline.settings_json")
    settings = json.loads(tags["pipeline.settings_json"])
    settings.update(settings_overrides)
    settings["pipeline_kind"] = "retrain"
    settings.setdefault("minimum_accuracy", 0.0)
    settings.setdefault("promote_on_pass", True)

    code = prepare_code(tags["code.repo"], tags["code.commit"])
    dataset = prepare_dataset(dataset_repo_url, dataset_commit)
    settings = prepare_original_manifests(dataset, settings)
    return _run_pipeline(code, dataset, settings)

