from typing import NamedTuple
import os

from kfp import dsl, kubernetes
from kfp.dsl import (
    Artifact,
    ClassificationMetrics,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
)

RUNTIME_IMAGE = "registry.rcg.sfu.ca/hallo/hallo-data-portal/kubeflow:test"
IMAGE_PULL_SECRET = "regcred"



@dsl.component(base_image="python:3.11-slim")
def write_manifest_bundle(
    train_b64: str,
    val_b64: str,
    test_b64: str,
    manifests: Output[Dataset],
):
    import base64
    import hashlib
    import json
    import zipfile
    from pathlib import Path

    target = Path(manifests.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {}
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in [
            ("train_datalad.txt", train_b64),
            ("val_datalad.txt", val_b64),
            ("test_datalad.txt", test_b64),
        ]:
            if not value:
                continue
            data = base64.b64decode(value)
            archive.writestr(name, data)
            rows = [line for line in data.decode("utf-8").splitlines() if line.strip()]
            metadata[name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "rows": len(rows),
            }

    manifests.metadata["kind"] = "portable-manifest-bundle"
    manifests.metadata["format"] = "zip"
    manifests.metadata["files"] = json.dumps(metadata, sort_keys=True)


@dsl.pipeline(name="upload-manifest-bundle")
def upload_manifest_bundle_pipeline(
    train_b64: str,
    val_b64: str,
    test_b64: str,
):
    write_manifest_bundle(
        train_b64=train_b64,
        val_b64=val_b64,
        test_b64=test_b64,
    )


@dsl.component(base_image=RUNTIME_IMAGE)
def resolve_sources(
    code_repo_url: str,
    dataset_repo_url: str,
    code_commit: str,
    dataset_commit: str,
    code_source: Output[Artifact],
    dataset: Output[Dataset],
) -> NamedTuple(
    "Outputs",
    [
        ("code_commit", str),
        ("dataset_commit", str),
        ("dataset_id", str),
        ("dataset_name", str),
    ],
):
    import json
    import os
    import stat
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    def git_auth_env():
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        token = env.get("GITHUB_TOKEN", "")
        if not token:
            return env, None

        handle = tempfile.NamedTemporaryFile(
            "w", delete=False, prefix="git-askpass-"
        )
        handle.write(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *sername*) printf '%s\\n' \"$GITHUB_USERNAME\" ;;\n"
            "  *)         printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
            "esac\n"
        )
        handle.close()
        os.chmod(
            handle.name,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
        )
        env["GIT_ASKPASS"] = handle.name
        return env, handle.name

    auth_env, askpass_path = git_auth_env()

    def run(command, cwd=None):
        return subprocess.check_output(
            command, cwd=cwd, env=auth_env, text=True
        ).strip()

    code_path = Path("/tmp/code")
    data_path = Path("/tmp/data")

    subprocess.run(
        ["git", "config", "--global", "user.name", "Kubeflow Pipeline"],
        check=True,
    )
    subprocess.run(
        ["git", "config", "--global", "user.email", "kubeflow@localhost"],
        check=True,
    )

    subprocess.run(["git", "clone", code_repo_url, str(code_path)], env=auth_env, check=True)
    resolved_code_commit = code_commit or run(["git", "rev-parse", "HEAD"], code_path)
    subprocess.run(
        ["git", "checkout", "--detach", resolved_code_commit],
        cwd=code_path,
        env=auth_env,
        check=True,
    )
    resolved_code_commit = run(["git", "rev-parse", "HEAD"], code_path)

    subprocess.run(["datalad", "clone", dataset_repo_url, str(data_path)], env=auth_env, check=True)
    resolved_dataset_commit = dataset_commit or run(["git", "rev-parse", "HEAD"], data_path)
    subprocess.run(
        ["git", "checkout", "--detach", resolved_dataset_commit],
        cwd=data_path,
        env=auth_env,
        check=True,
    )
    resolved_dataset_commit = run(["git", "rev-parse", "HEAD"], data_path)
    dataset_id = run(
        [
            "datalad",
            "configuration",
            "-d",
            str(data_path),
            "get",
            "datalad.dataset.id",
        ]
    )
    dataset_name = data_path.name

    code_record = {
        "repo": code_repo_url,
        "commit": resolved_code_commit,
    }
    dataset_record = {
        "repo": dataset_repo_url,
        "commit": resolved_dataset_commit,
        "dataset_id": dataset_id,
        "name": dataset_name,
    }

    Path(code_source.path).parent.mkdir(parents=True, exist_ok=True)
    Path(code_source.path).write_text(json.dumps(code_record, indent=2), encoding="utf-8")
    code_source.metadata["repo"] = code_repo_url
    code_source.metadata["commit"] = resolved_code_commit
    code_source.metadata["kind"] = "git-source"

    Path(dataset.path).parent.mkdir(parents=True, exist_ok=True)
    Path(dataset.path).write_text(json.dumps(dataset_record, indent=2), encoding="utf-8")
    dataset.metadata["repo"] = dataset_repo_url
    dataset.metadata["commit"] = resolved_dataset_commit
    dataset.metadata["dataset_id"] = dataset_id
    dataset.metadata["name"] = dataset_name
    dataset.metadata["versioning"] = "DataLad/git-annex"

    if askpass_path:
        Path(askpass_path).unlink(missing_ok=True)

    return (
        resolved_code_commit,
        resolved_dataset_commit,
        dataset_id,
        dataset_name,
    )


@dsl.component(
    base_image=RUNTIME_IMAGE,
    packages_to_install=["psutil>=5.9,<8"],
)
def train_local(
    code_source: Input[Artifact],
    dataset: Input[Dataset],
    manifest_bundle: Input[Dataset],
    pipeline_kind: str,
    run_name: str,
    seed: int,
    model: str,
    image_size: int,
    batch_size: int,
    workers: int,
    optimizer: str,
    learning_rate: float,
    epochs: int,
    factor: float,
    patience: int,
    early_stopping_patience: int,
    n_c_samples: int,
    val_n_c_samples: int,
    load_model_uri: str,
    trained_model: Output[Model],
    checkpoints: Output[Artifact],
    train_dataset: Output[Dataset],
    val_dataset: Output[Dataset],
    training_metrics: Output[Metrics],
    training_history: Output[Artifact],
    training_metadata: Output[Artifact],
    system_metrics: Output[Artifact],
):
    import csv
    import hashlib
    import json
    import os
    import shutil
    import stat
    import subprocess
    import sys
    import tempfile
    import threading
    import time
    from pathlib import Path

    import psutil

    def git_auth_env():
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        token = env.get("GITHUB_TOKEN", "")
        if not token:
            return env, None

        handle = tempfile.NamedTemporaryFile(
            "w", delete=False, prefix="git-askpass-"
        )
        handle.write(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *sername*) printf '%s\\n' \"$GITHUB_USERNAME\" ;;\n"
            "  *)         printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
            "esac\n"
        )
        handle.close()
        os.chmod(
            handle.name,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
        )
        env["GIT_ASKPASS"] = handle.name
        return env, handle.name

    auth_env, askpass_path = git_auth_env()

    def cmd(command, cwd=None, use_git_auth=False):
        subprocess.run(
            command,
            cwd=cwd,
            env=auth_env if use_git_auth else None,
            check=True,
        )

    code_info = json.loads(Path(code_source.path).read_text(encoding="utf-8"))
    dataset_info = json.loads(Path(dataset.path).read_text(encoding="utf-8"))

    code = Path("/tmp/code")
    data = Path("/tmp/data")
    cmd(["git", "clone", code_info["repo"], str(code)], use_git_auth=True)
    cmd(["git", "checkout", "--detach", code_info["commit"]], code, use_git_auth=True)
    cmd(["datalad", "clone", dataset_info["repo"], str(data)], use_git_auth=True)
    cmd(["git", "checkout", "--detach", dataset_info["commit"]], data, use_git_auth=True)
    cmd(["datalad", "get", "-r", "."], data, use_git_auth=True)
    venv = Path("/tmp/venv")
    cmd([sys.executable, "-m", "venv", str(venv)])
    runtime_python = str(venv / "bin" / "python")
    cmd([runtime_python, "-m", "pip", "install", "-r", str(code / "requirements.txt")])

    import zipfile

    work = Path("/tmp/work")
    bundle_dir = work / "manifest-bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(manifest_bundle.path, "r") as archive:
        archive.extractall(bundle_dir)

    material = work / "manifests"
    material.mkdir(parents=True, exist_ok=True)

    for name in ["train_datalad.txt", "val_datalad.txt", "test_datalad.txt"]:
        src = bundle_dir / name
        if not src.exists():
            continue
        lines = []
        for raw in src.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            path, label = raw.split("\t")
            candidate = Path(path)
            resolved = candidate if candidate.is_absolute() else data / candidate
            lines.append(f"{resolved.resolve()}\t{label}")
        (material / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    execution_id = "train-" + time.strftime("%Y%m%d%H%M%S")
    save_dir = work / "run"

    command = [
        runtime_python,
        "-u",
        "train_local.py",
        "--id",
        execution_id,
        "--run_name",
        run_name,
        "--seed",
        str(seed),
        "--save_dir",
        str(save_dir),
        "--batch_size",
        str(batch_size),
        "--workers",
        str(workers),
        "--model",
        model,
        "--image_size",
        str(image_size),
        "--optim",
        optimizer,
        "--factor",
        str(factor),
        "--patience",
        str(patience),
        "--paths_file",
        str(material / "train_datalad.txt"),
        "--test_paths_file",
        str(material / "test_datalad.txt"),
        "--n_epochs",
        str(epochs),
        "--n_early",
        str(early_stopping_patience),
        "--lr",
        str(learning_rate),
        "--device",
        "cpu",
        "--repo",
        dataset_info["repo"],
        "--commit",
        dataset_info["commit"],
        "--name",
        dataset_info["name"],
        "--dataset_root",
        str(data),
    ]
    if (material / "val_datalad.txt").exists():
        command += ["--val_paths_file", str(material / "val_datalad.txt")]
    if n_c_samples >= 0:
        command += ["--n_c_samples", str(n_c_samples)]
    if val_n_c_samples >= 0:
        command += ["--val_n_c_samples", str(val_n_c_samples)]
    if load_model_uri:
        # This POC accepts a local path only. A previous KFP Model artifact should
        # normally be wired as an Input[Model] instead of using this string.
        command += ["--load_path", load_model_uri]

    samples = []
    stop = threading.Event()

    def sample_system():
        process = psutil.Process()
        while not stop.is_set():
            children = process.children(recursive=True)
            rss = process.memory_info().rss + sum(
                p.memory_info().rss for p in children if p.is_running()
            )
            cpu = process.cpu_percent(interval=None) + sum(
                p.cpu_percent(interval=None) for p in children if p.is_running()
            )
            samples.append(
                {
                    "timestamp": time.time(),
                    "cpu_percent": cpu,
                    "rss_bytes": rss,
                }
            )
            stop.wait(5)

    monitor = threading.Thread(target=sample_system, daemon=True)
    monitor.start()
    try:
        cmd(command, code)
    finally:
        stop.set()
        monitor.join(timeout=10)

    result = json.loads(
        (save_dir / "pipeline-results" / f"train-{execution_id}.json").read_text()
    )

    Path(trained_model.path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result["best_checkpoint"], trained_model.path)
    model_sha256 = hashlib.sha256(Path(trained_model.path).read_bytes()).hexdigest()

    checkpoint_archive = shutil.make_archive(
        "/tmp/checkpoints",
        "gztar",
        root_dir=result["checkpoint_dir"],
    )
    Path(checkpoints.path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_archive, checkpoints.path)

    Path(training_history.path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result["training_history"], training_history.path)

    def manifest_info(path):
        content = Path(path).read_bytes()
        return {
            "sha256": hashlib.sha256(content).hexdigest(),
            "rows": len([x for x in content.decode("utf-8").splitlines() if x.strip()]),
        }

    portable_dir = Path(result["portable_manifest_dir"])
    train_manifest = portable_dir / f"cond_paths_file_{execution_id}_train.txt"
    Path(train_dataset.path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_manifest, train_dataset.path)
    train_info = manifest_info(train_dataset.path)

    train_dataset.metadata["split"] = "train"
    train_dataset.metadata["repo"] = dataset_info["repo"]
    train_dataset.metadata["commit"] = dataset_info["commit"]
    train_dataset.metadata["dataset_id"] = dataset_info["dataset_id"]
    train_dataset.metadata["name"] = dataset_info["name"] + "-train"
    train_dataset.metadata["manifest_sha256"] = train_info["sha256"]
    train_dataset.metadata["samples"] = train_info["rows"]

    val_manifest = portable_dir / f"cond_paths_file_{execution_id}_val.txt"
    Path(val_dataset.path).parent.mkdir(parents=True, exist_ok=True)
    if val_manifest.exists():
        shutil.copy2(val_manifest, val_dataset.path)
        val_info = manifest_info(val_dataset.path)
        val_dataset.metadata["present"] = True
        val_dataset.metadata["manifest_sha256"] = val_info["sha256"]
        val_dataset.metadata["samples"] = val_info["rows"]
    else:
        Path(val_dataset.path).write_text("", encoding="utf-8")
        val_dataset.metadata["present"] = False

    val_dataset.metadata["split"] = "val"
    val_dataset.metadata["repo"] = dataset_info["repo"]
    val_dataset.metadata["commit"] = dataset_info["commit"]
    val_dataset.metadata["dataset_id"] = dataset_info["dataset_id"]
    val_dataset.metadata["name"] = dataset_info["name"] + "-val"

    training_metrics.log_metric("final_train_loss", result["final_train_loss"])
    training_metrics.log_metric("final_learning_rate", result["final_learning_rate"])
    training_metrics.log_metric("epochs_completed", result["epochs_completed"])
    if result["best_val_loss"] is not None:
        training_metrics.log_metric("best_val_loss", result["best_val_loss"])

    system_path = Path(system_metrics.path)
    system_path.parent.mkdir(parents=True, exist_ok=True)
    with system_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "cpu_percent", "rss_bytes"],
        )
        writer.writeheader()
        writer.writerows(samples)

    metadata = {
        "execution_id": execution_id,
        "pipeline_kind": pipeline_kind,
        "code_repo": code_info["repo"],
        "code_commit": code_info["commit"],
        "dataset_repo": dataset_info["repo"],
        "dataset_commit": dataset_info["commit"],
        "dataset_id": dataset_info["dataset_id"],
        "dataset_name": dataset_info["name"],
        "manifest_bundle_uri": manifest_bundle.uri,
        "run_name": run_name,
        "seed": seed,
        "model": model,
        "image_size": image_size,
        "batch_size": batch_size,
        "workers": workers,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "factor": factor,
        "patience": patience,
        "early_stopping_patience": early_stopping_patience,
        "device": "cpu",
        "model_sha256": model_sha256,
    }
    Path(training_metadata.path).parent.mkdir(parents=True, exist_ok=True)
    Path(training_metadata.path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            trained_model.metadata[key] = value
    checkpoints.metadata["kind"] = "all-training-checkpoints"
    training_history.metadata["kind"] = "per-epoch-metric-history"
    system_metrics.metadata["kind"] = "system-metric-history"
    system_metrics.metadata["sampling_interval_seconds"] = 5
    if askpass_path:
        Path(askpass_path).unlink(missing_ok=True)


@dsl.component(
    base_image=RUNTIME_IMAGE,
    packages_to_install=["psutil>=5.9,<8", "scikit-learn>=1.4,<2"],
)
def evaluate_local(
    code_source: Input[Artifact],
    dataset: Input[Dataset],
    manifest_bundle: Input[Dataset],
    trained_model: Input[Model],
    pipeline_kind: str,
    model: str,
    image_size: int,
    test_dataset: Output[Dataset],
    evaluation_metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    evaluation_results: Output[Artifact],
    evaluation_metadata: Output[Artifact],
    system_metrics: Output[Artifact],
) -> NamedTuple(
    "Outputs",
    [("accuracy", float)],
):
    import csv
    import hashlib
    import json
    import os
    import shutil
    import stat
    import subprocess
    import sys
    import tempfile
    import threading
    import time
    from pathlib import Path

    import psutil
    from sklearn.metrics import confusion_matrix

    def git_auth_env():
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        token = env.get("GITHUB_TOKEN", "")
        if not token:
            return env, None

        handle = tempfile.NamedTemporaryFile(
            "w", delete=False, prefix="git-askpass-"
        )
        handle.write(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *sername*) printf '%s\\n' \"$GITHUB_USERNAME\" ;;\n"
            "  *)         printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
            "esac\n"
        )
        handle.close()
        os.chmod(
            handle.name,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
        )
        env["GIT_ASKPASS"] = handle.name
        return env, handle.name

    auth_env, askpass_path = git_auth_env()

    def cmd(command, cwd=None, use_git_auth=False):
        subprocess.run(
            command,
            cwd=cwd,
            env=auth_env if use_git_auth else None,
            check=True,
        )

    code_info = json.loads(Path(code_source.path).read_text(encoding="utf-8"))
    dataset_info = json.loads(Path(dataset.path).read_text(encoding="utf-8"))

    code = Path("/tmp/code")
    data = Path("/tmp/data")
    cmd(["git", "clone", code_info["repo"], str(code)], use_git_auth=True)
    cmd(["git", "checkout", "--detach", code_info["commit"]], code, use_git_auth=True)
    cmd(["datalad", "clone", dataset_info["repo"], str(data)], use_git_auth=True)
    cmd(["git", "checkout", "--detach", dataset_info["commit"]], data, use_git_auth=True)
    cmd(["datalad", "get", "-r", "."], data, use_git_auth=True)
    venv = Path("/tmp/venv")
    cmd([sys.executable, "-m", "venv", str(venv)])
    runtime_python = str(venv / "bin" / "python")
    cmd([runtime_python, "-m", "pip", "install", "-r", str(code / "requirements.txt")])

    import zipfile

    bundle_dir = Path("/tmp/manifest-bundle")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(manifest_bundle.path, "r") as archive:
        archive.extractall(bundle_dir)

    source_test = bundle_dir / "test_datalad.txt"
    test_manifest = Path("/tmp/test_datalad.txt")
    lines = []
    for raw in source_test.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        path, label = raw.split("\t")
        candidate = Path(path)
        resolved = candidate if candidate.is_absolute() else data / candidate
        lines.append(f"{resolved.resolve()}\t{label}")
    test_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = Path("/tmp/eval")
    execution_id = "eval-" + time.strftime("%Y%m%d%H%M%S")
    command = [
        runtime_python,
        "-u",
        "eval.py",
        "--id",
        execution_id,
        "--iut_paths_file",
        str(test_manifest),
        "--image_size",
        str(image_size),
        "--out_dir",
        str(out),
        "--model",
        model,
        "--load_path",
        trained_model.path,
        "--repo",
        dataset_info["repo"],
        "--commit",
        dataset_info["commit"],
        "--name",
        dataset_info["name"],
        "--dataset_root",
        str(data),
    ]

    samples = []
    stop = threading.Event()

    def sample_system():
        process = psutil.Process()
        while not stop.is_set():
            children = process.children(recursive=True)
            rss = process.memory_info().rss + sum(
                p.memory_info().rss for p in children if p.is_running()
            )
            cpu = process.cpu_percent(interval=None) + sum(
                p.cpu_percent(interval=None) for p in children if p.is_running()
            )
            samples.append(
                {
                    "timestamp": time.time(),
                    "cpu_percent": cpu,
                    "rss_bytes": rss,
                }
            )
            stop.wait(5)

    monitor = threading.Thread(target=sample_system, daemon=True)
    monitor.start()
    try:
        cmd(command, code)
    finally:
        stop.set()
        monitor.join(timeout=10)

    result = json.loads((out / "result.json").read_text())
    accuracy = float(result["accuracy"])
    evaluation_metrics.log_metric("accuracy", accuracy)

    y_true = []
    y_pred = []
    pred_csv = out / "pred.csv"
    if pred_csv.exists():
        with pred_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                y_pred.append(int(row["Pred"]))
                y_true.append(int(row["True"]))
    if y_true:
        labels = sorted(set(y_true) | set(y_pred))
        matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        classification_metrics.log_confusion_matrix(
            [str(label) for label in labels],
            matrix,
        )

    portable_test = out / "test_datalad.txt"
    Path(test_dataset.path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(portable_test, test_dataset.path)
    test_bytes = Path(test_dataset.path).read_bytes()
    test_dataset.metadata["split"] = "test"
    test_dataset.metadata["repo"] = dataset_info["repo"]
    test_dataset.metadata["commit"] = dataset_info["commit"]
    test_dataset.metadata["dataset_id"] = dataset_info["dataset_id"]
    test_dataset.metadata["name"] = dataset_info["name"] + "-test"
    test_dataset.metadata["manifest_sha256"] = hashlib.sha256(test_bytes).hexdigest()
    test_dataset.metadata["samples"] = len(
        [x for x in test_bytes.decode("utf-8").splitlines() if x.strip()]
    )

    archive = shutil.make_archive("/tmp/evaluation-results", "gztar", root_dir=out)
    Path(evaluation_results.path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, evaluation_results.path)
    evaluation_results.metadata["kind"] = "evaluation-results"
    evaluation_results.metadata["contains"] = "pred.csv, confusion matrix PNG, result JSON, portable test manifest"

    system_path = Path(system_metrics.path)
    system_path.parent.mkdir(parents=True, exist_ok=True)
    with system_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "cpu_percent", "rss_bytes"],
        )
        writer.writeheader()
        writer.writerows(samples)
    system_metrics.metadata["kind"] = "system-metric-history"
    system_metrics.metadata["sampling_interval_seconds"] = 5

    metadata = {
        "execution_id": execution_id,
        "pipeline_kind": pipeline_kind,
        "code_repo": code_info["repo"],
        "code_commit": code_info["commit"],
        "dataset_repo": dataset_info["repo"],
        "dataset_commit": dataset_info["commit"],
        "dataset_id": dataset_info["dataset_id"],
        "dataset_name": dataset_info["name"],
        "model_uri": trained_model.uri,
        "model": model,
        "image_size": image_size,
        "accuracy": accuracy,
    }
    Path(evaluation_metadata.path).parent.mkdir(parents=True, exist_ok=True)
    Path(evaluation_metadata.path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if askpass_path:
        Path(askpass_path).unlink(missing_ok=True)
    return (accuracy,)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["model-registry==0.3.12"],
)
def register_model(
    code_source: Input[Artifact],
    dataset: Input[Dataset],
    trained_model: Input[Model],
    training_metadata: Input[Artifact],
    evaluation_metadata: Input[Artifact],
    accuracy: float,
    minimum_accuracy: float,
    registered_model_name: str,
    promote_on_pass: bool,
    registry_address: str,
    registry_port: int,
) -> str:
    import hashlib
    import json
    import re
    import time
    from pathlib import Path

    from model_registry import ModelRegistry

    code_info = json.loads(Path(code_source.path).read_text(encoding="utf-8"))
    dataset_info = json.loads(Path(dataset.path).read_text(encoding="utf-8"))
    train_info = json.loads(Path(training_metadata.path).read_text(encoding="utf-8"))
    eval_info = json.loads(Path(evaluation_metadata.path).read_text(encoding="utf-8"))

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", registered_model_name).strip("-.")
    if not safe_name:
        safe_name = dataset_info["name"] + "-model"

    accepted = float(accuracy) >= float(minimum_accuracy)
    promoted = bool(accepted and promote_on_pass)
    version = "v" + time.strftime("%Y%m%d%H%M%S")
    sha256 = hashlib.sha256(Path(trained_model.path).read_bytes()).hexdigest()

    registry = ModelRegistry(
        server_address=registry_address,
        port=registry_port,
        author="kubeflow-pipeline",
        is_secure=False,
    )
    registry.register_model(
        safe_name,
        trained_model.uri,
        model_format_name="pytorch",
        model_format_version="2",
        version=version,
        metadata={
            "accuracy": float(accuracy),
            "minimum_accuracy": float(minimum_accuracy),
            "accepted": accepted,
            "promoted": promoted,
            "sha256": sha256,
            "pipeline_kind": train_info["pipeline_kind"],
            "code_repo": code_info["repo"],
            "code_commit": code_info["commit"],
            "dataset_repo": dataset_info["repo"],
            "dataset_commit": dataset_info["commit"],
            "dataset_id": dataset_info["dataset_id"],
            "dataset_name": dataset_info["name"],
            "run_name": train_info["run_name"],
            "model": train_info["model"],
            "image_size": train_info["image_size"],
            "batch_size": train_info["batch_size"],
            "optimizer": train_info["optimizer"],
            "learning_rate": train_info["learning_rate"],
            "epochs": train_info["epochs"],
            "training_execution_id": train_info["execution_id"],
            "evaluation_execution_id": eval_info["execution_id"],
        },
    )
    return version



def _configure_runtime_task(task):
    kubernetes.use_secret_as_env(
        task,
        secret_name="github-credentials",
        secret_key_to_env={
            "username": "GITHUB_USERNAME",
            "token": "GITHUB_TOKEN",
        },
    )
    kubernetes.set_image_pull_secrets(task, [IMAGE_PULL_SECRET])
    return task

def _workflow(
    *,
    pipeline_kind: str,
    code_repo_url: str,
    dataset_repo_url: str,
    manifest_bundle_uri: str,
    code_commit: str,
    dataset_commit: str,
    run_name: str,
    seed: int,
    model: str,
    image_size: int,
    batch_size: int,
    workers: int,
    optimizer: str,
    learning_rate: float,
    epochs: int,
    factor: float,
    patience: int,
    early_stopping_patience: int,
    n_c_samples: int,
    val_n_c_samples: int,
    load_model_uri: str,
    minimum_accuracy: float,
    registered_model_name: str,
    promote_on_pass: bool,
    registry_address: str,
    registry_port: int,
):
    sources = resolve_sources(
        code_repo_url=code_repo_url,
        dataset_repo_url=dataset_repo_url,
        code_commit=code_commit,
        dataset_commit=dataset_commit,
    )
    _configure_runtime_task(sources)

    manifests = dsl.importer(
        artifact_uri=manifest_bundle_uri,
        artifact_class=Dataset,
        reimport=False,
    )

    train = train_local(
        code_source=sources.outputs["code_source"],
        dataset=sources.outputs["dataset"],
        manifest_bundle=manifests.output,
        pipeline_kind=pipeline_kind,
        run_name=run_name,
        seed=seed,
        model=model,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        factor=factor,
        patience=patience,
        early_stopping_patience=early_stopping_patience,
        n_c_samples=n_c_samples,
        val_n_c_samples=val_n_c_samples,
        load_model_uri=load_model_uri,
    )
    _configure_runtime_task(train)

    evaluate = evaluate_local(
        code_source=sources.outputs["code_source"],
        dataset=sources.outputs["dataset"],
        manifest_bundle=manifests.output,
        trained_model=train.outputs["trained_model"],
        pipeline_kind=pipeline_kind,
        model=model,
        image_size=image_size,
    )
    _configure_runtime_task(evaluate)

    register_model(
        code_source=sources.outputs["code_source"],
        dataset=sources.outputs["dataset"],
        trained_model=train.outputs["trained_model"],
        training_metadata=train.outputs["training_metadata"],
        evaluation_metadata=evaluate.outputs["evaluation_metadata"],
        accuracy=evaluate.outputs["accuracy"],
        minimum_accuracy=minimum_accuracy,
        registered_model_name=registered_model_name,
        promote_on_pass=promote_on_pass,
        registry_address=registry_address,
        registry_port=registry_port,
    )


@dsl.pipeline(name="new-training-and-evaluation")
def new_training_pipeline(
    code_repo_url: str,
    dataset_repo_url: str,
    manifest_bundle_uri: str,
    code_commit: str = "",
    dataset_commit: str = "",
    run_name: str = "cpu-demo",
    seed: int = 3721,
    model: str = "ours",
    image_size: int = 512,
    batch_size: int = 1,
    workers: int = 0,
    optimizer: str = "adamw",
    learning_rate: float = 0.001,
    epochs: int = 2,
    factor: float = 0.9,
    patience: int = 5,
    early_stopping_patience: int = 10,
    n_c_samples: int = -1,
    val_n_c_samples: int = -1,
    load_model_uri: str = "",
    minimum_accuracy: float = 0.0,
    registered_model_name: str = "anime-attributor",
    promote_on_pass: bool = True,
    registry_address: str = "http://model-registry-service.kubeflow.svc.cluster.local",
    registry_port: int = 8080,
):
    _workflow(
        pipeline_kind="new",
        code_repo_url=code_repo_url,
        dataset_repo_url=dataset_repo_url,
        manifest_bundle_uri=manifest_bundle_uri,
        code_commit=code_commit,
        dataset_commit=dataset_commit,
        run_name=run_name,
        seed=seed,
        model=model,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        factor=factor,
        patience=patience,
        early_stopping_patience=early_stopping_patience,
        n_c_samples=n_c_samples,
        val_n_c_samples=val_n_c_samples,
        load_model_uri=load_model_uri,
        minimum_accuracy=minimum_accuracy,
        registered_model_name=registered_model_name,
        promote_on_pass=promote_on_pass,
        registry_address=registry_address,
        registry_port=registry_port,
    )


@dsl.pipeline(name="reproduce-training-and-evaluation")
def reproduce_training_pipeline(
    code_repo_url: str,
    dataset_repo_url: str,
    manifest_bundle_uri: str,
    code_commit: str,
    dataset_commit: str,
    run_name: str = "reproduce-cpu",
    seed: int = 3721,
    model: str = "ours",
    image_size: int = 512,
    batch_size: int = 1,
    workers: int = 0,
    optimizer: str = "adamw",
    learning_rate: float = 0.001,
    epochs: int = 2,
    factor: float = 0.9,
    patience: int = 5,
    early_stopping_patience: int = 10,
    n_c_samples: int = -1,
    val_n_c_samples: int = -1,
    load_model_uri: str = "",
    minimum_accuracy: float = 0.0,
    registered_model_name: str = "anime-attributor-reproduction",
    promote_on_pass: bool = False,
    registry_address: str = "http://model-registry-service.kubeflow.svc.cluster.local",
    registry_port: int = 8080,
):
    _workflow(
        pipeline_kind="reproduce",
        code_repo_url=code_repo_url,
        dataset_repo_url=dataset_repo_url,
        manifest_bundle_uri=manifest_bundle_uri,
        code_commit=code_commit,
        dataset_commit=dataset_commit,
        run_name=run_name,
        seed=seed,
        model=model,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        factor=factor,
        patience=patience,
        early_stopping_patience=early_stopping_patience,
        n_c_samples=n_c_samples,
        val_n_c_samples=val_n_c_samples,
        load_model_uri=load_model_uri,
        minimum_accuracy=minimum_accuracy,
        registered_model_name=registered_model_name,
        promote_on_pass=promote_on_pass,
        registry_address=registry_address,
        registry_port=registry_port,
    )


@dsl.pipeline(name="retrain-and-evaluate")
def retrain_pipeline(
    code_repo_url: str,
    dataset_repo_url: str,
    manifest_bundle_uri: str,
    code_commit: str,
    dataset_commit: str = "",
    run_name: str = "retrain-cpu",
    seed: int = 3721,
    model: str = "ours",
    image_size: int = 512,
    batch_size: int = 1,
    workers: int = 0,
    optimizer: str = "adamw",
    learning_rate: float = 0.0001,
    epochs: int = 3,
    factor: float = 0.9,
    patience: int = 5,
    early_stopping_patience: int = 10,
    n_c_samples: int = -1,
    val_n_c_samples: int = -1,
    load_model_uri: str = "",
    minimum_accuracy: float = 0.0,
    registered_model_name: str = "anime-attributor",
    promote_on_pass: bool = True,
    registry_address: str = "http://model-registry-service.kubeflow.svc.cluster.local",
    registry_port: int = 8080,
):
    _workflow(
        pipeline_kind="retrain",
        code_repo_url=code_repo_url,
        dataset_repo_url=dataset_repo_url,
        manifest_bundle_uri=manifest_bundle_uri,
        code_commit=code_commit,
        dataset_commit=dataset_commit,
        run_name=run_name,
        seed=seed,
        model=model,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        factor=factor,
        patience=patience,
        early_stopping_patience=early_stopping_patience,
        n_c_samples=n_c_samples,
        val_n_c_samples=val_n_c_samples,
        load_model_uri=load_model_uri,
        minimum_accuracy=minimum_accuracy,
        registered_model_name=registered_model_name,
        promote_on_pass=promote_on_pass,
        registry_address=registry_address,
        registry_port=registry_port,
    )
