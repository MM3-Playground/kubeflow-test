from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping


def run(cmd: list[str], cwd: str | None = None, env: Mapping[str, str] | None = None) -> str:
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})
    process = subprocess.run(
        cmd,
        cwd=cwd,
        env=merged_env,
        check=False,
        text=True,
        capture_output=True,
    )
    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="")
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )
    return process.stdout.strip()


@contextmanager
def git_https_auth_env(username: str | None, token: str | None) -> Iterator[dict[str, str]]:
    """Provide HTTPS Git credentials without putting the token in a URL or command line."""
    if not token:
        yield {"GIT_TERMINAL_PROMPT": "0"}
        return

    askpass_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", prefix="prefect-git-askpass-", delete=False) as script:
            script.write(
                "#!/bin/sh\n"
                "case \"$1\" in\n"
                "  *sername*) printf '%s\\n' \"$PREFECT_GIT_USERNAME\" ;;\n"
                "  *)         printf '%s\\n' \"$PREFECT_GIT_TOKEN\" ;;\n"
                "esac\n"
            )
            askpass_path = script.name

        os.chmod(askpass_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        yield {
            "GIT_ASKPASS": askpass_path,
            "GIT_TERMINAL_PROMPT": "0",
            "PREFECT_GIT_USERNAME": username or "x-access-token",
            "PREFECT_GIT_TOKEN": token,
        }
    finally:
        if askpass_path:
            Path(askpass_path).unlink(missing_ok=True)


def git_clone_or_update(
    repo: str,
    dest: str,
    commit: str | None = None,
    *,
    username: str | None = None,
    token: str | None = None,
) -> dict[str, str]:
    path = Path(dest).expanduser()
    with git_https_auth_env(username, token) as auth_env:
        if not path.exists():
            run(["git", "clone", repo, str(path)], env=auth_env)
        elif not (path / ".git").exists():
            raise RuntimeError(f"Code destination exists but is not a Git repository: {path}")

        run(["git", "fetch", "--all", "--tags", "--prune"], str(path), auth_env)
        if commit:
            run(["git", "checkout", "--detach", commit], str(path), auth_env)
        else:
            branch_process = subprocess.run(
                ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
                cwd=str(path),
                env={**os.environ, **auth_env},
                text=True,
                capture_output=True,
                check=False,
            )
            branch = branch_process.stdout.strip()
            if branch_process.returncode == 0 and branch:
                run(["git", "pull", "--ff-only"], str(path), auth_env)

        resolved = run(["git", "rev-parse", "HEAD"], str(path), auth_env)
        origin = run(["git", "remote", "get-url", "origin"], str(path), auth_env)

    return {
        "path": str(path.resolve()),
        "repo": origin,
        "commit": resolved,
        "name": Path(origin.removesuffix(".git")).name,
    }


def datalad_clone_or_update(
    repo: str,
    path: str | Path,
    commit: str | None = None,
    *,
    username: str | None = None,
    token: str | None = None,
) -> dict[str, str]:
    dataset_path = Path(path).expanduser().resolve()

    with git_https_auth_env(username, token) as auth_env:
        if not dataset_path.exists():
            dataset_path.parent.mkdir(parents=True, exist_ok=True)

            run(["datalad", "clone", repo, str(dataset_path)], env=auth_env)
        elif not (dataset_path / ".git").exists():
            raise RuntimeError(
                f"Dataset path already exists but is not a Git/DataLad "
                f"repository: {dataset_path}"
            )

        if commit:
            run(["git", "checkout", "--detach", commit], cwd=str(dataset_path), env=auth_env)

        # Materialize all annexed content recursively.
        run(["datalad", "get", "-r", "."], cwd=str(dataset_path), env=auth_env)

        resolved_commit = run(["git", "rev-parse", "HEAD"], cwd=str(dataset_path), env=auth_env)
        dataset_id = run(["datalad", "configuration", "get", "-d", str(dataset_path), "datalad.dataset.id"], env=auth_env)

        remote_url = run(["git", "remote", "get-url", "origin"], cwd=str(dataset_path), env=auth_env)

    return {
        "repo": remote_url,
        "path": str(dataset_path),
        "commit": resolved_commit,
        "name": dataset_path.name,
        "dataset_id": dataset_id,
    }


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_manifest(path: str | Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.rstrip().split("\t")
        if len(parts) != 2:
            raise ValueError(f"Invalid manifest line {line_number} in {path}: expected PATH<TAB>LABEL")
        rows.append((parts[0], int(parts[1])))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def write_portable_manifest(
    source: str | Path,
    destination: str | Path,
    dataset_root: str | Path,
) -> Path:
    """Store paths relative to the dataset root when possible."""
    root = Path(dataset_root).expanduser().resolve()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for raw_path, label in read_manifest(source):
            candidate = Path(raw_path).expanduser()
            if candidate.is_absolute():
                try:
                    rendered = candidate.resolve().relative_to(root).as_posix()
                except ValueError:
                    rendered = str(candidate)
            else:
                rendered = candidate.as_posix()
            handle.write(f"{rendered}\t{label}\n")
    return target.resolve()


def materialize_manifest(
    source: str | Path,
    destination: str | Path,
    dataset_root: str | Path,
) -> Path:
    """Convert portable/relative paths into absolute paths for the current dataset checkout."""
    root = Path(dataset_root).expanduser().resolve()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for raw_path, label in read_manifest(source):
            candidate = Path(raw_path).expanduser()
            resolved = candidate if candidate.is_absolute() else root / candidate
            handle.write(f"{resolved.resolve()}\t{label}\n")
    return target.resolve()
