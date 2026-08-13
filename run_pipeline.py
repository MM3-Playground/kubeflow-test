import argparse
import base64
import json
from pathlib import Path

from kfp import compiler, dsl

from pipeline.flows import (
    new_training_pipeline,
    reproduce_training_pipeline,
    retrain_pipeline,
    write_manifest_bundle,
)
from pipeline.helpers import write_portable_manifest


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="mode", required=True)

manifest_parser = sub.add_parser("manifests")
manifest_parser.add_argument("--config", required=True)
manifest_parser.add_argument("--output")

for name in ("new", "reproduce", "retrain"):
    command = sub.add_parser(name)
    command.add_argument("--output")

args = parser.parse_args()
Path("compiled").mkdir(exist_ok=True)

if args.mode == "manifests":
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    temp = Path(".manifest-upload")
    temp.mkdir(exist_ok=True)
    values = {}
    for key, name in [
        ("train_paths_file", "train_datalad.txt"),
        ("val_paths_file", "val_datalad.txt"),
        ("test_paths_file", "test_datalad.txt"),
    ]:
        source = config.get(key)
        if source:
            portable = write_portable_manifest(
                source,
                temp / name,
                config["dataset_root"],
            )
            values[key] = base64.b64encode(portable.read_bytes()).decode()
        else:
            values[key] = ""

    @dsl.pipeline(name="upload-manifest-bundle-local")
    def manifest_pipeline(
        train_b64: str = values["train_paths_file"],
        val_b64: str = values["val_paths_file"],
        test_b64: str = values["test_paths_file"],
    ):
        write_manifest_bundle(
            train_b64=train_b64,
            val_b64=val_b64,
            test_b64=test_b64,
        )

    output = args.output or "compiled/upload-manifest-bundle.yaml"
    compiler.Compiler().compile(manifest_pipeline, output)
else:
    pipeline = {
        "new": new_training_pipeline,
        "reproduce": reproduce_training_pipeline,
        "retrain": retrain_pipeline,
    }[args.mode]
    output = args.output or f"compiled/{args.mode}.yaml"
    compiler.Compiler().compile(pipeline, output)

print(output)
