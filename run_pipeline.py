import argparse
import json
from pathlib import Path
from pipeline.flows import (
    new_training_flow,
    reproduce_training_flow,
    retrain_flow,
    upload_manifest_bundle_flow,
)

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="mode", required=True)
for name in ("new", "reproduce", "retrain", "manifests"):
    command = sub.add_parser(name)
    command.add_argument("--config", required=True)
args = parser.parse_args()
config = json.loads(Path(args.config).read_text(encoding="utf-8"))
if args.mode == "new":
    result = new_training_flow(**config)
elif args.mode == "reproduce":
    result = reproduce_training_flow(**config)
elif args.mode == "retrain":
    result = retrain_flow(**config)
else:
    result = upload_manifest_bundle_flow(**config)
print(json.dumps(result, indent=2, default=str))
