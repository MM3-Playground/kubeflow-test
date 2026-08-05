#!/bin/bash
#SBATCH --job-name=mlflow-prefect-eval
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2
module load opencv/4.10.0
module load arrow/24.0.0
module load git-annex

source /envs/MLflow/bin/activate

# For DataLad
DATALAD_DATASET_PATH=

DATASET_REPO=$(git -C "$DATALAD_DATASET_PATH" remote get-url origin)
DATASET_COMMIT=$(git -C "$DATALAD_DATASET_PATH" rev-parse HEAD)
DATASET_NAME=$(basename -s .git "$DATASET_REPO")

# For MLflow
export MLFLOW_TRACKING_URI=
export MLFLOW_TRACKING_USERNAME=
export MLFLOW_TRACKING_PASSWORD=

MLFLOW_WORKSPACE=
MLFLOW_EXPERIMENT=

# For Prefect
export PREFECT_API_URL=
export PREFECT_API_AUTH_STRING=

CHECKPOINT=
python -u eval.py --id $SLURM_JOB_ID --iut_paths_file /scratch/${whoami}/$(basename $PWD)/training/test_datalad.txt --load_path $CHECKPOINT --image_size 512 --out_dir ./eval_{SLURM_JOB_ID}/ --model ours \
    --repo $DATASET_REPO --commit $DATASET_COMMIT --name $DATASET_NAME \
    --workspace $MLFLOW_WORKSPACE --experiment $MLFLOW_EXPERIMENT

rm -rf core.*

echo "End"