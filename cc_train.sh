#!/bin/bash
#SBATCH --job-name=mlflow-prefect-test
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2
module load opencv/4.10.0
module load arrow/24.0.0
module load git-annex

source /envs/MLflow/bin/activate

ROOT=$PWD

cd $ROOT
export NCCL_BLOCKING_WAIT=1  # Set this environment variable if you wish to use the NCCL backend for inter-GPU communication

mkdir -p /scratch/${whoami}/$(basename $PWD)

ln -sfn /scratch/${whoami}/$(basename $PWD) $ROOT/scratch
ln -sfn $ROOT /scratch/${whoami}/$(basename $PWD)/code 

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

srun python -u train_torch_test.py \
    --id $SLURM_JOB_ID \
    --save_dir /scratch/${whoami}/$(basename $PWD) \
    --batch_size 1 \
    --model ours \
    --image_size 512 \
    --factor 0.9 \
    --paths_file /scratch/${whoami}/$(basename $PWD)/training/train_datalad.txt --val_paths_file /scratch/${whoami}/$(basename $PWD)/training/val_datalad.txt \
    --n_epochs 2 \
    --repo $DATASET_REPO --commit $DATASET_COMMIT --name $DATASET_NAME \
    --workspace $MLFLOW_WORKSPACE --experiment $MLFLOW_EXPERIMENT

rm -rf core.*

echo "End"