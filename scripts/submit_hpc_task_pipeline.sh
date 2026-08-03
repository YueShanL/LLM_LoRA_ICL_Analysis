#!/usr/bin/env bash
#SBATCH --job-name=lia-hpc-pipeline
#SBATCH --output=lia-hpc-pipeline-%j.out
#SBATCH --error=lia-hpc-pipeline-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
python -m pip install -r requirements.txt
python scripts/hpc_task_pipeline.py --config "${1:-configs/hpc_task_pipeline.json}"
