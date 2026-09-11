#!/bin/bash
#SBATCH --job-name=lab_4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"

source $HOME/deep_learning/env_dl/bin/activate
mkdir -p $HOME/deep_learning/src/lab_4/part_3/output

echo "=== Execution ==="
python $HOME/deep_learning/src/lab_4/part_3/mini_batch_gd.py