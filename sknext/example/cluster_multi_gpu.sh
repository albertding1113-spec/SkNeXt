#!/bin/bash
#SBATCH -J mito_infer
#SBATCH -p rtxp6000bw,rtxp6000bws
#SBATCH -o /HWFS/zhaohu_lab/dingjiayi/task/output/261029_sknext_infer_multi_gpu.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH -t 30-00:00

set -euo pipefail

# Activate the environment containing the UPDATED SkNeXt checkout before sbatch,
# or put your cluster's conda/module activation commands here.
# 128 CPU cores total = 32 per GPU. Reduce this if the node/partition cannot supply it.
# Keep CUDA_VISIBLE_DEVICES as assigned by Slurm. Do not launch four sknext tasks:
# SkNeXt spawns its own GPU workers and coordinates a single set of output stores.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=none --gpu-bind=none \
    sknext \
    --config /HWFS/zhaohu_lab/dingjiayi/data/261020_SkNeXt_dataset/config/1_globular_infer.yaml \
    --run_id 1 \
    --gpu all
