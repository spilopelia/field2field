#!/bin/bash -l

#SBATCH -p a
#SBATCH --nodelist=a9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G
#SBATCH --output=logs/f2f-%j.log
#SBATCH --time=3-00:00:00
#SBATCH -J "f2f"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
#export TORCH_HOME=/tmp/torch_cache

# enable logging    
#export CUDA_LAUNCH_BLOCKING=1

conda activate d3m
cd $SLURM_SUBMIT_DIR
srun python3 /home/user/ckwan1/ml_project/field2field/trainer.py $@ --gpus 8 --num_nodes 1 --num_workers ${SLURM_CPUS_PER_GPU}