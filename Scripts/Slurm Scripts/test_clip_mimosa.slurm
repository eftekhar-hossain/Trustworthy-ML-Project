#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=pytorch-clip-mimosa-slurm-%J.out
#SBATCH --job-name=clip-mimosa

# Give this process 1 task and 1 GPU, then assign four CPUs per task
# (so 4 cores overall).

# If you want two GPUs:
# #SBATCH --gres=gpu:2
# #SBATCH --cpus-per-task=8
# This example, however, only uses one GPU.

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the Python and CUDA modules
module load anaconda
module load cuda

# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m

nvidia-smi


echo

# Activate the GPU version of PyTorch
conda activate CAP-6638

# Here we are going to run the Bengali Aggression Memes model
# GitHub Repository: https://github.com/eftekhar-hossain/Bengali-Aggression-Memes

# Run PyTorch Training
echo "Training Start:"
time python main.py --dataset mimosa --method mclip --epochs 50 --learning_rate 5e-4
echo

# You're done!
echo "Ending script..."
date
