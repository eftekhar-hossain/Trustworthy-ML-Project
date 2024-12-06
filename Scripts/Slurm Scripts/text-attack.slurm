#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=text-attack-transfer-slurm-%J.out
#SBATCH --job-name=attack-text

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


# Activate the GPU version of PyTorch
conda activate Trustworthy-ML-Project


echo "Transfer Attack"

# Run PyTorch Training

echo "Start Attacking: MIMOSA MCLIP"
time python text-attack.py --dataset mimosa --method mclip --text_attack translate
echo

echo "Start Attacking: MIMOSA DORA"
time python text-attack.py --dataset mimosa --method dora --text_attack translate
echo

echo "Start Attacking: BHM MAF"
time python text-attack.py --dataset bhm --method maf --text_attack translate
echo

echo "Start Attacking: BHM DORA"
time python text-attack.py --dataset bhm --method dora --text_attack translate
echo

echo "Start Attacking: BHM MCLIP"
time python text-attack.py --dataset bhm --method mclip --text_attack translate
echo

echo "Start Attacking: BHM DORA"
time python text-attack.py --dataset bhm --method dora --text_attack all
echo

echo "Start Attacking: BHM MCLIP"
time python text-attack.py --dataset bhm --method mclip --text_attack all
echo

echo "Start Attacking: BHM MAF"
time python text-attack.py --dataset bhm --method maf --text_attack all
echo

echo "Start Attacking: MIMOSA DORA"
time python text-attack.py --dataset mimosa --method dora --text_attack all
echo

echo "Start Attacking: MIMOSA MCLIP"
time python text-attack.py --dataset mimosa --method mclip --text_attack all
echo

echo "Start Attacking: MIMOSA MAF"
time python text-attack.py --dataset mimosa --method maf --text_attack all
echo

# You're done!
echo "Ending script..."
date