#!/bin/bash
#SBATCH -c 16
#SBATCH -t 1-00:00
#SBATCH -p gpu_yu
#SBATCH --account=yu_ky98_contrib
#SBATCH --mem=38G 
#SBATCH -x compute-gc-17-252
#SBATCH -o logs/milmoeCOMP%j.out
#SBATCH -e logs/milmoeCOMP%j.err
#SBATCH --gres=gpu:1

# For RAM , Do 12G for single expert , 24G for two etc ... Double the time as well
module load gcc/9.2.0
module load cuda/12.1
module load miniconda3

# === CHANGE THESE ===
source activate /home/gul075/.conda/envs/MOE_github

echo "==============================="
nvidia-smi
echo "==============================="

CUDA_VERSION=$(nvidia-smi | awk '/CUDA Version:/ {print $9}')
echo "==============================="
echo "Detected CUDA Version: $CUDA_VERSION"
echo "==============================="

python -c "import torch;print(torch.cuda.is_available())"

echo "configname is $1"

python train_mil_moe_complete.py --config-name $1

# Example : sbatch slurm_mil_moe.sh leukemia_FLT3_mil_NTU2