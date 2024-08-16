#!/bin/bash
#SBATCH -c 16
#SBATCH -t 4-00:00
#SBATCH -p gpu_yu
#SBATCH --account=yu_ky98_contrib
#SBATCH --mem=48G 
#SBATCH -o logs/milE2EGPU%j.out
#SBATCH -e logs/milE2EGPU%j.err
#SBATCH --gres=gpu:1

module load gcc/9.2.0
module load cuda/12.4  # Match this with the version detected by nvidia-smi
module load miniconda3

# Activate the conda environment
source activate /home/gul075/.conda/envs/MOE_github

# Show the GPU details
echo "==============================="
nvidia-smi
echo "==============================="

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | awk '/CUDA Version:/ {print $9}')
echo "==============================="
echo "Detected CUDA Version: $CUDA_VERSION"
echo "==============================="

# Check if CUDA is available
python -c "import torch;print('CUDA Available:', torch.cuda.is_available())"

# Print cuDNN version
python -c "import torch; print('cuDNN Version:', torch.backends.cudnn.version())"

echo "==============================="
echo "Job is running on node: $SLURMD_NODENAME"
echo "==============================="

echo "configname is $1"

# Print the content of the config file
echo "==============================="
echo "Content of the config file:"
cat "$1"  # Ensure the path is correct and the file exists
echo "==============================="

# Run your training script
python train_mil_e2e.py --config-name "$1"

# Example : sbatch slurm_mil_e2e_GPU.sh E2E_GPU_NTU_NoAug_FLT3
