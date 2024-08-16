#!/bin/bash
#SBATCH -c 16
#SBATCH -t X-XX:XX
#SBATCH -p XX
#SBATCH --account=XX
#SBATCH --mem=XXG 
#SBATCH -o logs/milE2EFineTune%j.out
#SBATCH -e logs/milE2EFineTune%j.err
#SBATCH --gres=gpu:1

module load gcc/9.2.0
module load cuda/12.4  # Match this with the version detected by nvidia-smi
module load miniconda3

# Activate the conda environment
source activate # Activate Env HERE

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


# Run your training script
python finetuning_e2e.py --config-name "$1"

# Example : sbatch slurm_finetuning_e2e.sh Finetune_e2e
