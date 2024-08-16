#!/bin/bash
#SBATCH -c 8
#SBATCH -t X-XX:XX
#SBATCH -p XX
#SBATCH --account=XX
#SBATCH --mem=XXG
#SBATCH -o logs/normalization%j.out
#SBATCH -e logs/normalization%j.err
#SBATCH --gres=gpu:1

module load gcc/9.2.0
module load cuda/12.1
module load miniconda3

# === CHANGE THESE ===
source activate # ADD ENV HERE

echo "==============================="
nvidia-smi
echo "==============================="

CUDA_VERSION=$(nvidia-smi | awk '/CUDA Version:/ {print $9}')
echo "==============================="
echo "Detected CUDA Version: $CUDA_VERSION"
echo "==============================="

python -c "import torch;print(torch.cuda.is_available())"

echo "patch folder: $1"
echo "norm patch folder: $2"

python normalize_patch.py \
    --patch_folder $1 \
    --norm_patch_folder $2 \
    --reference_image $3 

#example  : sbatch slurm_norm.sh /n/data2/hms/dbmi/kyu/lab/gul075/NGS_single_cell_patch /n/data2/hms/dbmi/kyu/lab/gul075/NGS_single_cell_patch_Norm /n/data2/hms/dbmi/kyu/lab/gul075/Ref_Mackenko_SC.png
