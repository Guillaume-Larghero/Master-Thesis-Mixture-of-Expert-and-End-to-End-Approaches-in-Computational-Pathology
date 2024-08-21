#!/bin/bash
# ==================================================================
# Usage:
#   sbatch slurm_features.sh <path_to_patches> <path_to_features>
#
# Notes:
#   I experimented with batch sizes, and 2048 to 3072 seem to work.
#   You need to first create a conda environment and a huggingface
#   token (for UNI) to run this script. Add the token as hf_token
#   argument.
# ==================================================================
#SBATCH -c 8
#SBATCH -t X-XX:XX
#SBATCH -p XX
#SBATCH --account=XX
#SBATCH --mem=XXG
#SBATCH -o logs/feature_extraction%j.out
#SBATCH -e logs/feature_extraction%j.err
#SBATCH --gres=gpu:1

module load gcc/9.2.0
module load cuda/12.1
module load miniconda3

# === CHANGE THESE ===
source activate # ADD ENVIRONMENT HERE
hf_token="" # <- add your huggingface token here to use the uni model (https://huggingface.co/settings/tokens)

n_parts=${3:-1}
part=${4:-0}

echo "==============================="
nvidia-smi
echo "==============================="

CUDA_VERSION=$(nvidia-smi | awk '/CUDA Version:/ {print $9}')
echo "==============================="
echo "Detected CUDA Version: $CUDA_VERSION"
echo "==============================="

python -c "import torch;print(torch.cuda.is_available())"

export CHIEF_PTH="/YOURPATHTOFILE/chief.onnx"
export CTRANS_PTH="/YOURPATHTOFILE/ctranspath.onnx"


python create_features.py \
    --patch_folder $1 \
    --feat_folder $2 \
    --device cuda \
    --models ctrans,lunit,resnet50,uni,swin224,phikon,chief,plip,gigapath,cigar \
    --hf_token $hf_token \
    --batch_size 2048 \
    --n_parts $n_parts \
    --part $part

# Example : sbatch slurm_features.sh /n/data2/hms/dbmi/kyu/lab/gul075/NTU10_single_cell_patch /n/data2/hms/dbmi/kyu/lab/gul075/NTU10_single_cell_Features
