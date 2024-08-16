#!/bin/bash
#SBATCH -c 16
#SBATCH -t X-XX:XX
#SBATCH -p XX
#SBATCH --account=XX
#SBATCH --mem=XXG
#SBATCH -x compute-gc-17-252
#SBATCH -o logs/finetunef%j.out
#SBATCH -e logs/finetunef%j.err
#SBATCH --gres=gpu:1

# For RAM , Do 12G for single expert , 24G for two etc ... Double the time as well
module load gcc/9.2.0
module load cuda/12.1
module load miniconda3

# === CHANGE THESE ===
source activate # ACTIVAE ENV HERE

echo "==============================="
nvidia-smi
echo "==============================="

CUDA_VERSION=$(nvidia-smi | awk '/CUDA Version:/ {print $9}')
echo "==============================="
echo "Detected CUDA Version: $CUDA_VERSION"
echo "==============================="


python -c "import torch;print(torch.cuda.is_available())"

echo "config name is $1"

python finetuning_MOE.py --config-name $1 modelpath="$2" modelname="$3" infdatasetpath="$4"

# Example : sbatch slurm_finetuning_MOE.sh Finetune7 /n/scratch/users/g/gul075/checkpoints/Leukemia/AML_APL2_CLIPPED/NTU/40xNORM/clipped 2024-07-18_15-18-58_leukemia_AMLAPL_mil_GIGAUNI_NTU_40xNorm.gigapath_uni_MIL_NTU_CLIPPEDSAVE /home/gul075/MOE_github/MOE/data/leukemia/AML_APL/Inference/Patch40xNorm
