#!/bin/bash
# ==================================================================
# Usage:
#   sbatch slurm_tiles.sh <slide_folder> <patch_folder> [<n_parts> <part>]
#
# Notes:
#   <n_parts> <part> is optional. If you leave it out, it will be 0 and 1.
#   Adjust memory according to your keep_random_n and magnifications
#   arguments.
# ==================================================================
#SBATCH -c 8
#SBATCH -t X-XX:XX
#SBATCH -p XX
#SBATCH --account=XX
#SBATCH --mem=XXG
#SBATCH -o logs/tile_extraction%j.out
#SBATCH -e logs/tile_extraction%j.err

module load gcc/9.2.0
module load cuda/12.1
module load miniconda3/23.1.0

# === CHANGE THIS ===
source activate # ADD ENVIRO HERE

which python3
python3 --version

n_parts=${3:-0}
part=${4:-1}

python create_MOE_tiles.py \
    --slide_folder $1 \
    --patch_folder $2 \
    --output_size 224 \
    --tissue_threshold 5 \
    --magnifications 40 20 10\
    --keep_random_n 2000 \
    --n_workers 8 \
    --n_parts $n_parts \
    --part $part


# ==================================================================
# Notes:
#   I typically run this script using a for loop. If there are N slides
#   I want to tile, I usually submit N / 2 jobs that run in parallel.
#   This works as we only request very limited resources.
# ==================================================================
#
# Example:
#
# success_file="successful_parts_ebrains_40x.txt"
# touch $success_file
# n_parts=1500
# for i in $(seq 0 $((n_parts - 1))); do
#     # Check if the current part has already been successfully executed
#     if grep -qx "$i" "$success_file"; then
#         echo "Part $i already successfully executed."
#     else
#         sbatch slurm_run_patch_extraction.sh $n_parts $i
#         exit_code=$?
#         if [ $exit_code -eq 0 ]; then
#             echo "$i" >> "$success_file"
#         else
#             echo "Failed to execute part $i"
#         fi
#         sleep 0.1
#     fi
# done



