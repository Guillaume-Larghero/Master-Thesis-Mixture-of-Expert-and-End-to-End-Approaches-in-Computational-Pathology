#!/bin/bash
#SBATCH -c 16
#SBATCH -t 0-02:00
#SBATCH -p short
#SBATCH --mem=16G 
#SBATCH -o logs/UMAP%j.out
#SBATCH -e logs/UMAP%j.err


# === CHANGE THESE ===
source activate /home/gul075/.conda/envs/MOE_github

echo "==============================="
echo "configname is UMAP"
echo "==============================="


python UMAPVisualisayion2.py --config-name UMAP

# Example : sbatch slurm_UMAP.sh 