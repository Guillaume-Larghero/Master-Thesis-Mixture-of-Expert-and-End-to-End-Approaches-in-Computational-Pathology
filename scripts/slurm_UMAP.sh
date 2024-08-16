#!/bin/bash
#SBATCH -c 16
#SBATCH -t 0-XX:00
#SBATCH -p XX
#SBATCH --mem=XXG 
#SBATCH -o logs/UMAP%j.out
#SBATCH -e logs/UMAP%j.err


# === CHANGE THESE ===
source activate #Activate Env HERE

echo "==============================="
echo "configname is UMAP"
echo "==============================="


python UMAPVisualisayion2.py --config-name UMAP

# Example : sbatch slurm_UMAP.sh 
