#!/bin/bash
#SBATCH -c 4
#SBATCH -t X-XX:XX
#SBATCH -p XX 
#SBATCH --mem=XG
#SBATCH -o logsSC/SC%j.out
#SBATCH -e logsSC/SC%j.err


source activate # Activate Env Here


INPUT_DIR=$1
OUTPUT_DIR=$2

echo "Input dir is $1"
echo "Output dir is $2"

python create_tiles_SC.py $INPUT_DIR $OUTPUT_DIR

# example : sbatch slurm_tiles_SC.sh /n/data2/hms/dbmi/kyu/lab/datasets/hematology/BWH_AML/BD2021-0032_Yu /n/data2/hms/dbmi/kyu/lab/gul075/Cytology_Tile_BWH_SC_100x
