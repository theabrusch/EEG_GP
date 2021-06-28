#!/bin/sh
#BSUB -J crossval_GP_08p
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 10:00
#BSUB -o logs/Output_08p_%J.out
#BSUB -e logs/Error_08p_%J.err

echo "Runnin script..."

python3 crossvalidation.py 
