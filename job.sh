#!/bin/sh
#BSUB -J crossval_GP
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 10:00
#BSUB -o logs/Output_%J.out
#BSUB -e logs/Error_%J.err

echo "Runnin script..."

python3 crossvalidation.py 
