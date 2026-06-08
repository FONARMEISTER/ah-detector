#!/bin/bash
#SBATCH --job-name=iasarantsev_vit
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err

source /home3/iasarantsev/bin/activate

pip install -r /home3/iasarantsev/requirements.txt -q

python3 -u ./text_training.py roberta

