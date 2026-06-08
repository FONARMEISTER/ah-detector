#!/bin/bash
#SBATCH --job-name=iasarantsev_vit
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err

# Активируй своё окружение
source /home3/iasarantsev/bin/activate

echo $VIRTUAL_ENV

pip install -r /home3/iasarantsev/requirements.txt

# Запусти блокнот
python3 -u ./vivit_training.py

