#!/bin/bash
#SBATCH --job-name=damnist-padpaf
#SBATCH -q gpu-single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --output=train-%j.out

source activate torch

python run_gan.py \
    --gpu 0 \
    --deterministic --seed 123 \
    --eval-every 1 --save-every 1 --loglevel DEBUG \
    --identifier damnist-padpaf-vae \
    --task damnist-fedvae --simulated-workers 8 \
    --sampling fixed --aggregation mean \
    --comm-rounds 1000 --local-epochs 1 \
    --local-opt adam-mom --global-opt adam-mom \
    --local-lr 0.001 --global-lr 0.01 \
    --batch-size 32 \
    --ssl-reg 0.002 \
    --vae-beta 0.01
