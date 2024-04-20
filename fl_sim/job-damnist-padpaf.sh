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
    --eval-every 5 --save-every 1 --loglevel DEBUG \
    --identifier damnist-padpaf \
    --task damnist-fedgan --simulated-workers 8 \
    --sampling fixed --aggregation mean \
    --comm-rounds 1000 --local-epochs 1 \
    --local-opt adam --global-opt adam \
    --local-lr 0.001 --global-lr 0.01 \
    --batch-size 32 \
    --D-iters 3 \
    --ssl-reg 0.1

# Tasks ({...} = mandatory choice, [...] = optional):
#   - mixture{1, 2, 3}
#   - femnist-gan[-conditional]
#   - damnist-fedgan[-conditional][-partial][-unseen]
#   - celeba-fedgan[-conditional]
#   - dacifar10-fedgan[-conditional][-partial]
#   - dacifar100-fedgan[-conditional][-partial]