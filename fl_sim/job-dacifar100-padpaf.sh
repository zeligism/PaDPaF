#!/bin/bash
#SBATCH --job-name=dacifar100-padpaf
#SBATCH -q gpu-single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --output=train-%j.out

source activate torch

python run_gan.py \
    --load "../outputs/id=dacifar100-padpaf/task=dacifar100-fedgan/lr=0.0003_0.003/seed=123/model/model.pth.tar" \
    --gpu 0 \
    --deterministic --seed 123 \
    --eval-every 5 --save-every 25 --loglevel DEBUG \
    --identifier dacifar100-padpaf \
    --task dacifar100-fedgan --simulated-workers 10 \
    --sampling fixed --aggregation mean \
    --comm-rounds 1000 --local-epochs 1 \
    --local-opt adam --global-opt adam \
    --local-lr 0.0003 --global-lr 0.003 \
    --batch-size 128 \
    --D-iters 5 \
    --ssl-reg 0.1

