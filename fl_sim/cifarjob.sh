#!/bin/bash
#SBATCH --job-name=PaDomDisPaF
#SBATCH -q gpu-single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=train-%j.log

source activate torch

defaults="--gpu 0 --loglevel DEBUG --deterministic --seed 123 -agg mean --sampling fixed -ep 0.5 --local-opt adam --global-opt adam --local-lr 0.001 --global-lr 0.01 -cr 300 -b 32 --lr-sched exp"
saving="--eval-every 5 --save-every 25"
load=""
#load="--load ../outputs/id=padpaf/task=damnist-fedgan-conditional-partial/lr=0.001_0.01/seed=123/model/model.pth.tar"
#load="--load ../outputs/id=padpaf-celeba-loadoptim-2D5/task=celeba-fedgan/lr=0.001_0.01/seed=123/model/model.pth.tar"

# CIFAR-10/100
python run_gan.py ${defaults} ${saving} --task dacifar100-fedgan-conditional-partial --identifier padpaf-fedbn -b 64 -sw 10

