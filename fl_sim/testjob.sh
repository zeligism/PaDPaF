#!/bin/bash
#SBATCH --job-name=damntest
#SBATCH -q gpu-single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --output=test-%j.log

source activate torch

defaults="--gpu 0 --loglevel DEBUG --deterministic -agg mean --seed 123 --sampling fixed -ep 1 --local-opt adam --global-opt adam --local-lr 0.001 --global-lr 0.01 --lr-sched exp -cr 300 -b 32"
saving="--eval-every 1 --save-every 25"

python run_gan.py ${defaults} ${saving} --task damnist-fedgan-conditional-unseen --identifier run-unseen-normal -sw 4
