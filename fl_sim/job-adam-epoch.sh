#!/bin/bash
#SBATCH --job-name=PaDomDisPaF
#SBATCH -q gpu-single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=train-%j.log

source activate torch

defaults="--gpu 0 --loglevel DEBUG --deterministic --seed 123 -agg mean --sampling fixed --local-opt adamw --global-opt adamw"
saving="--eval-every 5 --save-every 25"
load_dacifar_model="--load ../outputs/id=padpaf-revision-adam-epoch/task=dacifar10-fedgan/lr=0.0005_0.02/seed=123/model/model.pth.tar"

# CIFAR-10/100
python run_gan.py ${defaults} ${saving} --task dacifar10-fedgan --identifier padpaf-revision-test2 -b 64 -sw 10 -ep 1 -cr 3000 --D-iters 5 --local-lr 0.0003 --global-lr 0.03 --ssl-reg 0.5 --eval-every 5 ### ${load_dacifar_model}

