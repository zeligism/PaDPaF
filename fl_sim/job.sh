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

damnist_model="--load ../outputs/id=padpaf/task=damnist-fedgan-conditional-partial/lr=0.001_0.01/seed=123/model/model.pth.tar"
celeba_model="--load ../outputs/id=padpaf-celeba-loadoptim-2D5/task=celeba-fedgan/lr=0.001_0.01/seed=123/model/model.pth.tar"
dacifar_model=""
load="${damnist_model}"

# MNIST
#python run_gan.py ${defaults} ${saving} --task damnist-fedgan --identifier padpaf-copy3 -sw 8
#python run_gan.py ${defaults} ${saving} --task damnist-fedgan-conditional-partial --identifier padpaf -sw 8
#python run_gan.py ${defaults} ${saving} --task damnist-fedgan-unseen-conditional-partial --identifier padpaf -sw 5 -cr 100 --eval-every 1 ${load}
python run_gan.py ${defaults} ${saving} --task damnist-fedgan-unseen-conditional-partial --identifier padpaf-fedbn-nodiv4 -sw 5 -cr 26 --eval-every 1 ${load}

# CelebA
#celeba_setting="-sw 10 --sampling roundrobin -b 256 -ep 1 --eval-every 1 --save-every 10"
#python run_gan.py ${defaults} ${saving} --task celeba-fedgan-conditional --identifier padpaf-celeba ${celeba_setting}
#python run_gan.py ${defaults} ${saving} --task celeba-fedgan --identifier padpaf-celeba-loadoptim-2D5-cont ${celeba_setting} ${load} --global-lr 0.01 --local-lr 0.001 -ep 2 --D-iters 5

# CIFAR-10/100
#python run_gan.py ${defaults} ${saving} --task dacifar100-fedgan-conditional-partial --identifier padpaf-fedbn -b 64 -sw 10
