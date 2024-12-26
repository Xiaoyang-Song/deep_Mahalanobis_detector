#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=IM64-OD
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/deep_Mahalanobis_detector/jobs/im64-odin-1500.log

python OOD_Baseline_and_ODIN-GP-Eval.py --dataset imagenet10 --net_type densenet --ckpt imagenet10-64-0-o1 --nf 64 --gpu 0 --batch_size 100 --n_test 5000
