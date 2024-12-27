#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=IM32-OD
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/deep_Mahalanobis_detector/jobs/im32-odin-1500.log


python OOD_Baseline_and_ODIN-GP-Eval.py --dataset imagenet10 --net_type densenet --ckpt imagenet10-32-0-o1 --nf 32 --gpu 0 --batch_size 100 --n_test_prep 5000
