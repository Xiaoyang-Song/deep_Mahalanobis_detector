#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=MN32-OD
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/deep_Mahalanobis_detector/jobs/mn32-odin-2000.log


python OOD_Baseline_and_ODIN-GP-Eval.py --dataset mnist --net_type densenet --ckpt mnist-32 --nf 32 --gpu 0 --batch_size 100 --n_test_prep 5000
