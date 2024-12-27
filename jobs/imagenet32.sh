#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=IM32
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/deep_Mahalanobis_detector/jobs/im32-1500.log


# python OOD_Generate_Mahalanobis-GP.py --dataset imagenet10 --ckpt imagenet10-32-0-o1 --nf 32 --net_type densenet --gpu 0 --n_test_prep 5000
python OOD_Regression_Mahalanobis-GP-Eval.py --net_type densenet --ind_dset imagenet10 --nf 32
