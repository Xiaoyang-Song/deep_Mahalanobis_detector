#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=IM64
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/deep_Mahalanobis_detector/jobs/im64.log


python OOD_Generate_Mahalanobis-GP.py --dataset imagenet10 --ckpt imagenet10-64-0-o1 --nf 64 --net_type densenet --gpu 0
python OOD_Regression_Mahalanobis-GP-Eval.py --net_type densenet --ind_dset imagenet10 --nf 64
