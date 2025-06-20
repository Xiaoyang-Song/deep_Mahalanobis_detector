#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=svhn
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/deep_Mahalanobis_detector/jobs/SEEOOD_Baselines/out/svhn.log


python OOD_Baseline_and_ODIN.py --dataset svhn07 --net_type densenet --gpu 0 --metric original

python OOD_Generate_Mahalanobis.py --dataset svhn07 --net_type densenet --num_classes 8 --gpu 0

python OOD_Regression_Mahalanobis.py --net_type densenet --ind_dset svhn07