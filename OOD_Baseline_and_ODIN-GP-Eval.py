"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import calculate_log as callog
import densenet
import models
import os
import lib_generation
from tqdm import tqdm

from torchvision import transforms
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(
    description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200,
                    metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True,
                    help='cifar10 | cifar100 | svhn')
# Only for imagenet 10
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint')
parser.add_argument('--nf', type=int, default=None, help='n_features')
# For saving files
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='output/',
                    help='folder to output results')
parser.add_argument('--num_classes', type=int,
                    default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)


def main():
    # set the path to pre-trained model and output
    # pre_trained_net = 'pre_trained/' + args.net_type + '_' + args.dataset + '.pth'

    if args.dataset == 'mnist':
        experiment = 'mnist'
        pre_trained_net = f"/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/ckpt/{experiment}/densenet.pth"
    elif args.dataset == 'cifar10':
        experiment = 'CIFAR10'
        pre_trained_net = f"/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/ckpt/{experiment}/densenet.pth"
    elif args.dataset == 'imagenet10':
        assert args.ckpt is not None
        experiment = args.ckpt
        pre_trained_net = f"/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/ckpt/{experiment}/densenet_{args.dataset}.pth"
    else:
        assert False

    args.outf = args.outf + args.net_type + '_' + args.dataset + '_' + str(args.nf) + '/'
    os.makedirs(args.outf, exist_ok=True)
    # if os.path.isdir(args.outf) == False:
    #     os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100

    # CIFAR10-SVHN Between-Dataset Experiment
    elif args.dataset == 'cifar10':
        out_dist_list = ['svhn']

    # MNIST-FashionMNIST Between-Dataset Experiment
    elif args.dataset == 'mnist':
        out_dist_list = ['fm']
        num_channels=3
        n_features=64
    elif args.dataset == 'imagenet10':
        out_dist_list = ['DTD', 'LSUN-C', 'LSUN-R', 'Places365-small', 'iSUN', 'svhn']
        # out_dist_list = ['iSUN']
        num_channels = 3
        assert args.nf is not None
        n_features = args.nf

    # load networks
    # This part is customized
    if args.net_type == 'densenet':

        # Useless
        if args.dataset == 'mnist':
            model = models.DenseNet3GP(100, int(args.num_classes), num_channels,feature_size=n_features)
            model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
            in_transform = transforms.Compose([transforms.Resize((32, 32)), 
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()])

        # CIFAR10-SVHN Between-Dataset Experiment
        elif args.dataset == 'cifar10':
            model = models.DenseNet3(100, num_channels=3, num_classes=10)
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))
            in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            
        elif args.dataset == 'imagenet10':
            model = models.DenseNet3GP(100, num_channels=3, num_classes=10, feature_size=args.nf)
            model_state = torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu))
            model.load_state_dict(model_state)
            in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        else:
            model = torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu))
            in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)), ])

    elif args.net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(
            pre_trained_net, map_location="cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    model.cuda()
    print('load model: ' + args.net_type)

    # load dataset
    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)

    # measure the performance
    TPR=0.95
    M_list = [0, 0.0005, 0.001, 0.0014, 0.002,
              0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
    T_list = [1, 10, 100, 1000]
    # M_list = [0, 0.0005]
    # T_list = [1, 10]

    # BASELINE
    base_line_list_ind = []
    base_line_list_ood = []
    base_line_list_auroc = []
    # ODIN
    odin_ind_acc = [[], [], [], [], [], []]
    odin_ood_acc = [[], [], [], [], [], []]
    odin_auroc = [[], [], [], [], [], []]

    for T in tqdm(T_list):
        for m in M_list:
            magnitude = m
            temperature = T
            lib_generation.get_posterior(model, args.net_type, num_channels, test_loader, magnitude, temperature, args.outf, True)
            out_count = 0
            print('Temperature: ' + str(temperature) + ' / noise: ' + str(magnitude))
            for out_dist in out_dist_list:
                out_test_loader = data_loader.getNonTargetDataSet(
                    out_dist, args.batch_size, in_transform, args.dataroot)
                print('Out-distribution: ' + out_dist)
                lib_generation.get_posterior(model, args.net_type, num_channels, out_test_loader, magnitude, temperature, args.outf, False)

                # InD and OoD validation set
                dir_name = args.outf
                ind_val = np.loadtxt('{}/confidence_{}_In.txt'.format(dir_name, 'PoV'), delimiter=' ')
                ood_val = np.loadtxt('{}/confidence_{}_Out.txt'.format(dir_name, 'PoV'), delimiter=' ')
                ind_test = np.loadtxt('{}/confidence_{}_In.txt'.format(dir_name, 'PoT'), delimiter=' ')
                ood_test = np.loadtxt('{}/confidence_{}_Out.txt'.format(dir_name, 'PoT'), delimiter=' ')
                # Lower -> OOD; Higher -> InD
                threshold = np.quantile(ind_val, 1 - TPR)
                # Print out test statistics
                print(f"Testing set size: {len(ind_test)}, {len(ood_test)}")
                ind_acc = sum(ind_test >= threshold) / len(ind_test)
                ood_acc = sum(ood_test < threshold) / len(ood_test)
                # AUROC calculation
                scores = np.concatenate((ind_test, ood_test))  # Combine the arrays
                labels = np.concatenate((np.ones(ind_test.shape[0]), np.zeros(ood_test.shape[0])))  # Labels: 0 for InD, 1 for OoD
                # Calculate AUROC
                auroc = roc_auc_score(labels, scores)
                # Find threshold using validation set
                if temperature == 1 and magnitude == 0:
                    base_line_list_ind.append(ind_acc)
                    base_line_list_ood.append(ood_acc)
                    base_line_list_auroc.append(auroc)
                else:
                    odin_ind_acc[out_count].append(ind_acc)
                    odin_ood_acc[out_count].append(ood_acc)
                    odin_auroc[out_count].append(auroc)
                out_count += 1

    # print the results
    print('Baseline method: in_distribution: ' + args.dataset + '==========')

    for out_idx, name in enumerate(out_dist_list):
        print('out_distribution: ' + name)
        print(100*np.round(base_line_list_ind[out_idx], 2))
        print(100*np.round(base_line_list_ood[out_idx], 2))
        print(100*np.round(base_line_list_auroc[out_idx], 2))

    print('ODIN method: in_distribution: ' + args.dataset + '==========')
    for count_out, name in enumerate(out_dist_list):
        print('out_distribution: ' + name)
        print('IND ACC')
        print(100*np.round(odin_ind_acc[count_out],2))
        print(100*np.round(np.mean(odin_ind_acc[count_out]), 2))
        print("OOD ACC")
        print(100*np.round(odin_ood_acc[count_out],2))
        print(100*np.round(np.mean(odin_ood_acc[count_out]), 2))
        print("AUROC")
        print(100*np.round(odin_auroc[count_out],2))
        print(100*np.round(np.mean(odin_auroc[count_out]), 2))


if __name__ == '__main__':
    main()
