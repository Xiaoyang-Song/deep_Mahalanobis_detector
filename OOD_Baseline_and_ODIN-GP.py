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

    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
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
    elif args.dataset == 'mnist32':
        out_dist_list = ['fm32']
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
        if args.dataset == 'mnist32':
            # model = models.DenseNet3(100, int(args.num_classes))
            model = models.DenseNet3GP(100, int(args.num_classes), num_channels,feature_size=n_features)
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))
            in_transform = transforms.Compose([ transforms.Resize((32, 32)), 
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
    train_loader, test_loader = data_loader.getTargetDataSet(
        args.dataset, args.batch_size, in_transform, args.dataroot)

    # measure the performance
    M_list = [0, 0.0005, 0.001, 0.0014, 0.002,
              0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
    T_list = [1, 10, 100, 1000]
    # M_list = [0, 0.0005]
    # T_list = [1, 10]
    base_line_list = []
    ODIN_best_tnr = [0, 0, 0, 0, 0, 0]
    ODIN_best_results = [0, 0, 0, 0, 0, 0]
    ODIN_best_temperature = [-1, -1, -1, -1, -1, -1]
    ODIN_best_magnitude = [-1, -1, -1, -1, -1, -1]
    ODIN_tnr_lst = [[], [], [], [], [], []]
    ODIN_auroc_lst = [[], [], [], [], [], []]

    for T in tqdm(T_list):
        for m in M_list:
            magnitude = m
            temperature = T
            lib_generation.get_posterior(
                model, args.net_type, num_channels, test_loader, magnitude, temperature, args.outf, True)
            out_count = 0
            print('Temperature: ' + str(temperature) + ' / noise: ' + str(magnitude))
            for out_dist in out_dist_list:
                out_test_loader = data_loader.getNonTargetDataSet(
                    out_dist, args.batch_size, in_transform, args.dataroot)
                print('Out-distribution: ' + out_dist)
                lib_generation.get_posterior(model, args.net_type, num_channels, out_test_loader, magnitude, temperature, args.outf, False)
                if temperature == 1 and magnitude == 0:
                    test_results = callog.metric(args.outf, ['PoT'])
                    base_line_list.append(test_results)
                else:
                    val_results = callog.metric(args.outf, ['PoV'])

                    test_results = callog.metric(args.outf, ['PoT'])
                    ODIN_tnr_lst[out_count].append(test_results['PoT']['TNR95'])
                    ODIN_auroc_lst[out_count].append(test_results['PoT']['AUROC'])

                    if ODIN_best_tnr[out_count] < val_results['PoV']['TNR95']:
                        ODIN_best_tnr[out_count] = val_results['PoV']['TNR95']
                        ODIN_best_results[out_count] = callog.metric(args.outf, ['PoT'])
                        ODIN_best_temperature[out_count] = temperature
                        ODIN_best_magnitude[out_count] = magnitude
                out_count += 1

    # print the results
    mtypes = ['TNR95', 'TNR99', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print('Baseline method: in_distribution: ' + args.dataset + '==========')
    count_out = 0
    for results in base_line_list:
        print('out_distribution: ' + out_dist_list[count_out])
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR95']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['TNR99']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
        print('')
        count_out += 1

    print('ODIN method: in_distribution: ' + args.dataset + '==========')
    count_out = 0
    for results in ODIN_best_results:
        print('out_distribution: ' + out_dist_list[count_out])
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR95']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['TNR99']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
        print('temperature: ' + str(ODIN_best_temperature[count_out]))
        print('magnitude: ' + str(ODIN_best_magnitude[count_out]))
        print('')
        print(100*np.round(ODIN_tnr_lst[count_out],2))
        print(100*np.round(np.mean(ODIN_tnr_lst[count_out]), 2))
        print(100*np.round(ODIN_auroc_lst[count_out],2))
        print(100*np.round(np.mean(ODIN_auroc_lst[count_out]), 2))
        count_out += 1


if __name__ == '__main__':
    main()
