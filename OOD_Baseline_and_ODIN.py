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

from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(
    description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=100,
                    metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True,
                    help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='output/',
                    help='folder to output results')
parser.add_argument('--num_classes', type=int,
                    default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--metric', type=str, default='original', help='original | no_tuning')
args = parser.parse_args()
print(args)


def main():
    # set the path to pre-trained model and output
    # pre_trained_net = 'pre_trained/' + args.net_type + '_' + args.dataset + '.pth'

    if args.dataset == 'cifar10':
        experiment = 'CIFAR10-SVHN'
    elif args.dataset == 'svhn07':
        experiment = 'SVHN'
    elif args.dataset == 'mnist07':
        experiment = 'MNIST'
    elif args.dataset == 'fm07':
        experiment = 'FashionMNIST'
    elif args.dataset == 'mnist':
        experiment = 'MNIST-FashionMNIST'
    else:
        assert False

    # Please replace the following path with your own path for pretrained model
    pre_trained_net = f"/scratch/sunwbgt_root/sunwbgt98/xysong/deep_Mahalanobis_detector/pre_trained/[{experiment}]-pretrained-classifier.pt"

    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    os.makedirs(args.outf, exist_ok=True)
    # if os.path.isdir(args.outf) == False:
    #     os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100

    if args.dataset == 'svhn':
        out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize']

    # CIFAR10-SVHN Between-Dataset Experiment
    elif args.dataset == 'cifar10':
        out_dist_list = ['svhn']

    # MNIST Within-Dataset Experiment
    elif args.dataset == 'mnist07':
        out_dist_list = ['mnist89']

    # FashionMNIST Within-Dataset Experiment
    elif args.dataset == 'fm07':
        out_dist_list = ['fm89']

    # MNIST-FashionMNIST Between-Dataset Experiment
    elif args.dataset == 'mnist':
        out_dist_list = ['fm']

    # SVHN Within-Dataset Experiment
    elif args.dataset == 'svhn07':
        out_dist_list = ['svhn89']

    # load networks
    # This part is customized
    if args.net_type == 'densenet':

        # Useless
        if args.dataset == 'svhn':
            model = models.DenseNet3(100, int(args.num_classes))
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))

        # SVHN Within-Dataset Experiment
        elif args.dataset == 'svhn07':
            model = models.DenseNet3(100, num_channels=3, num_classes=8)
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))
            num_channels = 3

        # FashionMNIST Within-Dataset Experiment
        elif args.dataset == 'fm07':
            model = models.DenseNet3(100, num_channels=1, num_classes=8)
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))
            in_transform = transforms.Compose([transforms.ToTensor()])
            num_channels = 1

        # MNIST Within-Dataset Experiment
        elif args.dataset == 'mnist07':
            model = models.DenseNet3(100, num_channels=1, num_classes=8)
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))
            in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            num_channels = 1

        # MNIST-FashionMNIST Between-Dataset Experiment
        elif args.dataset == 'mnist':
            model = models.DenseNet3(100, num_channels=1, num_classes=10)
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))
            in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            num_channels = 1

        # CIFAR10-SVHN Between-Dataset Experiment
        elif args.dataset == 'cifar10':
            model = models.DenseNet3(100, num_channels=3, num_classes=10)
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))
            in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            num_channels = 3

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

    if args.metric == 'original':
        print("Evaluating with original ODIN method (OOD samples are used for tuning parameters...)")
        print("Note that this is an unfaithful comparison as the paper claims using no OOD samples.")
        # measure the performance
        M_list = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
        T_list = [1, 10, 100, 1000]
        M_list = [0.005]
        T_list = [1000]
        base_line_list = []
        ODIN_best_tnr = [0, 0, 0]
        ODIN_best_results = [0, 0, 0]
        ODIN_best_temperature = [-1, -1, -1]
        ODIN_best_magnitude = [-1, -1, -1]
        for T in T_list:
            for m in M_list:
                magnitude = m
                temperature = T
                lib_generation.get_posterior(model, args.net_type, num_channels, test_loader, magnitude, temperature, args.outf, True)
                out_count = 0
                print('Temperature: ' + str(temperature) +
                    ' / noise: ' + str(magnitude))
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
                        print(callog.metric(args.outf, ['PoT'])['PoT']['TNR95'])
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
            count_out += 1

    elif args.metric == 'no_tuning':
        # All possible parameter range
        M_list = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
        T_list = [1, 10, 100, 1000]

        # BASELINE
        base_line_list_tpr95 = []
        base_line_list_tpr99 = []
        base_line_list_auroc = []
        # ODIN
        odin_tpr_95 = [[], [], [], [], [], []]
        odin_tpr_99 = [[], [], [], [], [], []]
        odin_auroc = [[], [], [], [], [], []]

        for T in tqdm(T_list):
            for m in M_list:
                magnitude = m
                temperature = T
                lib_generation.get_posterior(model, args.net_type, num_channels, test_loader, magnitude, temperature, args.outf, True)
                out_count = 0
                print('Temperature: ' + str(temperature) + ' / noise: ' + str(magnitude))
                for out_dist in out_dist_list:
                    out_test_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, args.dataroot)
                    print('Out-distribution: ' + out_dist)
                    lib_generation.get_posterior(model, args.net_type, num_channels, out_test_loader, magnitude, temperature, args.outf, False)

                    # InD and OoD testing without using validation set
                    dir_name = args.outf
                    ind_val = np.loadtxt('{}/confidence_{}_In.txt'.format(dir_name, 'PoV'), delimiter=' ')
                    ood_val = np.loadtxt('{}/confidence_{}_Out.txt'.format(dir_name, 'PoV'), delimiter=' ')
                    ind_test = np.loadtxt('{}/confidence_{}_In.txt'.format(dir_name, 'PoT'), delimiter=' ')
                    # print(len(ind_test))
                    # ind_test = ind_test[0:n_ind_test]
                    ood_test = np.loadtxt('{}/confidence_{}_Out.txt'.format(dir_name, 'PoT'), delimiter=' ')
                    # print(len(ood_test))
                    # ood_test = ood_test[0:n_ood_test]
                    # Lower -> OOD; Higher -> InD
                    threshold95 = np.quantile(ind_test, 0.05)
                    threshold99 = np.quantile(ind_test, 0.01)
                    # Print out test statistics
                    print(f"Testing set size: {len(ind_test)}, {len(ood_test)}")
                    tpr95 = sum(ood_test < threshold95) / len(ood_test)
                    tpr99 = sum(ood_test < threshold99) / len(ood_test)
                    # AUROC calculation
                    scores = np.concatenate((ind_test, ood_test))  # Combine the arrays
                    labels = np.concatenate((np.ones(ind_test.shape[0]), np.zeros(ood_test.shape[0])))  # Labels: 0 for OoD, 1 for InD
                    # Calculate AUROC
                    auroc = roc_auc_score(labels, scores)
                    print(f"TPR95: {tpr95:.4f}, TPR99: {tpr99:.4f}, AUROC: {auroc:.4f}")
                    # Find threshold using validation set
                    if temperature == 1 and magnitude == 0:
                        base_line_list_tpr95.append(tpr95)
                        base_line_list_tpr99.append(tpr99)
                        base_line_list_auroc.append(auroc)
                    else:
                        odin_tpr_95[out_count].append(tpr95)
                        odin_tpr_99[out_count].append(tpr99)
                        odin_auroc[out_count].append(auroc)
                    out_count += 1

        # print the results
        prec = 5
        print('Baseline method: in_distribution: ' + args.dataset + '==========')

        for out_idx, name in enumerate(out_dist_list):
            print('out_distribution: ' + name)
            print(100*np.round(base_line_list_tpr95[out_idx], prec))
            print(100*np.round(base_line_list_tpr99[out_idx], prec))
            print(100*np.round(base_line_list_auroc[out_idx], prec))

        print('ODIN method: in_distribution: ' + args.dataset + '==========')
        for count_out, name in enumerate(out_dist_list):
            print('out_distribution: ' + name)
            print('TPR95')
            print(100*np.round(odin_tpr_95[count_out], prec))
            print(100*np.round(np.mean(odin_tpr_95[count_out]), prec))
            print("TPR99")
            print(100*np.round(odin_tpr_99[count_out], prec))
            print(100*np.round(np.mean(odin_tpr_99[count_out]), prec))
            print("AUROC")
            print(100*np.round(odin_auroc[count_out],prec))
            print(100*np.round(np.mean(odin_auroc[count_out]), prec))


if __name__ == '__main__':
    main()
