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
import models
import os
import lib_generation

from torchvision import transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(
    description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200,
                    metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True,
                    help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='./output/',
                    help='folder to output results')
parser.add_argument('--num_classes', type=int,
                    default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)


def main():
    C = 3
    # set the path to pre-trained model and output
    # pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    if args.dataset == 'mnist':
        experiment = 'mnist'
    elif args.dataset == 'cifar10':
        experiment = 'CIFAR10'
    else:
        assert False
    # GP-OOD
    pre_trained_net = f"/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/ckpt/{experiment}/densenet.pth"

    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
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
    elif args.dataset == 'mnist23689':
        out_dist_list = ['mnist17']
        C = 1

    # FashionMNIST Within-Dataset Experiment
    elif args.dataset == 'fm07':
        out_dist_list = ['fm89']
        C = 1

    # MNIST-FashionMNIST Between-Dataset Experiment
    elif args.dataset == 'mnist':
        out_dist_list = ['fm']
        C = 3
        n_features=64

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

        # FashionMNIST Within-Dataset Experiment
        elif args.dataset == 'fm07':
            model = models.DenseNet3(100, num_channels=1, num_classes=8)
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))
            in_transform = transforms.Compose([transforms.ToTensor()])

        # MNIST Within-Dataset Experiment
        elif args.dataset == 'mnist23689':
            model = models.DenseNet3(100, num_channels=1, num_classes=5)
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))
            in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        # MNIST-FashionMNIST Between-Dataset Experiment
        elif args.dataset == 'mnist':
            model = models.DenseNet3GP(100, num_channels=3, num_classes=10, feature_size=n_features)
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

    elif args.net_type == "dcd":
        if args.dataset == 'mnist23689':
            model = models.DC_D(5,  {'H': 28, 'W': 28, 'C': 1})
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

    # set information about feature extaction
    model.eval()
    if C == 1:
        temp_x = torch.rand(2, 1, 28, 28).cuda()
    else:
        temp_x = torch.rand(2, C, 32, 32).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(
        model, args.num_classes, feature_list, train_loader)

    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('Noise: ' + str(magnitude))
        for i in range(num_output):
            M_in = lib_generation.get_Mahalanobis_score(model, test_loader, args.num_classes, args.outf,
                                                        True, args.net_type, sample_mean, precision, i,
                                                        magnitude, C=C)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate(
                    (Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

        for out_dist in out_dist_list:
            out_test_loader = data_loader.getNonTargetDataSet(
                out_dist, args.batch_size, in_transform, args.dataroot)
            print('Out-distribution: ' + out_dist)
            for i in range(num_output):
                M_out = lib_generation.get_Mahalanobis_score(model, out_test_loader, args.num_classes, args.outf,
                                                             False, args.net_type, sample_mean, precision, i, magnitude, C=C)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate(
                        (Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(
                Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (
                str(magnitude), args.dataset, out_dist))
            Mahalanobis_data = np.concatenate(
                (Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)


if __name__ == '__main__':
    main()
