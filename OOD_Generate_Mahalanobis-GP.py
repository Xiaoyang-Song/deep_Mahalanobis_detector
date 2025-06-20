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
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200,
                    metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True,
                    help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
# Only for imagenet 10
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint')
parser.add_argument('--nf', type=int, default=None, help='n_features')
parser.add_argument('--n_test_prep', type=int, default=None, help='n_test')
# For saving files
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
    if args.dataset == 'mnist':
        assert args.ckpt is not None
        experiment = args.ckpt
        pre_trained_net = f"/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/ckpt/{experiment}/densenet_{args.dataset}.pth"
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
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100
        
    elif args.dataset == 'mnist32':
        out_dist_list = ['fm32', 'svhn', 'imagenet-c', 'cifar10']
        n_val = 2000
        n_ind_test = 2000
        n_ood_test = 2000
        num_channels = 3
        assert args.nf is not None
        n_features = args.nf

    elif args.dataset == 'imagenet10':
        out_dist_list = ['DTD', 'LSUN-C', 'LSUN-R', 'Places365-small', 'iSUN', 'svhn']
        n_val = 1500
        n_ind_test = 1600
        n_ood_test = 1600
        num_channels = 3
        assert args.nf is not None
        n_features = args.nf


    # load networks
    # This part is customized
    if args.net_type == 'densenet':
        # Useless
        if args.dataset == 'imagenet10':
            model = models.DenseNet3GP(100, num_channels=num_channels, num_classes=10, feature_size=args.nf)
            model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))

        elif args.dataset == 'mnist32':
            model = models.DenseNet3GP(100, num_channels=3, num_classes=10, feature_size=n_features)
            model.load_state_dict(torch.load(
                pre_trained_net, map_location="cuda:" + str(args.gpu)))
            in_transform = transforms.Compose([ transforms.Resize((32, 32)), 
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()])

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

    model.cuda()
    print('load model: ' + args.net_type)

    # load dataset
    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)

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
    sample_mean, precision = lib_generation.sample_estimator(model, args.num_classes, feature_list, train_loader)

    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in tqdm(m_list):
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
            out_test_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, args.dataroot, n_test=args.n_test_prep)
            print('Out-distribution: ' + out_dist)
            for i in range(num_output):
                M_out = lib_generation.get_Mahalanobis_score(model, out_test_loader, args.num_classes, args.outf,
                                                             False, args.net_type, sample_mean, precision, i, magnitude, C=C)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            print(Mahalanobis_in.shape)
            print(Mahalanobis_out.shape)
            print("After processing...")
            Mahalanobis_in = Mahalanobis_in[0:n_val + n_ind_test]
            Mahalanobis_out = Mahalanobis_out[0:n_val + n_ood_test]
            print(Mahalanobis_in.shape)
            print(Mahalanobis_out.shape)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.dataset, out_dist))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)


if __name__ == '__main__':
    main()
