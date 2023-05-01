# original code is from https://github.com/aaron-xichen/pytorch-playground
# modified by Kimin Lee
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset import *
import os


def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'mnist23689':
        dset = DSET('MNIST', True, batch_size,
                    batch_size, [2, 3, 6, 8, 9], [1, 7])
        train_loader = dset.ind_train_loader
        test_loader = dset.ind_val_loader

    elif data_type == 'fm07':
        dset = DSET('FashionMNIST', True, batch_size,
                    batch_size, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        train_loader = dset.ind_train_loader
        test_loader = dset.ind_val_loader
    elif data_type == 'mnist':
        dset = DSET('MNIST-FashionMNIST', False, batch_size,
                    batch_size, None, None)
        train_loader = dset.ind_train_loader
        test_loader = dset.ind_val_loader

    elif data_type == 'svhn07':
        dset = DSET('SVHN', True, batch_size,
                    batch_size, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        train_loader = dset.ind_train_loader
        test_loader = dset.ind_val_loader

    return train_loader, test_loader


def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader = getSVHN(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        _, test_loader = getCIFAR100(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(
            os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(
            testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(
            testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'mnist17':
        dset = DSET('MNIST', True, batch_size,
                    batch_size, [2, 3, 6, 8, 9], [1, 7])
        test_loader = dset.ood_val_loader
    elif data_type == 'svhn89':
        dset = DSET('SVHN', True, batch_size,
                    batch_size, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        test_loader = dset.ood_val_loader

    elif data_type == 'fm89':
        dset = DSET('FashionMNIST', True, batch_size,
                    batch_size, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        test_loader = dset.ood_val_loader

    elif data_type == 'fm':
        dset = DSET('MNIST-FashionMNIST', False, batch_size,
                    batch_size, None, None)
        test_loader = dset.ood_val_loader
    return test_loader
