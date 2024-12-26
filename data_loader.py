# original code is from https://github.com/aaron-xichen/pytorch-playground
# modified by Kimin Lee
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from dataset import *
import os

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import shutil
from tqdm import tqdm
import argparse

torch.manual_seed(2024)
np.random.seed(2024)


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
class ImageSubfolder(DatasetFolder):
    """Extend ImageFolder to support fold subsets
    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        class_to_idx (dict): Dict with items (class_name, class_index).
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        class_to_idx: Optional[Dict] = None,
    ):
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        if class_to_idx is not None:
            classes = class_to_idx.keys()
        else:
            classes, class_to_idx = self.find_classes(self.root)
        extensions = IMG_EXTENSIONS if is_valid_file is None else None,
        samples = self.make_dataset(self.root, class_to_idx, extensions[0], is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

def imagenet10_set_loader(bsz, dset_id, small=True):
    n = 32 if small else 224
    train_transform = transforms.Compose([
        transforms.Resize(size=(n, n), interpolation=transforms.InterpolationMode.BICUBIC),
        # trn.RandomResizedCrop(size=(224, 224), scale=(0.5, 1), interpolation=trn.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size=(n, n), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(n, n)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    root_dir = '../GP-ImageNet/data/'
    train_dir = root_dir + 'val'
    classes, _ = torchvision.datasets.folder.find_classes(train_dir)

    # # Choose class
    indices = [[895, 817, 10, 284, 352, 238, 30, 569, 339, 510],
               [648, 506, 608, 640, 539, 548, 446, 183, 809, 127],
               [961, 316, 227, 74, 322, 480, 933, 508, 158, 367],
               [247, 202, 622, 351, 367, 523, 796, 91, 39, 54],
               [114, 183, 841, 870, 730, 756, 554, 799, 97, 150],
               [795, 854, 631, 581, 669, 573, 310, 900, 569, 598],
               [310, 404, 382, 136, 786, 97, 858, 970, 391, 688],
               [744, 437, 606, 909, 96, 951, 384, 43, 461, 247],
               [534, 358, 139, 955, 304, 879, 998, 319, 359, 904],
               [461, 29, 22, 254, 560, 232, 700, 45, 363, 321],
               [8, 641, 417, 181, 813, 64, 396, 437, 7, 178]]
    index = indices[dset_id]

    classes = [classes[i] for i in index]
    # print(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_data = ImageSubfolder(root_dir + 'train', transform=train_transform, class_to_idx=class_to_idx)
    test_data = ImageSubfolder(root_dir + 'val', transform=test_transform, class_to_idx=class_to_idx)
    return train_data, test_data


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

    # CIFAR10 as InD
    if data_type == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_set = torchvision.datasets.CIFAR10('./Datasets/CIFAR-10', train=True, transform=train_transform, download=True)
        # Truncate testing data
        n_test = 5000
        test_set = torchvision.datasets.CIFAR10('./Datasets/CIFAR-10', train=False, transform=test_transform, download=True)
        test_set = torch.utils.data.Subset(test_set, range(n_test))
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=True, num_workers=1)
        
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(
            batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)

    # MNIST as InD
    elif data_type == 'mnist':
        transform = transforms.Compose([ transforms.Resize((32, 32)), 
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST("./Datasets", download=True, transform=transform)
        n_test=5000
        test_set = torchvision.datasets.MNIST("./Datasets", download=True, train=False, transform=transform)
        test_set = torch.utils.data.Subset(test_set, range(n_test))
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=True, num_workers=1)
    
    elif data_type == 'imagenet10':
        train_set, test_set = imagenet10_set_loader(256, 0)
        total_size = len(train_set)
        train_ratio = 0.8
        val_ratio = 0.2
        print('Training dataset size: ', total_size)
        # Calculate sizes for each split
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        if train_size + val_size != total_size:
            val_size = val_size + 1 # This is specifically for imagenet100


        # Perform the split
        train_dataset, validation_dataset = torch.utils.data.random_split(train_set, [train_size, val_size])
        print("Dataset size: ", len(train_dataset), len(validation_dataset), len(test_set))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_set = validation_dataset + test_set
        print(len(test_set))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot, n_test=5000):

    # CIFAR10-SVHN Between-Dataset Experiment: OoD
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
        
    # MNIST Within-Dataset Experiment: OoD 
    elif data_type == 'mnist17':
        dset = DSET('MNIST', True, batch_size,
                    batch_size, [2, 3, 6, 8, 9], [1, 7])
        test_loader = dset.ood_val_loader

    # SVHN Within-Dataset Experiment: OoD
    elif data_type == 'svhn89':
        dset = DSET('SVHN', True, batch_size,
                    batch_size, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        test_loader = dset.ood_val_loader

    # FashionMNIST Within-Dataset Experiment: OoD
    elif data_type == 'fm89':
        dset = DSET('FashionMNIST', True, batch_size,
                    batch_size, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        test_loader = dset.ood_val_loader

    # MNIST-FashionMNIST Between-Dataset Experiment: OoD
    elif data_type == 'fm':
        transform = transforms.Compose([ transforms.Resize((32, 32)), 
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor()])
        n_test=5000
        tset = torchvision.datasets.FashionMNIST("./Datasets", download=True, train=True, transform=transform)
        tset = torch.utils.data.Subset(tset, range(n_test))
        # Get data loader
        test_loader = torch.utils.data.DataLoader(tset, shuffle=False, batch_size=256)

    
    elif data_type == 'LSUN-C':
        print('######################################')
        print('Testing on LSUN-C')
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        data = torchvision.datasets.ImageFolder(root="../GP-ImageNet/data/LSUN/",
                                    transform=transforms.Compose([transforms.Resize((32, 32)), 
                                                                transforms.CenterCrop(32), 
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean, std)]))
        tset = torch.utils.data.Subset(data, range(n_test))
        test_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=16)

    elif data_type == 'LSUN-R':
        print('######################################')
        print('Testing on LSUN-R')
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        data = torchvision.datasets.ImageFolder(root="../GP-ImageNet/data/LSUN_resize/",
                                    transform=transforms.Compose([transforms.Resize((32, 32)), 
                                                                transforms.CenterCrop(32), 
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean, std)]))
        tset = torch.utils.data.Subset(data, range(n_test))
        test_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=16)

    elif data_type == 'iSUN':
        print('######################################')
        print('Testing on iSUN')
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        data = torchvision.datasets.ImageFolder(root="../GP-ImageNet/data/iSUN/",
                                    transform=transforms.Compose([transforms.Resize((32, 32)), 
                                                                transforms.CenterCrop(32), 
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean, std)]))
        tset = torch.utils.data.Subset(data, range(n_test))
        test_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=16)

    elif data_type == 'Places365-small':
        print('######################################')
        print('Testing on Places365')
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        data = datasets.Places365(root="../GP-ImageNet/data/", split='val', small=True, download=False, 
                                transform=transforms.Compose([transforms.Resize((32, 32)), 
                                                                transforms.CenterCrop(32), 
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean, std)]))
        tset = torch.utils.data.Subset(data, range(n_test))
        test_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=16)

    elif data_type == 'DTD':
        print('######################################')
        print('Testing on DTD Texture')
        data = torchvision.datasets.ImageFolder(root="../GP-ImageNet/data/dtd/images/",
                                    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.CenterCrop(32), 
                                                                  transforms.ToTensor(), 
                                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))
        tset = torch.utils.data.Subset(data, range(n_test))
        test_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=16)

    return test_loader
