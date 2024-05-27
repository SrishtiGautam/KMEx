import os,sys,time,copy

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils.model_february import *
from utils.dataloader_qd import load_QuickDraw
from utils.dataloader_nrm import NRM
from utils.dataloader_celeba import CelebA

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 128 if torch.cuda.is_available() else 0

def build_loader(DATASET, BATCH=128):

    m,s = NRM[DATASET]

    if DATASET == 'mnist':
        transform = transforms.Compose([transforms.Resize(32),transforms.Grayscale(num_output_channels=3),transforms.ToTensor(), transforms.Normalize(m, s)])
        train_dataset = datasets.MNIST('../Data/', train=True,  download=True, transform=transform)
        push_dataset  = datasets.MNIST('../Data/', train=False, download=True, transform=transform)
        test_dataset  = datasets.MNIST('../Data/', train=False, download=True, transform=transform)
    elif DATASET == 'fmnist':
        transform = transforms.Compose([transforms.Resize(32),transforms.Grayscale(num_output_channels=3),transforms.ToTensor(), transforms.Normalize(m, s)])
        train_dataset = datasets.FashionMNIST('../Data/', train=True,  download=True, transform=transform)
        push_dataset  = datasets.FashionMNIST('../Data/', train=True,  download=True, transform=transform)
        test_dataset  = datasets.FashionMNIST('../Data/', train=False, download=True, transform=transform)
    elif DATASET == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(m, s)])
        train_dataset = datasets.SVHN('../Data/', split='train', download=True, transform=transform)
        push_dataset  = datasets.SVHN('../Data/', split='train', download=True, transform=transform)
        test_dataset  = datasets.SVHN('../Data/', split='test',  download=True, transform=transform)
    elif DATASET == 'quickdraw':
        train_dataset = load_QuickDraw('../Data/QuickDraw/FLINT', split='train', channels=3)
        push_dataset  = load_QuickDraw('../Data/QuickDraw/FLINT', split='train', channels=3)
        test_dataset  = load_QuickDraw('../Data/QuickDraw/FLINT', split='test', channels=3)
    elif DATASET == 'stl10':
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(96, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(m, s)
            ])
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(m, s)
            ])
        train_dataset = datasets.STL10('../Data/', split='train', download=True, transform=transform_train)
        push_dataset = datasets.STL10('../Data/', split='train', download=True, transform=transform)
        test_dataset  = datasets.STL10('../Data/', split='test',  download=True, transform=transform)
    elif DATASET == 'cifar10':
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(m, s)
            ])
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(m, s)
            ])
        train_dataset = datasets.CIFAR10('../Data/', train=True,  download=True, transform=transform_train)
        push_dataset  = datasets.CIFAR10('../Data/', train=True,  download=True, transform=transform)
        test_dataset  = datasets.CIFAR10('../Data/', train=False, download=True, transform=transform)
    elif DATASET == 'celeba':
        transform_train = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(224, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(m, s)
            ])
        transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(m, s)
            ])
        train_dataset = CelebA(root='../Data/', split='train', target_type="attr",
                                download=False, transform=transform_train)
        push_dataset  = CelebA(root='../Data/', split='train', target_type="attr",
                                download=False, transform=transform)
        test_dataset  = CelebA(root='../Data/', split='test', target_type="attr",
                               download=False, transform=transform)
        train_dataset.attr_all = train_dataset.attr
        push_dataset.attr_all = push_dataset.attr
        test_dataset.attr_all = test_dataset.attr
        #
        train_dataset.attr = train_dataset.attr[:,20:21]
        push_dataset.attr = push_dataset.attr[:,20:21]
        test_dataset.attr = test_dataset.attr[:,20:21]
    else:
        print('! Wrong dataset !')
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH,
        num_workers=num_workers,
        drop_last=True
    )

    push_loader = torch.utils.data.DataLoader(
        push_dataset,
        shuffle=False,
        batch_size=BATCH,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False
    )


    return train_loader,push_loader,test_loader


##########################################################################

def build_model(title, model):


    if 'RESNET34' in title:
        model.LATENT = 512
        model.ARCHITECTURE = 'RESNET34'

        model.encoder = torchvision.models.resnet34(pretrained=True).to(device)
        model.encoder.fc = nn.Identity().to(device)
        model.clf = nn.Linear(model.LATENT, model.K, bias=False).to(device)

    elif 'DENSENET121' in title:
        model.LATENT = 1024
        model.ARCHITECTURE = 'DENSENET121'

        model.encoder = torchvision.models.densenet121(pretrained=True).features.to(device)
        model.encoder.add_module('avgpool',nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        model.encoder.add_module('flat',nn.Flatten().to(device))
        model.clf = nn.Linear(model.LATENT, model.K, bias=False).to(device)

    elif 'VGG16' in title:
        model.LATENT = 512
        model.ARCHITECTURE = 'VGG16'

        model.encoder = torchvision.models.vgg16(pretrained=True).to(device).features
        model.encoder.add_module('avgpool',nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        model.encoder.add_module('flat',nn.Flatten().to(device))
        model.clf = nn.Linear(model.LATENT, model.K, bias=False).to(device)

    if 'PROTOVAE' in title:
        model.ARCHITECTURE = 'PROTOVAE'
        #
        if('RESNET34') in title:
            model.encoder.avgpool = nn.Sequential(
            nn.Conv2d(in_channels=model.LATENT, out_channels=model.LATENT * 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(model.LATENT*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=model.LATENT * 2, out_channels=model.LATENT * 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(model.LATENT * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ).to(device)
        else:
            model.LATENT = 512
            model.encoder = ENCODER(model.LATENT*2, model.DATASET).to(device)

        model.decoder = DECODER(model.LATENT, model.DATASET).to(device)

        model.clf = nn.Linear(model.NB_PRT, model.K, bias=False).to(device)
        model.prt = torch.randn((model.NB_PRT,model.LATENT), dtype=torch.float32, requires_grad=True, device=device)
        # nn.init.orthogonal_(model.prt)

        cls = torch.repeat_interleave(torch.arange(model.K),model.H,0)
        model.prt_classes = torch.tensor(cls, dtype=torch.float32, requires_grad=False).to(device)

        prototype_class_identity = torch.zeros(model.NB_PRT,
                                                    model.K)

        for j in range(model.NB_PRT):
            prototype_class_identity[j, j // model.H] = 1

        positive_one_weights_locations = torch.t(prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = -0.5
        model.clf.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)


    # return model
