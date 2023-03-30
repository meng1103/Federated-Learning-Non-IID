import os.path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from models.sample import *
from utils.config import args_parser
import torch
import numpy as np
from models.ResNet import resnet18
import torch.optim as optim
import matplotlib.pyplot as plt



class GetDataSet(object):
    def __init__(self, args):
        self.args = args
        self.train_loader = None
        self.train_size = None
        self.train_label = None
        self.test_loader = None
        self.test_label = None
        self.test_size = None

        self.NICO_train_loader = []
        self.NICO_test_loader = None

        self.train_data = None
        self.train_dict = {}

        if self.args.dataset == 'mnist':
            self.MnistDataSetConstruct(self.args)
        elif self.args.dataset == 'cifar':
            self.CifarDataSetConstruct(self.args)
        elif self.args.dataset == 'cifar_rate' or self.args.dataset == 'mnist_rate':
            self.RateDataSetConstruct(self.args)
        elif self.args.dataset == 'cifar_LDA' or self.args.dataset == 'cifar100_LDA':
            self.LDADataSetConstruct(self.args)
        elif self.args.dataset == 'cifar100_rate':
            self.RateDataSetConstruct(self.args)

        else:
            exit("Error: Getdataset ")


    def CifarDataSetConstruct(self, args):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomHorizontalFlip()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_data = datasets.CIFAR10('../data', train=True, download=False, transform=train_transform)
        test_data = datasets.CIFAR10('../data', train=True, download=False, transform=test_transform)

        self.test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=True)

        self.train_size = self.train_data.data.shape[0]
        self.test_size = test_data.data.shape[0]

        if args.iid:
            train_split_dict = cifar_iid(self.train_data, args.num_clients)
            self.train_dict = train_split_dict


    def MnistDataSetConstruct(self, args):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        self.train_data = datasets.MNIST('../data', train=True, download=False, transform=train_transform)
        test_data = datasets.MNIST('../data', train=True, download=False, transform=test_transform)

        self.test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=True)

        self.train_size = self.train_data.data.shape[0]
        self.test_size = test_data.data.shape[0]

        if args.iid:
            train_split_dict = mnist_iid(self.train_data, args.num_clients)
            self.train_dict = train_split_dict



    def LDADataSetConstruct(self, args):
        if args.dataset == 'cifar_LDA':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.RandomHorizontalFlip()
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.train_data = datasets.CIFAR10('../data', train=True, download=False, transform=train_transform)
            test_data = datasets.CIFAR10('../data', train=False, download=False, transform=test_transform)

            self.test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=True)
            self.train_size = self.train_data.data.shape[0]
            self.test_size = test_data.data.shape[0]
        elif args.dataset == 'cifar100_LDA':
            train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.train_data = datasets.CIFAR100('../data', train=True, download=True, transform=train_transform)
            test_data = datasets.CIFAR100('../data', train=False, download=True, transform=test_transform)
            self.test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=True)
            self.train_size = self.train_data.data.shape[0]
            self.test_size = test_data.data.shape[0]
        else:
            exit("Error: Getdataset->LDADataSetConstruct")

        if args.dataset == 'cifar_LDA' and args.non_alpha == 1.0:
            train_split_dict = cifar_LDA1(args)    # 100个
            self.train_dict = train_split_dict
        elif args.dataset == 'cifar_LDA' and args.non_alpha == 0.1:
            train_split_dict = cifar_LDA01(args)    # 100个
            self.train_dict = train_split_dict
        elif args.dataset == 'cifar_LDA' and args.non_alpha == 0.5:
            train_split_dict = cifar_LDA05(args)    # 100个
            self.train_dict = train_split_dict
        elif args.dataset == 'cifar100_LDA' and args.non_alpha == 0.5:
            train_split_dict = cifar100_LDA05(args)
            self.train_dict = train_split_dict
        elif args.dataset == 'cifar100_LDA' and args.non_alpha == 0.1:
            train_split_dict = cifar100_LDA01(args)
            self.train_dict = train_split_dict

        else:
            exit("Error: LDADataSetConstruct")

    def RateDataSetConstruct(self, args):
        if args.dataset == 'mnist_rate':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])
            ])
            self.train_data = datasets.MNIST('../data', train=True, download=False, transform=train_transform)
            test_data = datasets.MNIST('../data', train=False, download=False, transform=test_transform)

            self.test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=True)
            self.train_size = self.train_data.data.shape[0]
            self.test_size = test_data.data.shape[0]
        elif args.dataset == 'cifar_rate':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.RandomHorizontalFlip()
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.train_data = datasets.CIFAR10('../data', train=True, download=False, transform=train_transform)
            test_data = datasets.CIFAR10('../data', train=False, download=False, transform=test_transform)

            self.test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=True)
            self.train_size = self.train_data.data.shape[0]
            self.test_size = test_data.data.shape[0]
        elif args.dataset == 'cifar100_rate':
            train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.train_data = datasets.CIFAR100('../data', train=True, download=False, transform=train_transform)
            test_data = datasets.CIFAR100('../data', train=False, download=False, transform=test_transform)
            self.test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=True)
            self.train_size = self.train_data.data.shape[0]
            self.test_size = test_data.data.shape[0]

        else:
            exit("Error: Getdataset->RateDataSetConstruct")

        if args.dataset == 'cifar_rate' and args.rate == 0.8:
            train_split_dict = cifar_rate08(args)    # 100个
            self.train_dict = train_split_dict

        elif args.dataset == 'cifar_rate' and args.rate == 0.5:
            train_split_dict = cifar_rate05(args)
            self.train_dict = train_split_dict

        elif args.dataset == 'mnist_rate' and args.rate == 0.8:
            train_split_dict = mnist_rate08(args)
            self.train_dict = train_split_dict

        elif args.dataset == 'mnist_rate' and args.rate == 0.5:
            train_split_dict = mnist_rate05(args)
            self.train_dict = train_split_dict

        elif args.dataset == 'cifar100_rate' and args.rate == 0.8:
            train_split_dict = cifar100_rate08(args)
            self.train_dict = train_split_dict
        elif args.dataset == 'cifar100_rate' and args.rate == 0.5:
            train_split_dict = cifar100_rate05(args)
            self.train_dict = train_split_dict

        else:
            exit("Error: Getdataset")





