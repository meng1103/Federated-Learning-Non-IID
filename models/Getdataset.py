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

        if self.args.dataset == 'cifar_rate' or self.args.dataset == 'mnist_rate':
            self.RateDataSetConstruct(self.args)
        elif self.args.dataset == 'cifar_LDA' or self.args.dataset == 'cifar100_LDA':
            self.LDADataSetConstruct(self.args)
        elif self.args.dataset == 'cifar100_rate':
            self.RateDataSetConstruct(self.args)

        else:
            exit("Error: Getdataset ")


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
            self.train_data = datasets.CIFAR10('../data', train=True, download=True, transform=train_transform)
            test_data = datasets.CIFAR10('../data', train=True, download=True, transform=test_transform)

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
            train_split_dict = cifar_LDA1(args)
            self.train_dict = train_split_dict
        elif args.dataset == 'cifar_LDA' and args.non_alpha == 0.1:
            train_split_dict = cifar_LDA01(args)
            self.train_dict = train_split_dict
        elif args.dataset == 'cifar_LDA' and args.non_alpha == 0.5:
            train_split_dict = cifar_LDA05(args)
            self.train_dict = train_split_dict
        elif args.dataset == 'cifar100_LDA' and args.non_alpha == 0.5:
            train_split_dict = cifar100_LDA05(args)
            self.train_dict = train_split_dict
        elif args.dataset == 'cifar100_LDA' and args.non_alpha == 0.1:
            train_split_dict = cifar100_LDA01(args)
            self.train_dict = train_split_dict


        else:
            print('未找到对应的分配文件，尝试构造：数据集：{}   LDA：{}'.format(self.args.dataset, self.args.non_alpha))
            train_labels = np.array(self.train_data.targets)
            file_path = self.dirichlet_split_noniid(train_labels, alpha=self.args.non_alpha, n_clients=self.args.num_clients)

            train_split_dict = openSampleFile(file_path)
            self.train_dict = train_split_dict

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
            self.train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
            test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transform)

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
            self.train_data = datasets.CIFAR10('../data', train=True, download=True, transform=train_transform)
            test_data = datasets.CIFAR10('../data', train=False, download=True, transform=test_transform)

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
            self.train_data = datasets.CIFAR100('../data', train=True, download=True, transform=train_transform)
            test_data = datasets.CIFAR100('../data', train=False, download=True, transform=test_transform)
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
            print('Corresponding distribution file not found，structure：datasets：{}   rate比例：{}'.format(self.args.dataset, self.args.rate))
            file_path = self.rate_cut()
            train_split_dict = openSampleFile(file_path)
            self.train_dict = train_split_dict



    def rate_cut(self):
        num_classes = self.args.num_classes
        rate = self.args.rate
        num_clients = self.args.num_clients
        print('num_classes：', num_classes)
        print('rate：', rate)
        print('num_clients：', num_clients)

        num_client_img = int(self.train_size / num_clients)
        num_same_client_img = int(num_client_img * rate)
        num_differ_client_img = int(num_client_img * (1 - rate))

        num_class_img = int(self.train_size / num_classes)


        if self.args.dataset == 'Tiny_rate' or self.args.dataset == 'Tiny_LDA':
            train_labels = []
            for k in range(len(self.train_data)):
                a, b = self.train_data.images[k]
                train_labels.append(b)
        else:
            train_labels = self.train_data.targets
            print('len train_labels=====', len(train_labels))


        idx_sort = np.argsort(train_labels)      
        print(idx_sort)


        class_idx = []
        for i in range(num_classes):
            cla = idx_sort[i * num_class_img: (i+1) * num_class_img]
            class_idx.append(cla)

        mix_img = []    


        clients = []
        for i in range(len(class_idx)):
            if i < num_clients:
                clients.append(class_idx[i][:num_same_client_img])
                for x in class_idx[i][num_same_client_img:]:
                    mix_img.append(x)
                    # print(len(mix_img))
            else:
                for x in class_idx[i]:
                    mix_img.append(x)

        np.random.shuffle(mix_img)

        for i in range(len(clients)):
            temp = mix_img[i * num_differ_client_img: (i+1) * num_differ_client_img]
            for j in temp:
                clients[i] = list(clients[i])
                clients[i].append(j)



        rootpath = '../cutdata'
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)
        file_path = rootpath + '/{}_{}_{}clients.dat'.format(self.args.dataset, self.args.rate, self.args.num_clients)
        accfile = open(file_path, 'w')


        for i in clients:
            temp = [str(k) for k in i]
            pp = ','.join(temp)
            accfile.write(pp)
            accfile.write('\n')
        accfile.close()

        return file_path

    def dirichlet_split_noniid(self, train_labels, alpha, n_clients):

        n_classes = train_labels.max() + 1

        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

        class_idcs = [np.argwhere(train_labels == y).flatten()
                      for y in range(n_classes)]

        client_idcs = [[] for _ in range(n_clients)]   
        for c, fracs in zip(class_idcs, label_distribution):

            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]


        rootpath = '../cutdata'
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)
        file_path = rootpath + '/{}_alpha{}_{}clients.dat'.format(args.dataset, args.non_alpha, args.num_clients)
        accfile = open(file_path, 'w')

        for i in client_idcs:
            temp = [str(k) for k in i]
            pp = ','.join(temp)
            accfile.write(pp)
            accfile.write('\n')
        accfile.close()

        return file_path


if __name__ == '__main__':
    args = args_parser()
    c = GetDataSet(args=args)
    print(c.test_size)
    print(c.train_size)

    print('c.train_dict', c.train_dict)
