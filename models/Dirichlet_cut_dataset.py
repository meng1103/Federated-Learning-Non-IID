import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import os
from PIL import Image
from utils.config import args_parser

torch.manual_seed(42)
np.random.seed(42)

def dirichlet_split_noniid(train_labels, alpha, n_clients):

    n_classes = train_labels.max() + 1

    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs



def rate_cut(self):
    num_classes = 10
    print('self.train_size:', self.train_size)
    train_labels = torch.LongTensor(self.train_data.targets)
    print('train_labels=====', train_labels)
    target_labels_split = []
    target_labels = torch.stack([train_labels == i for i in range(num_classes)])
    target_labels_index = []
    rand_labels_index = []
    last_labels_index = set()
    rate = float(1 / num_classes)
    for j in range(num_classes):
        idx = torch.where(target_labels[j])[0]
        target_labels_index.append(idx)



        batch_size = int(self.train_size * rate * args.rate)
        rand_set_index = set(np.random.choice(target_labels_index[j], size=batch_size, replace=False))
        last_set_index = rand_set_index.symmetric_difference(target_labels_index[j].numpy())
        rand_labels_index.append(rand_set_index)
        last_labels_index = last_labels_index.union(last_set_index)


    last_labels_index = list(last_labels_index)
    np.random.shuffle(last_labels_index)
    last_labels_index = torch.LongTensor(last_labels_index)

    last_size = round(self.train_size * rate * (1 - args.rate))

    last_index = torch.split(last_labels_index, last_size)

    for i in range(num_classes):


        rand_labels_index[i] = torch.LongTensor(list(rand_labels_index[i]))
        rand_size = int(self.train_size * args.rate / args.num_clients)
        last_size = round(self.train_size * (1 - args.rate) / args.num_clients)

        rand_labels_split = torch.split(rand_labels_index[i], rand_size)
        last_labels_split = torch.split(last_index[i], last_size)


        # 10 = num_clients / num_classes = 50 / 5
        one_class_num = int(args.num_clients / num_classes)
        for j in range(one_class_num):
            target = torch.cat([rand_labels_split[j], last_labels_split[j]], dim=0)

            target_labels_split.append(target)
            # print('idx', target)



    train_split = [Subset(self.train_data, x) for x in target_labels_split]
    self.train_loader = [DataLoader(x, batch_size=args.train_bs, shuffle=True) for x in train_split]


if __name__ == "__main__":
    args = args_parser()
    N_CLIENTS = args.num_clients

    train_data = datasets.CIFAR10(root="../data", download=False, train=True)
    test_data = datasets.CIFAR10(root="../data", download=False, train=False)
    data_num, num_class = len(train_data), len(train_data.classes)


    train_labels = np.array(train_data.targets)
    print('len--train_labels', len(train_labels))
    print('train_labels', train_labels)


    client_idcs = dirichlet_split_noniid(train_labels, alpha=args.non_alpha, n_clients=N_CLIENTS)

    print('client_idcs', len(client_idcs))
    for i in range(N_CLIENTS):
        print(len(client_idcs[i]))
        print(client_idcs[i])


    plt.figure(figsize=(20, 3))
    plt.hist([train_labels[idc] for idc in client_idcs], stacked=True,
             bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1),
             label=["Client {}".format(i) for i in range(N_CLIENTS)], rwidth=0.5)
    plt.xticks(np.arange(num_class), train_data.classes)

    plt.legend()
    plt.show()

