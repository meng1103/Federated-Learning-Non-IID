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

