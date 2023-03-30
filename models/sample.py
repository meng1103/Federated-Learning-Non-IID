import numpy as np
import torch
from torchvision import transforms, datasets

def openSampleFile(filepath):
    with open(filepath, 'r') as f:
        dict_user = {}
        index = 0
        while True:
            line = f.readline()
            if line.rstrip('\n') == '':
                break
            temp = []
            line = line[0:len(line)-2]
            line = line.split(',')
            for cur in line:
                temp.append(int(cur))
            dict_user[index] = set(temp)
            index += 1
            if not line:
                break
        return dict_user


def cifar_iid(dataset, num_clients):
    filepath = '..data/cifar_iid_100clients.dat'
    try:
        dict_user = openSampleFile(filepath)
    except FileNotFoundError:
        num_item = int(len(dataset) / num_clients)
        dict_user, all_idx = {}, [i for i in range(len(dataset))]
        for i in range(num_clients):
            dict_user[i] = set(np.random.choice(all_idx, num_item, replace=False))
            all_idx = list(set(all_idx) - dict_user[i])
    if dict_user == {}:
        return "Error"
    return dict_user


def mnist_iid(dataset, num_clients):
    filepath = '../data/mnist_iid_100clients.dat'
    try:
        dict_user = openSampleFile(filepath)
    except FileNotFoundError:
        num_item = int(len(dataset) / num_clients)
        dict_user, all_idx = {}, [i for i in range(len(dataset))]
        for i in range(num_clients):
            dict_user[i] = set(np.random.choice(all_idx, num_item, replace=False))
            all_idx = list(set(all_idx) - dict_user[i])
    if dict_user == {}:
        return "Error"
    return dict_user


def cifar_rate08(args):
    filepath = './data/cifar_rate_0.8_100clients.dat'
    try:
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
    if dict_users == {}:
        return "Error"
    return dict_users

def cifar100_rate08(args):
    filepath = './cutdata/cifar100_rate_0.8_100clients.dat'
    try:
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
    if dict_users == {}:
        return "Error"
    return dict_users

def cifar100_rate05(args):
    filepath = ''
    if args.num_clients == 100:
        print('--sample --- cifar100_rate05 --- 100clients')
        filepath = './cutdata/cifar100_rate_0.5_100clients.dat'
    elif args.num_clients == 50:
        print('--sample --- cifar100_rate05 --- 50clients')
        filepath = './cutdata/cifar100_rate_0.5_50clients.dat'
    try:
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
    if dict_users == {}:
        print('文件读取错误')
        return "Error"
    return dict_users


def cifar100_LDA01(args):
    filepath = ''
    if args.num_clients == 100:
        print('--sample --- cifar100_LDA01 --- 100clients')
        filepath = './cutdata/cifar100_LDA_alpha0.1_100clients.dat'
    elif args.num_clients == 50:
        print('--sample --- cifar100_LDA01 --- 50clients')
        filepath = './cutdata/cifar100_LDA_alpha0.1_50clients.dat'
    try:
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
    if dict_users == {}:
        return "Error"
    return dict_users

def cifar100_LDA05(args):
    filepath = ''
    if args.num_clients == 100:
        print('--sample --- cifar100_LDA05 --- 100clients')
        filepath = './cutdata/cifar100_LDA_alpha0.5_100clients.dat'
    elif args.num_clients == 50:
        print('--sample --- cifar100_LDA05 --- 50clients')
        filepath = './cutdata/cifar100_LDA_alpha0.5_50clients.dat'
    try:
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
        exit("Error: sample  cifar100_LDA05")
    if dict_users == {}:
        return "Error"
    return dict_users

def cifar_rate05(args):
    filepath = '../data/cifar_rate_0.5_100clients.dat'
    try:
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
    if dict_users == {}:
        return "Error"
    return dict_users

def mnist_rate08(args):
    filepath = '../data/mnist_rate_0.8_100clients.dat'
    try:
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
    if dict_users == {}:
        return "Error"
    return dict_users

def mnist_rate05(args):
    filepath = '../data/mnist_rate_0.5_100clients.dat'
    try:
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
    if dict_users == {}:
        return "Error"
    return dict_users

def cifar_LDA1(args):
    filepath = '../cutdata/cifar_LDA_alpha1.0_100clients.dat'
    try:
        print('cifar_LDA_alpha1.0_100clients.dat ')
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
    if dict_users == None:
        return "Error"
    return dict_users

def cifar_LDA05(args):
    filepath = '../cutdata/cifar_LDA_alpha0.5_100clients.dat'

    try:
        print('cifar_LDA_alpha0.5_100clients.dat ')
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
    if dict_users == {}:
        return "Error"
    return dict_users

def cifar_LDA01(args):
    filepath = '../cutdata/cifar_LDA_alpha0.1_100clients.dat'
    try:
        print('cifar_LDA_alpha0.1_100clients.dat ')
        dict_users = openSampleFile(filepath)
    except FileNotFoundError:
        dict_users = None
    if dict_users == {}:
        return "Error"
    return dict_users

