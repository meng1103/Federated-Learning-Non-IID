import copy
import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.Clients import ClientUpdate
from models.Getdataset import GetDataSet
from config import args_parser
import gc
from models.Nets import *
from models.MobileNetV2 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fed_sacffold_train():
    acc_list = []
    train_loss_list = []
    test_loss_list = []
    print("----------fed----------")
    print("dataset：{}   rate:{}   ".format(args.dataset, args.rate))
    global_net, clients_net, optimizer = FL.get_model()

    control_global = copy.deepcopy(global_net).to(device)
    control_weights = control_global.state_dict()

    # model for local control varietes
    if args.dataset == 'mnist_rate' or args.dataset == 'mnist_LDA':
        local_controls = [CNNMnist().to(device) for _ in range(args.num_clients)]
    elif args.dataset == 'cifar_rate' or args.dataset == 'cifar_LDA':
        local_controls = [LeNet5().to(device) for _ in range(args.num_clients)]
    elif args.dataset == 'cifar100_rate' or args.dataset == 'cifar100_LDA':
        local_controls = [MobileNetV2().to(device) for _ in range(args.num_clients)]
    else:
        local_controls = None
    # for k in range(len(clients_net)):
    #     model = copy.deepcopy(clients_net[k])
    #     local_controls.append(model)
    for net in local_controls:
        net.load_state_dict(control_weights)

    delta_c = copy.deepcopy(global_net.state_dict())
    delta_x = copy.deepcopy(global_net.state_dict())

    client_loader = []
    test_loader = None
    for idx in range(args.num_clients):

        client_loader_ = FL.split_data(idx)
        client_loader.append(client_loader_)
        test_loader = getdata.test_loader

    # for i in range(len(client_loader)):
    #     if i == 0:
    #         print(i, '=======================================')
    #         for j, (img, label) in enumerate(client_loader[i]):
    #             print(label)

    for i in range(args.epoch):
        for ci in delta_c:
            delta_c[ci] = 0.0
        for ci in delta_x:
            delta_x[ci] = 0.0
        action_index = np.random.choice(range(args.num_clients), args.num_selected, replace=False)
        print('FL selected action_clients:', action_index)

        train_loss = 0
        for idx in action_index:
            loss, local_delta_c, local_delta, control_local_w = FL.client_train_scaffold(
                clients_net[idx], global_net, optimizer[idx], client_loader[idx], control_local=local_controls[idx], control_global=control_global)
            # print('local_delta', local_delta)
            train_loss += loss
            if i != 0:
                local_controls[idx].load_state_dict(control_local_w)
            # line16
            for w in delta_c:
                if i == 0:
                    delta_x[w] += clients_net[idx].state_dict()[w]
                else:
                    delta_x[w] += local_delta[w]
                    delta_c[w] += local_delta_c[w]
                    # clean
            gc.collect()
            torch.cuda.empty_cache()
            # update the delta C (line 16)
        for w in delta_c:
            delta_c[w] /= args.num_selected
            delta_x[w] /= args.num_selected
        control_global_W = control_global.state_dict()
        global_weights = global_net.state_dict()
        for w in control_global_W:
            # control_global_W[w] += delta_c[w]
            if i == 0:
                global_weights[w] = delta_x[w]
            else:
                global_weights[w] += delta_x[w]
                control_global_W[w] += (args.num_selected / args.num_clients) * delta_c[w]

        control_global.load_state_dict(control_global_W)
        global_net.load_state_dict(global_weights)

        #########scaffold algo complete##################

        FL.updata_model(global_net, clients_net)

        test_loss, accuracy = FL.test(global_net, test_loader)
        train_loss /= args.num_selected
        accuracy = float(accuracy)

        print("Round {:3d}, Testing accuracy:{:.4f}".format(i + 1, accuracy))
        print("Train_loss:{:.5f}, Test_loss:{:.5f}".format(train_loss, test_loss))
        print("-" * 100)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        acc_list.append(accuracy)

        rootpath = './log'
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)
        accfile = open(rootpath + '/SCAFFOLD.dat', 'w')
        for ac in acc_list:
            temp = str(ac)
            accfile.write(temp)
            accfile.write('\n')
        accfile.close()


if __name__ == '__main__':
    random.seed(1)
    torch.manual_seed(1)
    args = args_parser()
    FL = ClientUpdate(args)
    getdata = GetDataSet(args)


    fed_sacffold_train()
