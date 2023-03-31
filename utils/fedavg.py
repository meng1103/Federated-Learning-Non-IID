import os.path
import numpy as np


def fedavg(args, FL, getdata):
    acc_list = []
    train_loss_list = []
    test_loss_list = []
    print("----------fed----------")
    print("dataset：{}   rate:{}   ".format(args.dataset, args.rate))
    global_net, clients_net, optimizer = FL.get_model()

    client_loader = []
    test_loader = None
    for idx in range(args.num_clients):

        client_loader_ = FL.split_data(idx)
        client_loader.append(client_loader_)
        test_loader = getdata.test_loader


    for i in range(args.epoch):
        action_index = np.random.choice(range(args.num_clients), args.num_selected, replace=False)
        print('FL  selected  action_clients:', action_index)

        train_loss = 0
        for idx in action_index:
            loss = FL.client_train_fedavg(clients_net[idx], optimizer[idx], client_loader[idx])
            train_loss += loss
        FL.FedAvg(global_net, [clients_net[idx] for idx in action_index])
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
        accfile = open(rootpath + '/Fedavg.dat', 'w')
        for ac in acc_list:
            temp = str(ac)
            accfile.write(temp)
            accfile.write('\n')
        accfile.close()


