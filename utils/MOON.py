import copy
import os.path
import numpy as np


def MOON(args, FL, getdata):
    acc_list = []
    train_loss_list = []
    test_loss_list = []
    print("----------fed----------")
    print("dataset：{}   rate:{}   ".format(args.dataset, args.rate))
    global_net, clients_net, optimizer = FL.get_moon_model()

    previous_net = copy.deepcopy(clients_net)
    for i in range(len(previous_net)):
        previous_net[i].eval()
        for param in previous_net[i].parameters():
            param.requires_grad = False

    client_loader = []
    test_loader = None
    for idx in range(args.num_clients):

        client_loader_ = FL.split_data(idx)
        client_loader.append(client_loader_)
        test_loader = getdata.test_loader


    for i in range(args.epoch):
        action_index = np.random.choice(range(args.num_clients), args.num_selected, replace=False)
        print('FL selected  action_clients:', action_index)

        train_loss = 0
        for idx in action_index:
            loss = FL.client_train_moon(clients_net[idx], global_net, optimizer[idx], client_loader[idx],
                                        previous_net[idx])
            train_loss += loss
        FL.FedAvg(global_net, [clients_net[idx] for idx in action_index])

        for idx in action_index:
            previous_net[idx].load_state_dict(clients_net[idx].state_dict())
            # previous_net[idx].parameters().requires_grad = False

        FL.updata_model(global_net, clients_net)

        test_loss, accuracy = FL.moontest(global_net, test_loader)
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
        accfile = open(rootpath + '/MOON.dat', 'w')
        for ac in acc_list:
            temp = str(ac)
            accfile.write(temp)
            accfile.write('\n')
        accfile.close()




