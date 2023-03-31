import os.path
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_prev_grads(model):
    prev_grads = None
    for param in model.parameters():
        if not isinstance(prev_grads, torch.Tensor):
            prev_grads = torch.zeros_like(param.view(-1))
        else:
            prev_grads = torch.cat((prev_grads, torch.zeros_like(param.view(-1))), dim=0)
    return prev_grads

def feddyn(args, FL, getdata):
    acc_list = []
    train_loss_list = []
    test_loss_list = []
    print("----------FedDyn----------")
    print("dataset：{}   rate:{}   ".format(args.dataset, args.rate))
    global_net, clients_net, optimizer = FL.get_model()
    clients_prev_grads = []

    client_loader = []
    test_loader = None
    for idx in range(args.num_clients):

        client_loader_ = FL.split_data(idx)
        client_loader.append(client_loader_)
        test_loader = getdata.test_loader
        clients_prev_grads.append(init_prev_grads(clients_net[idx]))

    h = {
        key: torch.zeros(params.shape, device=device)
        for key, params in global_net.state_dict().items()
    }

    for i in range(args.epoch):
        action_index = np.random.choice(range(args.num_clients), args.num_selected, replace=False)
        print('FL select clients:', action_index)

        train_loss = 0
        for idx in action_index:
            loss, prev_grads = FL.client_train_FedDyn(clients_net[idx], global_net, optimizer[idx], client_loader[idx], clients_prev_grads[idx])
            train_loss += loss
            clients_prev_grads[idx] = prev_grads

        select_client = []
        for k in action_index:
            select_client.append(clients_net[k])
        h = {
            key: prev_h
            - args.feddyn_alpha * 1 / args.num_clients * sum(theta.state_dict()[key] - old_params for theta in select_client)
            for (key, prev_h), old_params in zip(h.items(), global_net.state_dict().values())
        }
        new_parameters = {
            key: (1 / args.num_selected) * sum(theta.state_dict()[key] for theta in select_client)
            for key in global_net.state_dict().keys()
        }
        new_parameters = {
            key: params - (1 / args.feddyn_alpha) * h_params
            for (key, params), h_params in zip(new_parameters.items(), h.values())
        }
        global_net.load_state_dict(new_parameters)


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
        accfile = open(rootpath + '/FedDyn.dat', 'w')
        for ac in acc_list:
            temp = str(ac)
            accfile.write(temp)
            accfile.write('\n')
        accfile.close()


