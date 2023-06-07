import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from models.Nets import *
from models.ResNet import *
from models.MobileNetV2 import *

from models.Getdataset import GetDataSet
from torch.nn import DataParallel
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class ClientUpdate(object):
    def __init__(self, args):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().to(device)

    def gpus(self, model):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        return model

    def get_model(self, ):
        global_net, clients_net = None, None
        if self.args.model == 'cnn' and (self.args.dataset == 'cifar_LDA' or self.args.dataset == 'cifar_rate'):
            print('-----model:{}   dataset:{}'.format(self.args.model, self.args.dataset))
            global_net = CNNCifar().to(device)
            clients_net = [CNNCifar().to(device) for _ in range(self.args.num_clients)]
        elif self.args.model == 'lenet' and (self.args.dataset == 'cifar_LDA' or self.args.dataset == 'cifar_rate'):
            print('-----model:{}   dataset:{}'.format(self.args.model, self.args.dataset))
            global_net = LeNet5().to(device)
            clients_net = [LeNet5().to(device) for _ in range(self.args.num_clients)]
        elif self.args.model == 'cnn' and (self.args.dataset == 'mnist_LDA' or self.args.dataset == 'mnist_rate'):
            print('-----model:{}   dataset:{}'.format(self.args.model, self.args.dataset))
            global_net = CNNMnist().to(device)
            clients_net = [CNNMnist().to(device) for _ in range(self.args.num_clients)]
        elif self.args.model == 'mobilenet' and (self.args.dataset == 'cifar100_rate' or self.args.dataset == 'cifar100_LDA'):
            print('-----model:{}   dataset:{}'.format(self.args.model, self.args.dataset))
            global_net = MobileNetV2(num_classes=self.args.num_classes).to(device)
            clients_net = [MobileNetV2(num_classes=self.args.num_classes).to(device) for _ in range(self.args.num_clients)]
        else:
            exit("Error: Clients -- get_model() --- no model")


        for model in clients_net:
            model.load_state_dict(global_net.state_dict())

        optimizer = [torch.optim.Adam(model.parameters(), lr=self.args.lr) for model in clients_net]
        return global_net, clients_net, optimizer

    def get_moon_model(self, ):
        global_net, clients_net = None, None
        if self.args.model == 'cnn' and self.args.dataset == 'cifar_LDA':
            print('-----model:{}   dataset:{}'.format(self.args.model, self.args.dataset))
            global_net = MOONCNNCifar().to(device)
            clients_net = [MOONCNNCifar().to(device) for _ in range(self.args.num_clients)]
        elif self.args.model == 'cnn' and self.args.dataset == 'cifar_rate':
            print('-----model:{}   dataset:{}'.format(self.args.model, self.args.dataset))
            global_net = MOONCNNCifar().to(device)
            clients_net = [MOONCNNCifar().to(device) for _ in range(self.args.num_clients)]
        elif self.args.model == 'lenet' and (self.args.dataset == 'cifar_rate' or self.args.dataset == 'cifar_LDA'):
            print('-----model:{}   dataset:{}'.format(self.args.model, self.args.dataset))
            global_net = MOONLeNet().to(device)
            clients_net = [MOONLeNet().to(device) for _ in range(self.args.num_clients)]
        elif self.args.model == 'mobilenet' and self.args.dataset == 'cifar100_LDA':
            print('-----model:{}   dataset:{}'.format(self.args.model, self.args.dataset))
            global_net = MOONMobileNetV2().to(device)
            clients_net = [MOONMobileNetV2().to(device) for _ in range(self.args.num_clients)]
        else:
            exit("Error: Clients -- get_moon_model -- no model")


        for model in clients_net:
            model.load_state_dict(global_net.state_dict())

        optimizer = [torch.optim.Adam(model.parameters(), lr=self.args.lr) for model in clients_net]
        return global_net, clients_net, optimizer

    def client_train_fedavg(self, clients_net, optimizer, train_loader):
        clients_net.train()
        epoch_loss = []
        for i in range(self.args.train_ep):
            batch_loss = []
            for batch_index, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                output = clients_net(data)
                optimizer.zero_grad()
                # 防止出现batch为1的情况，此时少个维度
                if len(output.shape) == 1:
                    output = torch.unsqueeze(output, dim=0)
                loss = self.loss_func(output, target)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return sum(epoch_loss)/len(epoch_loss)

    def client_train_scaffold(self, client_net, global_net, optimizer, train_loader, control_local, control_global):
        client_net.to(device)
        global_weights = global_net.state_dict()
        client_net.train()
        epoch_loss = []
        learn_rate = self.args.lr

        # Set optimizer for the local updates

        control_global_w = control_global.state_dict()
        control_local_w = control_local.state_dict()
        count = 0
        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                client_net.zero_grad()
                log_probs = client_net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                local_weights = client_net.state_dict()
                for w in local_weights:
                    #line 10 in algo
                    local_weights[w] = local_weights[w] - self.args.lr*(control_global_w[w]-control_local_w[w])

                # update local model params
                client_net.load_state_dict(local_weights)

                count += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        new_control_local_w = control_local.state_dict()
        control_delta = copy.deepcopy(control_local_w)
        model_weights = client_net.state_dict()
        local_delta = copy.deepcopy(model_weights)
        for w in model_weights:
            # line 12 in algo
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (
                        global_weights[w] - model_weights[w]) / (count * self.args.lr)
            # line 13
            control_delta[w] = new_control_local_w[w] - control_local_w[w]
            local_delta[w] -= global_weights[w]
        # update new control_local model
        # control_local.load_state_dict(new_control_local_w)
        return sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, new_control_local_w

    def client_train_FedDyn(self, clients_net, global_net, optimizer, train_loader, prev_grads):
        clients_net.train()
        epoch_loss = []

        par_flat = None         # theta t-1
        for name, param in global_net.named_parameters():
            if not isinstance(par_flat, torch.Tensor):
                par_flat = param.view(-1)
            else:
                par_flat = torch.cat((par_flat, param.view(-1)), dim=0)

        for i in range(self.args.train_ep):
            batch_loss = []
            for batch_index, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                y = clients_net(data)
                optimizer.zero_grad()

                # epoch_loss = {}
                loss_a = self.loss_func(y, target)
                # epoch_loss['task loss'] = loss.item()
                # === Dynamic regularization === #

                curr_params = torch.cat([p.reshape(-1) for p in clients_net.parameters()])
                lin_penalty = torch.sum(curr_params * prev_grads)

                norm_penalty = (self.args.feddyn_alpha / 2.0) * torch.linalg.norm(curr_params - par_flat, 2) ** 2

                loss_b = loss_a - lin_penalty + norm_penalty
                loss = loss_b
                # epoch_loss['Quad Penalty'] = quad_penalty.item()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(parameters=clients_net.parameters(), max_norm=10)

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        cur_flat = torch.cat([p.detach().reshape(-1) for p in clients_net.parameters()])
        prev_grads -= self.args.feddyn_alpha * (cur_flat - par_flat)    # ht
        return sum(epoch_loss) / len(epoch_loss), prev_grads


    def client_train_FedProx(self, clients_net, global_net, optimizer, train_loader):
        clients_net.train()
        epoch_loss = []
        for i in range(self.args.train_ep):
            batch_loss = []
            for batch_index, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                output = clients_net(data)
                optimizer.zero_grad()

                proximal_term = 0.0
                for w, w_t in zip(clients_net.parameters(), global_net.parameters()):
                    proximal_term += (w - w_t).norm(2)

                loss = self.loss_func(output, target) + (self.args.mu / 2) * proximal_term
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return sum(epoch_loss)/len(epoch_loss)


    def client_train_moon(self, clients_net, global_net, optimizer, train_loader, previous_net):
        clients_net.train()
        epoch_loss = []
        cos = torch.nn.CosineSimilarity(dim=-1)

        for i in range(self.args.train_ep):
            epoch_loss_collector = []
            for batch_index, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                data.requires_grad = False
                target.requires_grad = False
                target = target.long()

                _, pro1, out = clients_net(data)
                _, pro2, _ = global_net(data)
                if len(out.shape) == 1:
                    out = torch.unsqueeze(out, dim=0)
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                previous_net.to(device)
                _, pro3, _ = previous_net(data)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)


                logits /= self.args.temperature
                labels = torch.zeros(data.size(0)).cuda().long()

                loss2 = self.args.moon_mu * self.loss_func(logits, labels)
                loss1 = self.loss_func(out, target)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())

            epoch_loss0 = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss.append(epoch_loss0)

        return sum(epoch_loss) / len(epoch_loss)


    def FedAvg(self, global_net, clients_net):
        globel_dict = global_net.state_dict()
        for k in globel_dict.keys():
            globel_dict[k] = torch.stack([clients_net[i].state_dict()[k].float() for i in range(len(clients_net))], 0).mean(0)
        global_net.load_state_dict(globel_dict)

    def updata_model(self, global_net, clients_net):
        for model in clients_net:
            model.load_state_dict(global_net.state_dict())

    def test(self, global_net, test_loader):
        global_net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for index, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = global_net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.data.max(dim=1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).long().to(device).sum()
        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        return test_loss, accuracy

    def moontest(self, global_net, test_loader):
        global_net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for index, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                _, _, output = global_net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.data.max(dim=1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).long().to(device).sum()
        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        return test_loss, accuracy



