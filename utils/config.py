import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated argument

    parser.add_argument('--FL', type=str, default='fedavg', help="FL name  fedavg/moon/fedprox/scaffold/feddyn")
    parser.add_argument('--test_bs', type=int, default=64, help="test batch_size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning_rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning_rate_decay")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")


    parser.add_argument('--train_bs', type=int, default=50, help="client batch_size")
    parser.add_argument('--train_ep', type=int, default=5, help="client epoch")

    # other

    parser.add_argument('--epoch', type=int, default=700, help="fl epoch")



    parser.add_argument('--rate', type=float, default=0.5, help="rate")
    parser.add_argument('--non_alpha', type=float, default=0.5, help="LDA")


    parser.add_argument('--model', type=str, default='lenet', help="model name")

    parser.add_argument('--dataset', type=str, default='cifar_LDA', help="dataset name cifar_LDA,cifar_rate,"
                                                                           "cifar100_LDA,cifar100_rate")
    parser.add_argument('--num_classes', type=int, default=10, help="class num")
    parser.add_argument('--num_clients', type=int, default=100, help="client num")
    parser.add_argument('--num_selected', type=int, default=10, help="selected num")



    parser.add_argument('--mu', type=float, default=0.01, help='FedProx ')
    parser.add_argument('--temperature', type=float, default=0.5, help='MOON')
    parser.add_argument('--moon_mu', type=float, default=1, help='MOON ')
    parser.add_argument('--feddyn_alpha', type=float, default=0.01, help='feddyn alpha')

    args = parser.parse_args()
    print('config')
    return args

