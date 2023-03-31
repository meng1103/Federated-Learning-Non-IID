import random
import torch

from models.Clients import ClientUpdate
from models.Getdataset import GetDataSet
from config import args_parser
from fedavg import fedavg
from fedprox import fedprox
from MOON import MOON
from scaffold import scaffold
from FedDyn import feddyn


if __name__ == '__main__':
    random.seed(1)
    torch.manual_seed(1)
    args = args_parser()
    FL = ClientUpdate(args)
    getdata = GetDataSet(args)

    if args.FL == 'fedavg':
        fedavg(args, FL, getdata)
    elif args.FL == 'fedprox':
        fedprox(args, FL, getdata)
    elif args.FL == 'feddyn':
        feddyn(args, FL, getdata)
    elif args.FL == 'moon':
        MOON(args, FL, getdata)
    elif args.FL == 'scaffold':
        scaffold(args, FL, getdata)
