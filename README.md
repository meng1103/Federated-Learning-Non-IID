# Federated-Learning-Non-IID
Federated Learning Algorithm (Pytorch) : FedAvg, FedProx, MOON, SCAFFOLD, FedDyn


# Requirements
pytorch >= 1.6
torchvision >= 0.9.0

## model
CNN, LeNet, MobileNet, ResNet, GoogLeNet

## dataset 
MNIST, CIFAR10, CIFAR100


## Usage
```
python main.py --method fedalign --client_number 16 --thread_number 16 --comm_round 25 --lr 0.01 --epochs 20 --width 0.25 --mu 0.45 --data_dir data/cifar100
```
