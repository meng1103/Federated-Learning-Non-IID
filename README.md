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
python main.py --FL fedavg --train_bs 50 --train_ep 5 --epoch 500 --non_alpha 0.5 --model lenet --dataset cifar_LDA --num_selected 10 --num_clients 100
```
