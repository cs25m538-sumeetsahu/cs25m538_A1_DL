#########################################################################
### CS688W: Systems for Deep Learning
### Department of Computer Science and Engineering
### IIT Madras, Chennai, Tamil Nadu
### Assignment-1: Exploring VGG6 on CIFAR-10 with different configuration
### Roll Num: CS25M538
#########################################################################

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size):
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Load data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    return trainloader, testloader
