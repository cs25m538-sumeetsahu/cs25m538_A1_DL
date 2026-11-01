#########################################################################
### CS688W: Systems for Deep Learning
### Department of Computer Science and Engineering
### IIT Madras, Chennai, Tamil Nadu
### Assignment-1: Exploring VGG6 on CIFAR-10 with different configuration
### Roll Num: CS25M538
#########################################################################

import torch.nn as nn

class VGG6(nn.Module):
    def __init__(self, activation_fn):
        super(VGG6, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            activation_fn(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
