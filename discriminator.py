import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, d, n):
        super(Discriminator, self).__init__()
        self.d = d
        self.n = n
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=d, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(d)
        self.fc1 = nn.Linear(d*(n//4)**2, d)
        self.bn3 = nn.BatchNorm1d(d)
        self.fc2 = nn.Linear(d, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = x.view(-1, self.d*((self.n//4)**2))
        x = self.fc1(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = torch.sigmoid(self.fc2(x))
        return x

