import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dz, d, n):
        super(Generator, self).__init__()
        self.d = d
        self.n = n
        self.fc1 = nn.Linear(dz, d, bias=True)
        self.bn1 = nn.BatchNorm1d(d)
        self.fc2 = nn.Linear(d, d*((n//4)**2), bias=True)
        self.bn2 = nn.BatchNorm1d(d*((n//4)**2))
        self.tconv1 = nn.ConvTranspose2d(in_channels=d, out_channels=d, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.tconv2 = nn.ConvTranspose2d(in_channels=d, out_channels=1, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(d)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = x.view((-1, self.d, self.n//4, self.n//4))
        x = self.tconv1(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.tconv2(x)
        x = torch.sigmoid(x)
        return x