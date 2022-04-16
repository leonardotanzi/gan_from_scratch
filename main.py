import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, dz, d, n):
        super(Generator, self).__init__()
        self.d = d
        self.n = n
        self.fc1 = nn.Linear(dz, d)
        self.bn1 = nn.BatchNorm1d(d)
        self.fc2 = nn.Linear(d, d*(n//4)**2)
        self.bn2 = nn.BatchNorm1d(d*(n//4)**2)
        self.tconv1 = nn.ConvTranspose2d(in_channels=d, out_channels=d, kernel_size=(2, 2), stride=(2, 2))
        # self.tconv1 = nn.ConvTranspose2d(d, d, kernel_size=4, padding=1, stride=2) #  d x 7 x 7 --> d x 14 x 14
        self.tconv2 = nn.ConvTranspose2d(in_channels=d, out_channels=1, kernel_size=(2, 2), stride=(2, 2))
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


class Discriminator(nn.Module):
    def __init__(self, d, n):
        super(Discriminator, self).__init__()
        self.d = d
        self.n = n
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=d, kernel_size=(2, 2), stride=(2, 2))
        # self.conv1 = nn.Conv2d(1, d, kernel_size=4, padding=1, stride=2) #  1 x 28 x 28 --> d x 14 x 14
        self.bn1 = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=(2, 2), stride=(2, 2))
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
        x = x.view(-1, self.d*self.n//4**2)
        x = self.fc1(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = torch.sigmoid(self.fc2(x))
        return x



if __name__ == "__main__":

    torch.device("cuda")
    device = "cuda:0"
    model_gen = Generator(10, 20, 32).to(device)
    x = torch.randn(2, 10).to(device)
    y = model_gen(x)
    model_dis = Discriminator(10, 32).to(device)
    x = torch.randn(2, 1, 32, 32).to(device)
    y = model_dis(x)
