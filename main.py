import torch
import torch.nn as nn
from torchsummary import summary


class Generator(nn.Module):
    def __init__(self, dz, d, n):
        super(Generator, self).__init__()
        self.d = d
        self.n = n
        self.fc1 = nn.Linear(dz, d)
        self.fc2 = nn.Linear(d, int(d*(n/4)**2))
        self.tconv = nn.ConvTranspose2d(in_channels=d, out_channels=d, kernel_size=(2, 2))

    def forward(self, x):
        x = nn.ReLU(self.fc1(x))
        x = nn.BatchNorm1d(x)
        x = nn.ReLU(self.fc2(x))
        x = nn.BatchNorm1d(x)
        x = x.view((-1, self.d, self.n/4, self.n/4))
        x = nn.ReLU(self.tconv(x))
        x = nn.BatchNorm2d(x)
        x = nn.Sigmoid()(self.tconv(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, d, n):
        super(Discriminator, self).__init__()
        self.d = d
        self.n = n
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=d, padding="same", stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=d, padding="same", stride=2)
        self.fc1 = nn.Linear(d*(n/4)**2, d)
        self.fc2 = nn.Linear(d, 1)

    def forward(self, x):
        x = nn.ReLU(self.conv1(x))
        x = nn.BatchNorm2d(x)
        x = nn.ReLU(self.conv2(x))
        x = nn.BatchNorm2d(x)
        x = nn.Flatten()
        x = nn.ReLU(self.fc1(x))
        x = nn.BatchNorm1d(x)
        x = nn.Sigmoid()(self.fc2(x))
        return x



if __name__ == "__main__":

    torch.device("cuda")
    model = Generator(10, 20, 32)
    print(summary(model, input_size=(1, 10))

