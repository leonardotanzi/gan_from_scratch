import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)

    device = "cuda:0"
    image_size = 28
    dz = 28
    d = 128
    epochs = 50
    batch_per_epoch = 200
    batch_size = 64
    nb_batch = 200
    init_lr = 0.0002

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST("..\\data", train=True, download=True, transform=transform)
    test_dataset = MNIST("..\\data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model_gen = Generator(dz, d, image_size).to(device)
    model_dis = Discriminator(d, image_size).to(device)

    optimizer_g = torch.optim.Adam(model_gen.parameters(), lr=init_lr, betas=(0.4, 0.999))
    optimizer_d = torch.optim.Adam(model_dis.parameters(), lr=init_lr, betas=(0.4, 0.999))

    for epoch in range(epochs):

        running_loss_d = 0.0
        running_loss_g = 0.0
        num_batches = 0

        print(f"Epoch number {epoch+1}")

        for i, (images, _) in enumerate(train_loader):

            print(f"Step {i} / {len(train_loader)}")

            images = images.to(device)
            # labels = labels.to(device)

            z = torch.rand(batch_size, dz).to(device)

            p_one = torch.ones(batch_size, 1).to(device)
            p_zero = torch.zeros(batch_size, 1).to(device)

            # GENERATOR
            optimizer_g.zero_grad()
            output_gen = model_gen(z)
            # plot = output_gen[0]
            # imshow(plot)
            output_dis_fake = model_dis(output_gen)
            loss_gen = F.binary_cross_entropy(output_dis_fake, p_one)
            loss = loss_gen
            loss_g = loss.detach().item()
            loss.backward(retain_graph=True)
            optimizer_g.step()

            # DISCRIMINATOR
            optimizer_d.zero_grad()
            output_gen = model_gen(z)
            output_dis_fake = model_dis(output_gen)
            loss_dis_fake = F.binary_cross_entropy(output_dis_fake, p_zero)
            output_dis_real = model_dis(images)
            if i + 1 == len(train_loader):  # to handle last batch that is not 64
                p_one = torch.ones(len(images), 1).to(device)
            loss_dis_real = F.binary_cross_entropy(output_dis_real, p_one)
            loss = loss_dis_real + loss_dis_fake
            loss_d = loss.detach().item()
            loss.backward()
            optimizer_d.step()

            # COMPUTE STATS
            running_loss_d += loss_d
            running_loss_g += loss_g
            num_batches += 1

            # AVERAGE STATS THEN DISPLAY
        total_loss_d = running_loss_d / num_batches
        total_loss_g = running_loss_g / num_batches
        print('epoch=', epoch, '\t lr=', init_lr, '\t loss_d=', total_loss_d, '\t loss_g=', total_loss_g)
        plt.imshow(output_gen.view(batch_size, image_size, image_size).detach().cpu()[0, :, :], cmap='gray')
        plt.show()