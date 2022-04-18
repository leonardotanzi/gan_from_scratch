import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator


if __name__ == "__main__":

    device = "cuda:0"
    image_size = 28
    dz = 28
    d = 128
    epochs = 50
    batch_per_epoch = 200
    batch_size = 64

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST("..\\data", train=True, download=True, transform=transform)
    test_dataset = MNIST("..\\data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model_gen = Generator(dz, d, image_size).to(device)
    model_dis = Discriminator(d, image_size).to(device)

    init_lr = 0.001
    optimizer_g = torch.optim.Adam(model_gen.parameters(), lr=init_lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(model_dis.parameters(), lr=init_lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        print(f"Epoch number {epoch+1}")
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            z = torch.rand(batch_size, dz).to(device)
            output_gen = model_gen(z)
            output_dis_fake = model_dis(output_gen)
            labels_gen = torch.unsqueeze(torch.cat((torch.ones(batch_size), torch.zeros(images.shape[0]))), 1).to(
                device)
            loss_gen = F.binary_cross_entropy(output_dis_fake, torch.unsqueeze(torch.zeros(batch_size)))
            optimizer_g.zero_grad()
            loss_gen.backward()
            optimizer_g.step()

            # mix_images = torch.cat((images, output_gen), dim=0)
            # labels_dis = torch.unsqueeze(torch.cat((torch.zeros(images.shape[0]), torch.ones(batch_size))), 1).to(device)

            output_dis_real = model_dis(images)
            loss_dis_real = F.binary_cross_entropy(output_dis_real, torch.unsqueeze(torch.ones(images.shape[0])))

            output_dis_fake = model_dis(output_gen)
            loss_dis_fake = F.binary_cross_entropy(output_dis_fake, torch.unsqueeze(torch.zeros(batch_size)))

            optimizer_d.zero_grad()
            loss_dis_real.backward()
            loss_dis_fake.backward()
            optimizer_d.step()

            # loss_gen = nn.BCELoss()
            # loss_gen.backward()
            #


            # loss = (output, labels)
            # loss

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()