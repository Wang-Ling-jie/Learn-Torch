import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../CIFAR10_dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.maxpool1 = MaxPool2d(3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


myNetwork = MyNetwork()

writer = SummaryWriter("../logs_maxpool")
step = 0

for data in dataloader:
    img, tag = data
    writer.add_images("input", img, step)

    output = myNetwork(img)
    writer.add_images("output", output, step)
    step += 1