import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../CIFAR10_dataset", transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

myNetwork = MyNetwork()

for data in dataloader:
    img, target = data
    print(img.shape)
    output = torch.flatten(img)
    print(output.shape)
    output = myNetwork(output)
    print(output.shape)