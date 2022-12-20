import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class MyNetwork(nn.Module):

    def __init__(self) -> None:
        super(MyNetwork, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

myNetwork = MyNetwork()
print(myNetwork)

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    img, target = data
    output = myNetwork(img)
    # torch.Size([16, 3, 32, 32])
    writer.add_images("input", img, step)

    # torch.Size([16, 6, 30, 30]) -> [xxx, 3, 30, 30]
    # Reshape the output
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step = step + 1
