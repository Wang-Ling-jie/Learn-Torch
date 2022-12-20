import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))

print(input.shape)
dataset = torchvision.datasets.CIFAR10("../CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.relu1(input)
        return output

myNetwork = MyNetwork()

# writer = SummaryWriter("../logs_relu")
#
# step = 0
# for data in dataloader:
#     img, target = data
#     output = myNetwork(img)
#     writer.add_images("input", img, global_step=step)
#     writer.add_images("output", output, global_step=step)
#     step += 1
#
# writer.close()

output = myNetwork(input)
print(output)
