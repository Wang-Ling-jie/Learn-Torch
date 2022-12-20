import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataLoader = DataLoader(dataset, batch_size=64)


class MyNetWork(nn.Module):

    def __init__(self):
        super(MyNetWork, self).__init__()

        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
myNetWork = MyNetWork()
optim = torch.optim.SGD(myNetWork.parameters(), lr=0.02)

for epoch in range(20):
    running_loss = 0.0
    for data in dataLoader:
        img, target = data
        output = myNetWork(img)
        result_loss = loss(output, target)
        optim.zero_grad()
        result_loss.backward()  # 计算梯度
        optim.step()  # 更新data
        running_loss += result_loss
    print(running_loss)
