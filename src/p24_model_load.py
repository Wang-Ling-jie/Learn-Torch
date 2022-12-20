import torch
import torchvision.models
from torch import nn

#加载模型
# 方式1（对应保存方式1）
model = torch.load("vgg16_method1.pth")
print(model)

# 方式2（对应保存方式2）
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(model)

# 陷阱
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

model = torch.load("myNetwork_method1.pth")
print(model)
