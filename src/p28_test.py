import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

img_path = "../images/airplane.png"

img = Image.open(img_path)
print(img)
img = img.convert("RGB")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

img = transform(img)
print(img.shape)

writer = SummaryWriter("../test_logs")
writer.add_image("test_image", img)

# 创建网络模型
# 搭建神经网络
class MyNetWork(nn.Module):

    def __init__(self):
        super(MyNetWork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("myNetwork_57.pth", map_location=torch.device('cpu'))
print(model)

img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)

print(output.argmax(1))