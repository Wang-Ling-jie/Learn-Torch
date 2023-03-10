import torchvision.datasets
from torch.utils.data import DataLoader

# 准备好的测试数据集
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("../CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("../dataloader_logs")

# 两轮取数据,两轮之间会打乱图片选取顺序
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step += 1

writer.close()