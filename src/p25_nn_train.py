import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from p25_model import *
from torch.utils.data import DataLoader

# 准备数据集

train_data = torchvision.datasets.CIFAR10(root="../CIFAR10_dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../CIFAR10_dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("Length of train data:{}".format(train_data_size))
print("Length of test data:{}".format(test_data_size))

# 利用dataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
myNetWork = MyNetWork()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(myNetWork.parameters(), learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 100


# 添加tensorboard
writer = SummaryWriter("../logs_train")

# 优化器
for i in range(epoch):
    print("------------The {}th epoch------------".format(i+1))

    # 训练步骤开始
    myNetWork.train()
    for data in train_dataloader:
        img, target = data
        output = myNetWork(img)
        loss = loss_fn(output, target)
        # 优化器调优模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if (total_train_step % 100 == 0):
            print("total train step:{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    # 测试步骤开始
    myNetWork.eval()
    total_test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, target = data
            output = myNetWork(img)
            loss = loss_fn(output, target)
            total_test_loss = total_test_loss + loss + loss.item()
            correct = (output.argmax(1) == target).sum()
            total_correct += correct

        print("Total test loss:{}".format(total_test_loss))
        print("Total test accuracy:{}".format(total_correct/test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_correct/test_data_size, total_test_step)
        total_test_step += 1

    torch.save(myNetWork, "myNetwork_{}.pth".format(i))
    # torch.save(myNetWork.state_dict(), "myNetwork_{}.pth".format(i)):
    print("Models saved")

writer.close()

