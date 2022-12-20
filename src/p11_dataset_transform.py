import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="../CIFAR10_dataset", train=True, transform=dataset_transform,
                                         download=True)
test_set = torchvision.datasets.CIFAR10(root="../CIFAR10_dataset", train=False, transform=dataset_transform,
                                        download=True)

# print(test_set.classes)
# print(test_set[0])
#
# img, target = test_set[0]
# print(target)
# print(test_set.classes[target])
#
# img.show()

print(type(test_set))
print(test_set[0])

writer = SummaryWriter("../p11_logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
