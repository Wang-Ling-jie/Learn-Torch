from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# Python的用法 -》 tensor的数据类型
# 通过Transforms.ToTensor去解决两个问题
# 1. transforms如何使用  (python)
# 2. 为什么需要tensor数据类型

# 绝对路径 "E:\PytorchPrograms\learn_torch\data\train\ants_image\0013035.jpg"
# 相对路径 "data/train/ants_image/0013035.jpg"
img_path = "../data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("../logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

print(tensor_img)
# 绝对路径