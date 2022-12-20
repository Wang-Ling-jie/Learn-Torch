from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("../logs")

img = Image.open("../images/img.png")
print(img)

# To Tensor
trans_to_tensor = transforms.ToTensor()
img_tensor = trans_to_tensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor.shape)
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 3, 5], [10, 10, 10])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
print(img_resize.size)
# img_resize_PIL -> ToTensor -> img_resize tensor
img_resize = trans_to_tensor(img_resize)
writer.add_image("Resized", img_resize, 0)

# Composed resize
trans_resize_2 = transforms.Resize(512)
# PIL image -> PIL image -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_to_tensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resized", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((200, 300))
trans_compose_2 = transforms.Compose([trans_random, trans_to_tensor])

for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
