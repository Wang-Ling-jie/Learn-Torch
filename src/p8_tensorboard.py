from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("../logs")
img_path = "../data/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

writer.add_image("train", img_array, 1, dataformats='HWC')
# draw y = 2*x
for i in range(100):
    writer.add_scalar("y=3x", 3*i, i)

writer.close()



