from PIL import Image
import os
import numpy as np

# 获得所有.png格式的文件
png_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.png')]

images = [Image.open(f) for f in png_files]

# 设置堆叠图像的角度
angle = 10

new_im_width = int((len(images) - 1) * images[0].width * np.tan(np.radians(angle)) + images[0].width)
new_im_height = int(images[0].height)

# 创建新的图片
new_im = Image.new('RGB', (new_im_width, new_im_height), 'white')

# 在新的图片上堆叠每一张图片
for i, im in enumerate(images):
    # 计算新的位置
    new_position = (int(i * im.width * np.tan(np.radians(angle))), 0)

    # 将图片粘贴到新的位置
    new_im.paste(im, new_position)

# 保存新的图片
new_im.save("3D_view.png")
