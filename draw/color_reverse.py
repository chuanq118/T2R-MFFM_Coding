import cv2
import os

# 获取当前文件的绝对路径
cur_path = os.path.dirname(os.path.abspath(__file__))

# 拼接得到 mask.png 的全路径
img_path = os.path.join(cur_path, 'img', 'mask.png')

# 使用 OpenCV 读取图片
image = cv2.imread(img_path)

# 对图片颜色进行翻转
image_reverse = cv2.bitwise_not(image)

# 拼接得到 mask_reverse.png 的全路径
img_reverse_path = os.path.join(cur_path, 'img', 'mask_reverse.png')

# 保存反转后的图片
cv2.imwrite(img_reverse_path, image_reverse)