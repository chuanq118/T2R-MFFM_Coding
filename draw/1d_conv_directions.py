import numpy as np
import cv2

# 创建192 x 192全黑图像
img = np.zeros((192, 192, 3), np.uint8)
img[:] = [255]


def draw_border_line():
    # 绘制边框线
    img[:, 0:2] = [0]
    img[:, 63:65] = [0]
    img[:, 126:128] = [0]
    img[:, 189:191] = [0]

    img[0:2, :] = [0]
    img[63:65, :] = [0]
    img[126:128, :] = [0]
    img[189:191, :] = [0]


# 绘制 1
img[64:129, :] = [96]
draw_border_line()
cv2.imwrite('1d_conv/1.png', img)

# 绘制 2
img[:] = [255]
img[:, 64:129] = [96]
draw_border_line()
cv2.imwrite('1d_conv/2.png', img)

# 绘制 3
img[:] = [255]
img[64:129, 0:129] = [96]
img[0:64, 128:190] = [96]
draw_border_line()
cv2.imwrite('1d_conv/3.png', img)
