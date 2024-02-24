import cv2
import numpy as np

# 创建一个300x300的黑色图像
img = np.zeros((300,300,3),np.uint8)

# 设置方格的大小
sq_size = 100

# 计算方格的4个顶点坐标
pts = np.array([
    [[0,0],[sq_size,0],[sq_size,sq_size],[0,sq_size]],
    [[sq_size,0],[sq_size*2,0],[sq_size*2,sq_size],[sq_size,sq_size]],
    [[sq_size*2,0],[300,0],[300,sq_size],[sq_size*2,sq_size]],

    [[0,sq_size],[sq_size,sq_size],[sq_size,sq_size*2],[0,sq_size*2]],
    [[sq_size,sq_size],[sq_size*2,sq_size],[sq_size*2,sq_size*2],[sq_size,sq_size*2]],
    [[sq_size*2,sq_size],[300,sq_size],[300,sq_size*2],[sq_size*2,sq_size*2]],

    [[0,sq_size*2],[sq_size,sq_size*2],[sq_size,300],[0,300]],
    [[sq_size,sq_size*2],[sq_size*2,sq_size*2],[sq_size*2,300],[sq_size,300]],
    [[sq_size*2,sq_size*2],[300,sq_size*2],[300,300],[sq_size*2,300]]
])

# 透视变换矩阵
M = cv2.getPerspectiveTransform(np.float32(pts),np.float32(pts))

# 应用透视变换
img = cv2.warpPerspective(img,M,(300,300))

# 绘制网格线
for pt in pts:
    cv2.polylines(img,[pt],True,(255,255,255),2)

# 显示结果
cv2.imshow('Perspective', img)
cv2.waitKey(0)
cv2.destroyAllWindows()