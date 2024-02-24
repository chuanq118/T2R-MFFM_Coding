import cv2
import numpy as np

img = cv2.imread('img/region_demo_plane.jpg')
bg = np.ones_like(img)*255
final = np.where(img==0,bg,img)

cv2.imwrite('img/region_demo_plane_white_bg.jpg',final)