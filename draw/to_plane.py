import cv2
import numpy as np

img = cv2.imread('img/region_demo.png')

points1 = np.float32([[0,0],[800,0],[800,800],[0,800]])
points2 = np.float32([[100,100],[700,100],[500,600],[200,600]])

M = cv2.getPerspectiveTransform(points1,points2)

result = cv2.warpPerspective(img,M,(800,800))

cv2.imwrite('img/region_demo_plane.jpg',result)