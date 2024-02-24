import cv2

img = cv2.imread(r'D:\dataset\BJRoad\train_val\image\10_100_sat.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('gray.jpg', gray)