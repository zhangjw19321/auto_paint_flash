import cv2
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('f1.png', cv2.IMREAD_UNCHANGED)
img_gray = 255 - img[:, :, 3]

cv2.imshow("ff",img_gray)
cv2.waitKey(3000)