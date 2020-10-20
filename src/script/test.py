import os
import sys


import cv2
'''
frame = cv2.imread("f3.png")
b = cv2.imread("backgroud.jpeg")
b[:500,:500] = frame
cv2.imshow("fff",b)

cv2.waitKey(20000)
'''
frame = cv2.imread("t_girl.png",-1)
frame = cv2.imread("renwu_zero.png")
# b,g,r,a = cv2.split(frame)
# t = cv2.merge([b,g,r,a])
# cv2.imwrite("tt.png",t)
# f = cv2.imread("tt.png")
cv2.imshow("mianfei",frame)
cv2.waitKey(5000)
