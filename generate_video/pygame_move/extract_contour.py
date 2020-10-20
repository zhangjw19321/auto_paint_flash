#coding=utf-8
import cv2
import numpy as np  
 
img = cv2.imread("size.png", 0)
# canny detection
img = cv2.GaussianBlur(img,(3,3),0)
canny = cv2.Canny(img, 50, 150)
print("type is: ",type(canny))
location_record = []
# find the contours
ori = cv2.imread("size.png")
b,g,r = cv2.split(ori)
a = np.zeros(b.shape,dtype=b.dtype)
for i in range(canny.shape[0]-2):
    print("*"*10,np.where(canny[i]==255)[0])
    if len(np.where(canny[i]==255)[0]) == 0:
        continue
    t_min,t_max = np.where(canny[i]==255)[0].min(),np.where(canny[i]==255)[0].max()
    print(t_min,t_max)
    a[i][t_min:t_max] = np.ones(t_max-t_min)*255
img_al = cv2.merge((b,g,r,a))
cv2.imwrite("img_alpha.png",img_al)




'''
cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''