import cv2
import numpy as np
import copy
from grabcut import *
def add_image_to_background(foreground,background,x=0,y=0):
    img1 = copy.deepcopy(background) 
    img2 = foreground
    # 把logo放在左上角，所以我们只关心这一块区域
    rows, cols = img2.shape[:2]
    roi = img1[:rows, :cols]
    # 创建掩膜
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # 保留除logo外的背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    dst = cv2.add(img1_bg, img2) # 进行融合
    img1[y:y+rows, x:x+cols] = dst # 融合后放在原图上
    return img1
    # cv2.imshow('res',img1)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()


def add_image_to_background_cloud(foreground,background,x=0,y=0):
    img1 = copy.deepcopy(background) 
    img2 = foreground
    # 把logo放在左上角，所以我们只关心这一块区域
    rows, cols = img2.shape[:2]
    roi = img1[y:y+rows, x:x+cols]
    # 创建掩膜
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # 保留除logo外的背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # cv2.imwrite("log.png",img1_bg)
    dst = cv2.add(img1_bg, img2) # 进行融合
    img1[y:y+rows, x:x+cols] = dst # 融合后放在原图上
    return img1

def add_image_to_background_object(foreground,background,x=0,y=0):
    img1 = copy.deepcopy(background) 
    img2 = foreground
    # 把logo放在左上角，所以我们只关心这一块区域
    rows, cols = img2.shape[:2]
    height,width = img1.shape[:2]
    print("info: ",rows,cols,height,width)
    roi = img1[-rows-y:-y, x:x+cols]
    cv2.imwrite("roi.png",roi)
    # 创建掩膜
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("img2gray.png",img2gray)
    ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    cv2.imwrite("mask_inv.png",mask)
    # 保留除logo外的背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    cv2.imwrite("log.png",img1_bg)
    dst = cv2.add(img1_bg, img2) # 进行融合
    img1[-rows-y:-y, x:x+cols] = dst # 融合后放在原图上
    return img1


if __name__ == "__main__":
    # bf = cv2.imread('backgroud.jpeg')  
    # ff = cv2.imread('carton5_contour.png')
    # # ff = cv2.imread('colored_carton5.jpeg')
    # f = add_image_to_background_cloud(ff,bf,100,200)
    # cv2.imshow("ff",f)
    # cv2.waitKey(5000)
    cap = cv2.VideoCapture("taiji5.mp4")
    frame_count = 100
    background = cv2.imread("backgroud.jpeg")
    for i in range(frame_count):
        ret,frame = cap.read()
        if ret:
            contour = extract_contour_grabcut(frame)
            cv2.imshow("contour",contour)
            transparent_frame = add_image_to_background_cloud(contour,background,400,300)
            cv2.imshow("total",transparent_frame) 
            cv2.waitKey(1)