
import cv2 as cv
import numpy as np
import cv2
# 读入图片

# 转换成灰度图
def find_picture_location(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    rows_index = np.nonzero(thresh)[0]
    cols_index = np.nonzero(thresh)[1] 
    top,bottom = np.min(rows_index),np.max(rows_index)
    left,right = np.min(cols_index),np.max(cols_index)
    return (left,top),(right,bottom)
    # rec = cv2.rectangle(src,(left,top),(right,bottom),(255,0,0),2)
    # cv2.imshow("ff",rec)
    # cv2.waitKey(3000)

def get_red_region(frame):
    # obtain user draw region -- default is blue
    img = frame
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    # need to modify
    low_blue = np.array([100,43,46])
    high_blue = np.array([124,255,255])
    mask = cv2.inRange(hsv,low_blue,high_blue)
    mask_inv = cv2.bitwise_not(mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1]
    # print("cnt is: ",cnt,type(cnt),cnt.shape)
    left = cnt[0][0][0]
    top = cnt[0][0][1]
    right = cnt[2][0][0]
    bottom = cnt[4][0][1]
    # img = cv2.rectangle(img,(left,top),(right,bottom),(255,255,0),2)
    # cv2.drawContours(img,[cnt],0,(0,255,0),3)
    useful_frame_roi = img[top:bottom,left:right]
    cv2.imshow("contours",useful_frame_roi)
    cv2.waitKey(3000)

def get_red_rect(frame,colored_frame):
    frame = np.uint8(frame)
    print(frame.dtype)
    ret,thresh=cv2.threshold(frame,200,255,cv2.THRESH_BINARY)
    contouts,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = contouts
    inner_x,inner_y,inner_w,inner_h = 999,999,999,999
    board = 30
    for i in cnt:
        #坐标赋值
        x,y,w,h = cv2.boundingRect(i)
        if w < inner_w and h < inner_h:
            inner_x,inner_y,inner_w,inner_h = x+board,y+board,w-2*board,h-2*board
    out = cv2.rectangle(colored_frame,(inner_x,inner_y),(inner_x+inner_w,inner_y+inner_h),(0,0,255),2)
    cv2.imshow('out',out)
    cv2.waitKey(0)


if __name__ == "__main__":
    src = cv2.imread('picture.png')
    mask = cv2.imread("test.png",0)
    get_red_rect(mask,src)
    # find_picture_location(src)

    
