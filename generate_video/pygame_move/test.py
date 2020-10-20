import cv2
import numpy as np
def contour_filter(frame):
        _, contours, _ = cv2.findContours(frame,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_frame = np.zeros(frame.shape, np.uint8)
        for i, contour in enumerate(contours):
            c_area = cv2.contourArea(contour)
            if self.contour_min_area <= c_area <= self.contour_max_area:
                mask = np.zeros(frame.shape, np.uint8)
                cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
                mask = cv2.bitwise_and(frame, mask)
                new_frame = cv2.bitwise_or(new_frame, mask)
        frame = new_frame

        if self.contour_disp_flag:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)
        return frame

def test():
 
    # 以灰度方式读取图像
    img = cv2.imread('renwu.png', cv2.IMREAD_GRAYSCALE)
    mask = img.copy()
    
    # 二值化，100为阈值，小于100的变为255，大于100的变为0
    # 也可以根据自己的要求，改变参数：
    # cv2.THRESH_BINARY
    # cv2.THRESH_BINARY_INV
    # cv2.THRESH_TRUNC
    # cv2.THRESH_TOZERO_INV
    # cv2.THRESH_TOZERO
    _, binaryzation = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 找到所有的轮廓
    _,contours, _ = cv2.findContours(binaryzation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    area = []
    
    # 找到最大的轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    
    # 填充最大的轮廓
    mask = cv2.drawContours(mask, contours, max_idx, (255,0,0), cv2.FILLED)
    
    # 保存填充后的图像
    cv2.imwrite('masked.png', mask)

def test2():
    frame=cv2.imread("renwu.png")
    # setup initial location of window
    while 1:
        frame = cv2.imread(r"0.jpg")
        roi = cv2.selectROI(windowName="roi", img=frame, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi
        print(w,h)
        img = frame[y:y + h, x:x + w]
        cv2.imshow('a', img)
        height, width = img.shape[:2]
        mask = np.zeros(img.shape[:2],np.uint8)
        # 背景模型
        bgdModel = np.zeros((1,65),np.float64)
        # 前景模型
        fgdModel = np.zeros((1,65),np.float64)
        rect = (w//20, h//20, width -w//10, height -  h//10)
        # 使用grabCut算法
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        cv2.imshow('image', img)
        k = cv2.waitKey(3000)
import cv2 as cv
from matplotlib import pyplot as plt
def test3():
    img = cv.imread('renwu.png')
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (0,0,img.shape[0],img.shape[1])
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,50,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    plt.imshow(img),plt.colorbar(),plt.show()


if __name__ == "__main__":
    # frame = cv2.imread("renwu.png")
    f = test3()
    # cv2.imshow("ff.png",frame)