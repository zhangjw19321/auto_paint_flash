import cv2
import numpy as np

import cv2
import pyzbar.pyzbar as pyzbar
import json
import threading
from time import sleep

def scan_qr():
    global auto_paint_flag
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    while True:
        ret, im = cap.read()
        # cv2.imshow("im",im)
        # cv2.waitKey(30)
        # continue
        decodedObjects = pyzbar.decode(im)
        if (len(decodedObjects) != 0):
            info = str(decodedObjects[0].data,encoding = "utf-8")
            if info == "auto paint":
                print("start to paint")
                auto_paint_flag = True
        else:
            auto_paint_flag = False

class ConvLSTM():
    def __init__(self):
        self.auto_paint_flag = False
        self.qr_left,self.qr_top,self.qr_width,self.qr_height = None,None,None,None
        self.start_recognize_qr_flag = True
        self.cap = cv2.VideoCapture(0)
        # save camera frame
        self.raw_frame = None
        # thread
        self.camera_thread = threading.Thread(target=self.scan_qr)
    def scan_qr(self):
        while self.start_recognize_qr_flag:
            ret, im = self.cap.read()
            # im = cv2.imread("picture.png")
            cv2.imshow("im",im)
            cv2.waitKey(30)
            # continue
            decodedObjects = pyzbar.decode(im)
            if (len(decodedObjects) != 0):
                info = str(decodedObjects[0].data,encoding = "utf-8")
                if info == "auto paint":
                    print("start to paint")
                    cv2.imwrite("picture.png",im)
                    self.raw_frame = im
                    print("in scan qr raw frame size is: ",self.raw_frame.shape)
                    # self.raw_frame = cv2.imread("picture.png")
                    self.auto_paint_flag = True
                    self.start_recognize_qr_flag = False
                    # extract qr region
                    self.qr_left,self.qr_top,self.qr_width,self.qr_height = decodedObjects[0].rect
            else:
                self.auto_paint_flag = False
    def take_picture_thread(self):
        self.camera_thread.start()
    def get_blue_region(self,frame):
        # obtain user draw region -- default is blue
        print("come into get blue region")
        img = frame
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # need to modify
        low_blue = np.array([100,43,46])
        high_blue = np.array([124,255,255])
        print("come into get blue region2")
        mask = cv2.inRange(hsv,low_blue,high_blue)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) 
        opened1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=1)
        mask = cv2.morphologyEx(opened1, cv2.MORPH_CLOSE, kernel,iterations=1)
        frame = np.uint8(mask)
        print(frame.dtype)
        ret,thresh=cv2.threshold(frame,200,255,cv2.THRESH_BINARY)
        contouts,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnt = contouts
        inner_x,inner_y,inner_w,inner_h = 999,999,999,999
        board = 25
        for i in cnt:
            #坐标赋值
            x,y,w,h = cv2.boundingRect(i)
            if 200 < w < inner_w and 200 < h < inner_h:
                inner_x,inner_y,inner_w,inner_h = x+board,y+board,w-2*board,h-2*board
        out = cv2.rectangle(frame,(inner_x,inner_y),(inner_x+inner_w,inner_y+inner_h),(0,0,255),2)
        # ret,out = cv2.threshold(out,50,255,cv2.THRESH_BINARY)
        cv2.imwrite("out.png",out)
        # cv2.imshow('out',out)
        # cv2.waitKey(0)


        # cv2.imshow("ff",mask)
        # cv2.waitKey(0)
        # mask_inv = cv2.bitwise_not(mask)
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[0]
        # print("cnt is: ",cnt,type(cnt),cnt.shape)
        # print("come into get blue region3")
        # cv2.drawContours(img,[cnt],0,(0,255,0),3)
        # cv2.imshow("ff",img)
        # cv2.waitKey(0)
        useful_region_left = inner_x
        useful_region_top = inner_y
        useful_region_right = inner_x + inner_w
        useful_region_bottom = inner_y + inner_h 
        return useful_region_left,useful_region_top,useful_region_right,useful_region_bottom
        # img = cv2.rectangle(img,(left,top),(right,bottom),(255,255,0),2)
        # cv2.drawContours(img,[cnt],0,(0,255,0),3)
        useful_frame_roi = img[top:bottom,left:right]
        return useful_frame_roi
        # cv2.imshow("contours",useful_frame_roi)
        # cv2.waitKey(3000)
    def extract_picture(self):
        print("come into extract picture")
        print(self.raw_frame is not None and not self.start_recognize_qr_flag)
        qr_padding = 10
        blank = np.ones((self.qr_height + qr_padding,self.qr_width + qr_padding,3),np.uint8) * 255
        print("self qr size is: ", self.qr_height,self.qr_width,self.qr_top,self.qr_left,self.raw_frame.shape)
        self.raw_frame[self.qr_top - int(qr_padding/2):self.qr_top+self.qr_height + int(qr_padding/2),self.qr_left - int(qr_padding/2):self.qr_left+self.qr_width + int(qr_padding/2)] = blank
        user_left,user_top,user_right,user_bottom = self.get_blue_region(self.raw_frame)
        print("user region: ",user_left,user_right,user_top,user_bottom)
        print("self raw frame shape is: ", self.raw_frame.shape)
        self.raw_frame = self.raw_frame[user_top:user_bottom,user_left:user_right]
        # binary to convienient to extract picture
        gray = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2GRAY)
        # 二值化
        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        rows_index = np.nonzero(thresh)[0]
        cols_index = np.nonzero(thresh)[1] 
        top,bottom = np.min(rows_index),np.max(rows_index)
        left,right = np.min(cols_index),np.max(cols_index)
        picture_roi = self.raw_frame[top:bottom,left:right]
        picture_roi_padding = cv2.copyMakeBorder(picture_roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255,255,255))
        sized_piture = cv2.resize(picture_roi_padding,(512,512))
        cv2.imwrite("padding.png",sized_piture)
        frame = cv2.imread("padding.png",0)
        cv2.imwrite("gray.png",frame)
        ret,thresh = cv2.threshold(frame,100,255,cv2.THRESH_BINARY_INV)
        convert = cv2.bitwise_not(thresh)
        cv2.imwrite("thresh.png",convert)
        return convert
    def smart_paint_image(self):


    def run(self):
        while True:
            if self.raw_frame is not None and not self.start_recognize_qr_flag:
                self.extract_picture()
                sleep(10)
                self.start_recognize_qr_flag = True
                sleep(5)
            


                        








if __name__ == "__main__":
    # frame = cv2.imread("src/picture_with_qr.png")
    lstm = ConvLSTM()
    lstm.take_picture_thread()
    lstm.run()

