import cv2
import numpy as np

import cv2
import pyzbar.pyzbar as pyzbar
import json
import threading
from time import sleep



import numpy as np
import cv2 as cv
import cv2
from add_pic import *
from extract_contour_by_tools import *
from generate_flash import *
from AttentionedDeepPaint.colorize import *
from Animate.animate_api import *
from show_image import *
from threading_flash import *
import multiprocessing

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
        self.extract_paint_frame = None
        # thread
        self.camera_thread = threading.Thread(target=self.scan_qr)
        self.driving_video = "src/driving_video/dance7.mp4"
        self.background_image = "src/backgroud/backgroud.jpeg"
        self.cloud_image = "src/backgroud/cloud_contour.png"
        self.animation_video = "src/result_0.mp4"
        # process video files
        self.flash_cap = []
        self.process_picture_num = 0
    def scan_qr(self):
        while True:
            ret, im = self.cap.read()
            # im = cv2.imread("picture.png")
            cv2.imshow("im",im)
            cv2.waitKey(30)
            # continue
            decodedObjects = pyzbar.decode(im)
            if (len(decodedObjects) != 0) and self.start_recognize_qr_flag:
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
        print("come out get blue region")
        return useful_frame_roi
        # cv2.imshow("contours",useful_frame_roi)
        # cv2.waitKey(3000)
    def resize_user_contour(self):
        frame = cv2.imread("thresh.png")
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY_INV)
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        board = 20
        max_contour_x,max_contour_y,max_contour_w,max_contour_h = 0,0,0,0
        for cidx,cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w * h > max_contour_w * max_contour_h:
                max_contour_x,max_contour_y,max_contour_w,max_contour_h = (x, y, w, h) 
        ori = frame[max_contour_y-board:max_contour_y+max_contour_h+2*board,max_contour_x-board:max_contour_x+max_contour_w+2*board]
        user_contour_frame = cv2.resize(ori,(512,512))
        # cv2.imwrite("usr.png",user_contour_frame)
        return user_contour_frame
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
        self.extract_paint_frame = cv2.bitwise_not(thresh)
        cv2.imwrite("thresh.png",self.extract_paint_frame)
        user_contour_sized_frame = self.resize_user_contour()
        cv2.imwrite("user_sized_contour.png",user_contour_sized_frame)
        print("come out extract picture")
        return user_contour_sized_frame
    def generate_animate_flash(self):
        print("come into generate flash")
        raw_frame = self.extract_paint_frame
        ################# step two: auto paint ################
        colored_frame = paint_color(raw_frame)
        save_colored_name = "src/temp/colored_frame.png"
        cv2.imwrite(save_colored_name,colored_frame)
        print("step one finish")
        ################ step three: extract contour #########
        transparent_coloed_frame = koutu(save_colored_name)
        cv2.imwrite("src/temp/transparent_colored.png",transparent_coloed_frame)
        print("step two finish")
        ############### step four: generate animate ##########
        image = "src/temp/transparent_colored.png"
        model = "taichi"
        save_video_name = "src/temp/test_flash/result_" + str(self.process_picture_num) + ".mp4"
        self.process_picture_num += 1
        if self.process_picture_num > 4:
            self.process_picture_num = 0
        animate(image,self.driving_video,model,save_video_name)
        self.start_recognize_qr_flag = True
        print("come out generate flash")
        ############# step five: generate flash ############
        # flow_picture(self.background_image,self.cloud_image,self.animation_video)
    def run(self):
        while True:
            if self.raw_frame is not None and not self.start_recognize_qr_flag:
                self.extract_picture()
                self.generate_animate_flash()
                print("****flash has benn generated")
                self.start_recognize_qr_flag = True
                
            


                        








if __name__ == "__main__":
    # frame = cv2.imread("src/picture_with_qr.png")
    # p = multiprocessing.Process(target = worker, args = (3,))
    '''
    video_path = "src/temp/test_flash"    
    # generate_raw_flash(video_path)
    f = Flash()
    combine_flash_thread = multiprocessing.Process(target=f.generate_raw_flash,args=(video_path,))
    combine_flash_thread.start()
    print("thread combine flash start")

    show_file_thread = multiprocessing.Process(target=show_image)
    show_file_thread.start()
    print("thread show image start")
    sleep(5)
    '''
    # os.system("python3 threading_flash.py &")
    # os.system("python3 show_image.py &")
    lstm = ConvLSTM()
    lstm.take_picture_thread()
    lstm.run()

