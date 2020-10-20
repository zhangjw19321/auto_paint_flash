import cv2
import cv2 as cv
import threading
import time
import numpy as np
from add_pic import *
import multiprocessing


class CombineFalsh():
    def __init__(self):
        self.backgroud_cap = cv2.VideoCapture("src/backgroud3.mp4")
        self.raw_cloud_frame = cv2.imread("src/backgroud/cloud_contour.png")
        self.cloud_frame = cv2.resize(self.raw_cloud_frame,(int(self.raw_cloud_frame.shape[1]/3),int(self.raw_cloud_frame.shape[1]/3)))
        self.frame = None
        self.frame_lock = multiprocessing.Lock()
        self.add_cap_flag = False
        self.cap_list = []
        self.cap_list.append(self.backgroud_cap)
    def fangshe(self,img,rate = 0.1):
        rows, cols, channels = img.shape
        p1 = np.float32([[0,0], [cols-1,0], [0,rows-1]])
        p2 = np.float32([[0,rows*rate], [cols*(1-rate),rows*rate], [cols*rate,rows*(1-rate)]])
        # p2 = np.float32([[0,rows*0.3], [cols*0.8,rows*0.1], [cols*0.15,rows*0.7]])
        M = cv.getAffineTransform(p1, p2)
        dst = cv.warpAffine(img, M, (cols,rows))
        return dst
    def add_new_cap(self,add_cap):
        time.sleep(0.1)
        for i in range(500):
            print("come here")
            _,object_frame = add_cap.read()
            # self.frame_lock.acquire()
            print("self frame shape: ", self.frame.shape)
            self.frame = add_image_to_background_cloud(object_frame,self.frame,100 + i,200)
            # self.frame = add_image_to_background_cloud(trans_cloud_frame,background,i,0)
            # self.frame_lock.release()
            time.sleep(0.05)
    def add_new_cap_thread(self,new_cap):
        new_thread = threading.Thread(target=self.add_new_cap,args=(new_cap,))
        new_thread.start()
    def update_flash(self,new_cap):
        for i in range(2000):
            for cap in self.cap_list:
                _,frame = cap.read()
                print("i is::",i)
                rate = 0.02 / 100 * i 
                trans_cloud_frame = self.fangshe(self.cloud_frame,rate)
                _,background = self.backgroud_cap.read()
                background = cv2.resize(background,(1920,1080))
                # self.frame_lock.acquire()
                self.frame = add_image_to_background_cloud(trans_cloud_frame,background,i,0) 
                # self.frame_lock.release()
                time.sleep(0.05)  
            else:
                object_frame = 

    def flash_show(self):
        time.sleep(0.1)
        while True:
            # self.frame_lock.acquire()
            cv2.imshow("flow",self.frame)
            cv2.waitKey(200)
            # self.frame_lock.release()
    def show_thread(self):
        show_window_thread = threading.Thread(target=self.flash_show)
        show_window_thread.start() 

    def run(self):
        update_frame_thread = threading.Thread(target=self.update_flash)
        update_frame_thread.start()
        self.show_thread()
if __name__ == "__main__":
    combine = CombineFalsh()
    
    combine.run() 
    # cap = cv2.VideoCapture("src/result.mp4")
    # print("****************")
    # combine.add_new_cap_thread(cap,) 
    # print("%%%%%%%%%%%%%%")  
