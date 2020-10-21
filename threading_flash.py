import cv2
import os
from add_pic import *
from extract_contour_by_tools import *
from generate_flash import *
# need to kenow the nums of video and the video path
import threading
from time import sleep
import random
class Flash:
    def __init__(self):
        self._runing = True
        self.cloud_frame = cv.imread("src/backgroud/cloud_contour.png")
        self.cloud_frame = cv2.resize(self.cloud_frame,(int(self.cloud_frame.shape[1]/6),int(self.cloud_frame.shape[1]/6)))
    def terminate(self):
        self._runing = False
    def generate_raw_flash(self,video_path):
        while True:
            video_files = os.listdir(video_path)
            video_files.sort()
            print("video files is; ",video_files)
            # cv2.namedWindow("image",cv2.WINDOW_NORMAL)
            if len(video_files) == 1:
                cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                i = 0
                while self._runing:
                    print("in thread 1 loop")
                    i += 1
                    try:
                        _,frame = cap.read()
                        if frame is None:
                            break
                        rate = 0.02 / 100 * i 
                        trans_cloud_frame = fangshe(self.cloud_frame,rate)
                        frame = add_image_to_background_cloud(trans_cloud_frame,frame,i,0)
                        cv2.imwrite("file.png",frame)
                        sleep(0.05)
                        current_file_num = len(os.listdir(video_path))
                        if current_file_num != 1:
                            break
                        if i > 1000:
                            i =  0
                    except:
                        continue
                    
            if len(video_files) == 2:
                cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
                i = 0
                while self._runing:
                    i += 1
                    if i > 1000:
                        i = 0
                    print("in thread 2 loop")
                    try:
                        b_ret,background = cap.read()
                        if not b_ret:
                            cap.release()
                            cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                            b_ret,background = cap.read()
                        o1_ret,object_frame1 = cap1.read()
                        if not o1_ret:
                            cap1.release()
                            cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
                            o1_ret,object_frame1 = cap1.read()
                        background = cv2.resize(background,(1920,1080))
                        
                        rate = 0.02 / 100 * i 
                        trans_cloud_frame = fangshe(self.cloud_frame,rate)
                        print("^^^^^ : ",video_files[1])
                        frame = add_image_to_background_cloud(trans_cloud_frame,background,i,0)
                        total_frame = add_image_to_background_cloud(object_frame1,frame,200 + i,400)
                        cv2.imwrite("file.png",total_frame)
                        sleep(0.05)
                        current_file_num = len(os.listdir(video_path))
                        if current_file_num != 2:
                            break
                    except:
                        continue
                    # cv2.imshow("frame1",total_frame)
                    # cv2.waitKey(50)
            if len(video_files) == 3:
                cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
                cap2 = cv2.VideoCapture(os.path.join(video_path,video_files[2]))
                i = 0
                while self._runing:
                    print("in thread 3 loop")
                    i += 1
                    if i > 1000:
                        i = 0
                    try:
                        b_ret,background = cap.read()
                        if not b_ret:
                            cap.release()
                            cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                            b_ret,background = cap.read()
                        background = cv2.resize(background,(1920,1080))
                        
                        rate = 0.02 / 100 * i 
                        trans_cloud_frame = fangshe(self.cloud_frame,rate)
                        background = add_image_to_background_cloud(trans_cloud_frame,background,i,0)

                        o1_ret,object_frame1 = cap1.read()
                        
                        if not o1_ret:
                            cap1.release()
                            cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
                            o1_ret,object_frame1 = cap1.read()
                        total_frame = add_image_to_background_cloud(object_frame1,background,200 + i,400)
                        o2_ret,object_frame2 = cap2.read()
                        if not o2_ret:
                            cap2.release()
                            cap2 = cv2.VideoCapture(os.path.join(video_path,video_files[2]))
                            o2_ret,object_frame2 = cap2.read()
                        total_frame = add_image_to_background_cloud(object_frame2,total_frame,300 + i,550)
                        cv2.imwrite("file.png",total_frame)
                        sleep(0.05)
                        current_file_num = len(os.listdir(video_path))
                        if current_file_num != 3:
                            break
                    except:
                        continue
                    # cv2.imshow("image",total_frame)
                    # cv2.waitKey(50)
            if len(video_files) == 4:
                cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
                cap2 = cv2.VideoCapture(os.path.join(video_path,video_files[2]))
                cap3 = cv2.VideoCapture(os.path.join(video_path,video_files[3]))
                i = 0
                while True:
                    i += 1
                    print("in thread 4 loop")
                    if i > 1000:
                        i = 0
                    try:
                        b_ret,background = cap.read()
                        if not b_ret:
                            cap.release()
                            cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                            b_ret,background = cap.read()
                        background = cv2.resize(background,(1920,1080))
                        rate = 0.02 / 100 * i 
                        trans_cloud_frame = fangshe(self.cloud_frame,rate)
                        background = add_image_to_background_cloud(trans_cloud_frame,background,i,0)
                        o1_ret,object_frame1 = cap1.read()
                        if not o1_ret:
                            cap1.release()
                            cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
                            o1_ret,object_frame1 = cap1.read()
                        total_frame = add_image_to_background_cloud(object_frame1,background,100 + i,100)
                        o2_ret,object_frame2 = cap2.read()
                        if not o2_ret:
                            cap2.release()
                            cap2 = cv2.VideoCapture(os.path.join(video_path,video_files[2]))
                            o2_ret,object_frame2 = cap2.read()
                        total_frame = add_image_to_background_cloud(object_frame2,total_frame,300 + i,350)
                        o3_ret,object_frame3 = cap3.read()
                        if not o3_ret:
                            cap3.release()
                            cap3 = cv2.VideoCapture(os.path.join(video_path,video_files[3]))
                            o3_ret,object_frame3 = cap3.read()
                        total_frame = add_image_to_background_cloud(object_frame3,total_frame,200 + i,500)
                        cv2.imwrite("file.png",total_frame)
                        sleep(0.05)
                        current_file_num = len(os.listdir(video_path))
                        if current_file_num != 4:
                            break
                        # print("into 4 files")
                        sleep(0.05)
                    except:
                        continue
                    # cvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv v2.imshow("image",f)
                    # cv2.waitKey(50)
            if len(video_files) == 5:
                cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
                cap2 = cv2.VideoCapture(os.path.join(video_path,video_files[2]))
                cap3 = cv2.VideoCapture(os.path.join(video_path,video_files[3]))
                cap4 = cv2.VideoCapture(os.path.join(video_path,video_files[4]))
                i = 0
                while True:
                    i += 1
                    print("in thread 4 loop")
                    if i > 1000:
                        i = 0
                    try:
                        b_ret,background = cap.read()
                        if not b_ret:
                            cap.release()
                            cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                            b_ret,background = cap.read()
                        background = cv2.resize(background,(1920,1080))
                        rate = 0.02 / 100 * i 
                        trans_cloud_frame = fangshe(self.cloud_frame,rate)
                        background = add_image_to_background_cloud(trans_cloud_frame,background,i,0)
                        o1_ret,object_frame1 = cap1.read()
                        if not o1_ret:
                            cap1.release()
                            cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
                            o1_ret,object_frame1 = cap1.read()
                        total_frame = add_image_to_background_cloud(object_frame1,background,100 + i,100)
                        o2_ret,object_frame2 = cap2.read()
                        if not o2_ret:
                            cap2.release()
                            cap2 = cv2.VideoCapture(os.path.join(video_path,video_files[2]))
                            o2_ret,object_frame2 = cap2.read()
                        total_frame = add_image_to_background_cloud(object_frame2,total_frame,200 + 2*i,350)
                        o3_ret,object_frame3 = cap3.read()
                        if not o3_ret:
                            cap3.release()
                            cap3 = cv2.VideoCapture(os.path.join(video_path,video_files[3]))
                            o3_ret,object_frame3 = cap3.read()
                        total_frame = add_image_to_background_cloud(object_frame3,total_frame,300 + i,500)
                        o4_ret,object_frame4 = cap4.read()
                        if not o4_ret:
                            cap4.release()
                            cap4 = cv2.VideoCapture(os.path.join(video_path,video_files[4]))
                            o4_ret,object_frame4 = cap4.read()
                        total_frame = add_image_to_background_cloud(object_frame4,total_frame,400 + i,650)
                        cv2.imwrite("file.png",total_frame)
                        sleep(0.05)
                        current_file_num = len(os.listdir(video_path))
                        if current_file_num != 5:
                            break
                        # print("into 4 files")
                        sleep(0.05)
                    except:
                        continue
            if len(video_files) == 6:
                cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
                cap2 = cv2.VideoCapture(os.path.join(video_path,video_files[2]))
                cap3 = cv2.VideoCapture(os.path.join(video_path,video_files[3]))
                cap4 = cv2.VideoCapture(os.path.join(video_path,video_files[4]))
                cap5 = cv2.VideoCapture(os.path.join(video_path,video_files[4]))
                i = 0
                while True:
                    i += 1
                    print("in thread 4 loop")
                    if i > 1000:
                        i = 0
                    try:
                        b_ret,background = cap.read()
                        if not b_ret:
                            cap.release()
                            cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
                            b_ret,background = cap.read()
                        background = cv2.resize(background,(1920,1080))
                        rate = 0.02 / 100 * i 
                        trans_cloud_frame = fangshe(self.cloud_frame,rate)
                        background = add_image_to_background_cloud(trans_cloud_frame,background,i,0)
                        o1_ret,object_frame1 = cap1.read()
                        if not o1_ret:
                            cap1.release()
                            cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
                            o1_ret,object_frame1 = cap1.read()
                        total_frame = add_image_to_background_cloud(object_frame1,background,100 + i,100)
                        o2_ret,object_frame2 = cap2.read()
                        if not o2_ret:
                            cap2.release()
                            cap2 = cv2.VideoCapture(os.path.join(video_path,video_files[2]))
                            o2_ret,object_frame2 = cap2.read()
                        total_frame = add_image_to_background_cloud(object_frame2,total_frame,200 + i,350)
                        o3_ret,object_frame3 = cap3.read()
                        if not o3_ret:
                            cap3.release()
                            cap3 = cv2.VideoCapture(os.path.join(video_path,video_files[3]))
                            o3_ret,object_frame3 = cap3.read()
                        total_frame = add_image_to_background_cloud(object_frame3,total_frame,300 + i,500)
                        o4_ret,object_frame4 = cap4.read()
                        if not o4_ret:
                            cap4.release()
                            cap4 = cv2.VideoCapture(os.path.join(video_path,video_files[4]))
                            o4_ret,object_frame4 = cap4.read()
                        total_frame = add_image_to_background_cloud(object_frame4,total_frame,150 + i,650)
                        o5_ret,object_frame5 = cap5.read()
                        if not o5_ret:
                            cap5.release()
                            cap5 = cv2.VideoCapture(os.path.join(video_path,video_files[5]))
                            o5_ret,object_frame5 = cap5.read()
                        total_frame = add_image_to_background_cloud(object_frame5,total_frame,60 + i,800)
                        cv2.imwrite("file.png",total_frame)
                        sleep(0.05)
                        current_file_num = len(os.listdir(video_path))
                        if current_file_num != 6:
                            break
                        # print("into 4 files")
                        sleep(0.05)
                    except:
                        continue

if __name__ == "__main__":
    video_path = "src/temp/test_flash"    
    # generate_raw_flash(video_path)
    f = Flash()
    thread1 = threading.Thread(target=f.generate_raw_flash,args=(video_path,))
    thread1.start()


'''
    import time
    time.sleep(15)
    print("&&&&&&&&&&&&10s ready for second picture")
    f.terminate()
    f1 = Flash()
    print("*****************************************")
    thread2 = threading.Thread(target=f1.generate_raw_flash,args=(video_path,))
    thread2.start()
'''