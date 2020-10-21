import cv2
import os
from add_pic import *
from extract_contour_by_tools import *
# need to kenow the nums of video and the video path
import threading
def generate_raw_flash(video_path):
    video_files = os.listdir(video_path)
    video_files.sort()
    if len(video_files) == 1:
        cap = cv2.VideoCapture(video_files[0])
        while True:
            _,frame = cap.read()
            cv2.imshow("frame1",frame)
    if len(video_files) == 2:
        cap = cv2.VideoCapture(video_files[0])
        cap1 = cv2.VideoCapture(video_files[1])
        while True:
            _,background = cap.read()
            background = cv2.resize(background,(1920,1080))
            _,object_frame1 = cap1.read()
            total_frame = add_image_to_background_cloud(object_frame1,background,300 + i,400)
            cv2.imshow("frame1",total_frame)
            cv2.waitKey(50)
    if len(video_files) == 3:
        cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
        cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
        cap2 = cv2.VideoCapture(os.path.join(video_path,video_files[2]))
        i = 0
        while True:
            i += 1
            _,background = cap.read()
            background = cv2.resize(background,(1920,1080))
            _,object_frame1 = cap1.read()
            total_frame = add_image_to_background_cloud(object_frame1,background,300 + i,400)
            _,object_frame2 = cap2.read()
            total_frame = add_image_to_background_cloud(object_frame2,total_frame,300 + i,600)
            cv2.imshow("frame1",total_frame)
            cv2.waitKey(50)
    if len(video_files) == 4:
        cap = cv2.VideoCapture(os.path.join(video_path,video_files[0]))
        cap1 = cv2.VideoCapture(os.path.join(video_path,video_files[1]))
        cap2 = cv2.VideoCapture(os.path.join(video_path,video_files[2]))
        cap3 = cv2.VideoCapture(os.path.join(video_path,video_files[3]))
        i = 0
        while True:
            i += 1
            print("in thread 2 loop")
            _,background = cap.read()
            background = cv2.resize(background,(1920,1080))
            _,object_frame1 = cap1.read()
            total_frame = add_image_to_background_cloud(object_frame1,background,300 + i,100)
            _,object_frame2 = cap2.read()
            total_frame = add_image_to_background_cloud(object_frame2,total_frame,300 + 2*i,350)
            _,object_frame3 = cap3.read()
            total_frame = add_image_to_background_cloud(object_frame3,total_frame,300 + i,500)
            cv2.imshow("frame2",total_frame)
            cv2.waitKey(50)

if __name__ == "__main__":
    video_path = "src/temp/test_flash"    
    # generate_raw_flash(video_path)
    thread1 = threading.Thread(target=generate_raw_flash,args=(video_path,))
    thread1.start()
    import time
    time.sleep(10)
    print("******")
    thread2 = threading.Thread(target=generate_raw_flash,args=(video_path,))
    thread2.start()
    