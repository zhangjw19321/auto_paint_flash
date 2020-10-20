import numpy as np
import cv2 as cv
import cv2
from add_pic import *
from extract_contour_by_tools import *



def fangshe(img,rate = 0.1):
    rows, cols, channels = img.shape
    p1 = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    p2 = np.float32([[0,rows*rate], [cols*(1-rate),rows*rate], [cols*rate,rows*(1-rate)]])
    # p2 = np.float32([[0,rows*0.3], [cols*0.8,rows*0.1], [cols*0.15,rows*0.7]])
    M = cv.getAffineTransform(p1, p2)
    dst = cv.warpAffine(img, M, (cols,rows))
    return dst
    # cv.imshow('original', img)
    # cv.imshow('result', dst)
    # cv.waitKey(9000)
    # cv.destroyAllWindows()
def cloud_flow():
    background = cv2.imread("backgroud.jpeg")
    cloud_frame = cv.imread("cloud_contour.png")
    cloud_frame = cv2.resize(cloud_frame,(int(cloud_frame.shape[1]/3),int(cloud_frame.shape[1]/3)))
    for i in range(600):
        rate = 0.04 / 100 * i 
        trans_cloud_frame = fangshe(cloud_frame,rate)
        total_frame = add_image_to_background_cloud(trans_cloud_frame,background,i,0)
        cv2.imshow("flow",total_frame)
        cv2.waitKey(50)
def add_new_flash(cap,location):
    from time import sleep
    background = cv2.imread(backgroud_image)
    while True:
        cloud_frame = cv.imread(cloud_image)
        cap = cv2.VideoCapture(video_path)
        backgroud_cap = cv2.VideoCapture("src/backgroud3.mp4")
        # sleep(5)
        cloud_frame = cv2.resize(cloud_frame,(int(cloud_frame.shape[1]/3),int(cloud_frame.shape[1]/3)))
        frame_count = 2000
        
        for i in range(frame_count):
            print("i is::",i)
            rate = 0.02 / 100 * i 
            trans_cloud_frame = fangshe(cloud_frame,rate)
            _,background = backgroud_cap.read()
            background = cv2.resize(background,(1920,1080))
            flow_frame = add_image_to_background_cloud(trans_cloud_frame,background,i,0)
            ret,object_frame = cap.read()
            if object_frame is None:
                print("generate flash is finish")
                break
            if i % 30 == 8:
                p_name = str(i) + ".png"
                pass
                # cv2.imwrite(p_name,object_frame)
            if ret:
                # contour = extract_contour_grabcut(object_frame)
                contour = object_frame
                total_frame = add_image_to_background_cloud(contour,flow_frame,300 + i,400)
                total_frame = add_image_to_background_cloud(contour,total_frame,300 + i,700)
            cv2.imshow("flow",total_frame)
            cv2.waitKey(30)
    cap.release()
    cv2.destroyAllWindows()



def flow_picture(backgroud_image,cloud_image,video_path):
    from time import sleep
    background = cv2.imread(backgroud_image)
    while True:
        cloud_frame = cv.imread(cloud_image)
        cap = cv2.VideoCapture(video_path)
        backgroud_cap = cv2.VideoCapture("src/backgroud/backgroud3.mp4")
        # sleep(5)
        cloud_frame = cv2.resize(cloud_frame,(int(cloud_frame.shape[1]/3),int(cloud_frame.shape[1]/3)))
        frame_count = 2000
        
        for i in range(frame_count):
            print("i is::",i)
            rate = 0.02 / 100 * i 
            trans_cloud_frame = fangshe(cloud_frame,rate)
            _,background = backgroud_cap.read()
            background = cv2.resize(background,(1920,1080))
            flow_frame = add_image_to_background_cloud(trans_cloud_frame,background,i,0)
            ret,object_frame = cap.read()
            if object_frame is None:
                print("generate flash is finish")
                break
            if i % 30 == 8:
                p_name = str(i) + ".png"
                pass
                # cv2.imwrite(p_name,object_frame)
            if ret:
                # contour = extract_contour_grabcut(object_frame)
                contour = object_frame
                total_frame = add_image_to_background_cloud(contour,flow_frame,300 + i,400)
                total_frame = add_image_to_background_cloud(contour,total_frame,300 + i,700)
            cv2.imshow("flow",total_frame)
            cv2.waitKey(30)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    pass 
    # flow_picture()
    # frame = cv.imread("cloud_contour.png")
    # fangshe(frame)
