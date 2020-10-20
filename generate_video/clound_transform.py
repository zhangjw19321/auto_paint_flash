import numpy as np
import cv2 as cv
import cv2
from add_pic import *
from extract_contour_by_tools import *

################## step one: get a picutre #############
image_file = "carton5.jpeg"
raw_frame = cv2.imread(image_file)

################# step two: auto paint ################
from AttentionedDeepPaint.colorize import *
colored_frame = paint_color(raw_frame)
save_colored_name = "colored_frame.png"
cv2.imwrite(save_colored_name,colored_frame)

################ step three: extract contour #########
transparent_coloed_frame = koutu(save_colored_name)
cv2.imwrite("transparent_colored.png",transparent_coloed_frame)









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
def flow_picture():
    from time import sleep
    background = cv2.imread("backgroud.jpeg")
    cloud_frame = cv.imread("cloud_contour.png")
    cap = cv2.VideoCapture("dance_teacher.mp4")
    for i in range(10):
        ret,_ = cap.read()
    sleep(5)
    cloud_frame = cv2.resize(cloud_frame,(int(cloud_frame.shape[1]/3),int(cloud_frame.shape[1]/3)))
    frame_count = 1000
    for i in range(frame_count):
        rate = 0.02 / 100 * i 
        trans_cloud_frame = fangshe(cloud_frame,rate)
        flow_frame = add_image_to_background_cloud(trans_cloud_frame,background,i,0)
        ret,object_frame = cap.read()
        if i % 30 == 8:
            p_name = str(i) + ".png"
            cv2.imwrite(p_name,object_frame)
        if ret:
            contour = extract_contour_grabcut(object_frame)
            total_frame = add_image_to_background_cloud(contour,flow_frame,300 + i,400)
        cv2.imshow("flow",total_frame)
        cv2.waitKey(50)


if __name__ == "__main__":
    pass 
    # flow_picture()
    # frame = cv.imread("cloud_contour.png")
    # fangshe(frame)
