import os
import cv2

def show_frame(frame,name):
    cv2.imshow(name,frame)
    cv2.waitKey(3000)
# back_frame = cv2.imread("backgroud.jpeg")
# back_frame = cv2.resize(back_frame,(1920,1080))
def select_roi(image):
    cloud_frame = cv2.imread(image,0)
    color_frame = cv2.imread(image)
    ret, cloud_frame = cv2.threshold(cloud_frame, 250, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(cloud_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = -1
    location_queue = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > max_area:
            max_area = area
            location_queue = cv2.boundingRect(c)
    x, y, w, h = location_queue
    cv2.rectangle(color_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    show_frame(color_frame,"ff")
    return color_frame[y:y+h,x:x+w]
if __name__ == "__main__":
    image_cloud = "cloud_contour.png"
    cloud_roi = select_roi(image_cloud)
    cv2.imwrite("roi_cloud.png",cloud_roi)

    