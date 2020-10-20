import cv2
from grabcut import *
from extract_contour import *
from add_pic import *
def convertRGB2RGBA(frame):
    img = frame
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    # 最小值为0
    alpha_channel[:, :] = 255
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

def single_test():
    background = cv2.imread("backgroud.jpeg")
    background_alpha = convertRGB2RGBA(background)
    colored_frame = cv2.imread("colored_carton5.jpeg")
    contour = extract_contour_grabcut(colored_frame)
    transparent_frame = transparent(contour)
    shape = transparent_frame.shape
    background_alpha[0:shape[1],0:shape[0]] = transparent_frame
    cv2.imwrite("total.png",background_alpha) 
    # cv2.waitKey(3000)

def background_transfor():
    cap = cv2.VideoCapture("taiji5.mp4")
    frame_count = 100
    background = cv2.imread("backgroud.jpeg")
    background_alpha = convertRGB2RGBA(background)
    for i in range(frame_count):
        ret,frame = cap.read()
        if ret:
            contour = extract_contour_grabcut(frame)
            transparent_frame = add_image_to_background(contour,background,300,300)
            cv2.imshow("total",transparent_frame) 
            cv2.waitKey(1)

if __name__ == "__main__":
    background_transfor()
