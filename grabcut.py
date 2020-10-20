import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def extract_contour_grabcut(frame):
    img = frame
    OLD_IMG = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    SIZE = (1, 65)
    bgdModle = np.zeros(SIZE, np.float64)
    fgdModle = np.zeros(SIZE, np.float64)
    rect = (1, 1, img.shape[1], img.shape[0])
    s_time = time.time()
    cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 2, cv2.GC_INIT_WITH_RECT)
    e_time = time.time()
    print("single cost time is: ", e_time - s_time)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img *= mask2[:, :, np.newaxis]
    return img
'''
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("grabcut"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(OLD_IMG, cv2.COLOR_BGR2RGB))
plt.title("original"), plt.xticks([]), plt.yticks([])

plt.show()
'''

if __name__ == "__main__":
    frame = cv2.imread("src/input/bear.jpeg",0)
    _,frame = cv2.threshold(frame,20,255,cv2.THRESH_BINARY)
    print(type(frame))
    contour = extract_contour_grabcut(frame)
    cv2.imshow("contour", contour)
    cv2.waitKey(5000)
    cv2.imwrite("carton5_contour.png",contour)