import time
import cv2
def show_image():
    while True:
        frame = cv2.imread("file.png")
        if frame is not None:
            cv2.imshow("file",frame)
            cv2.waitKey(30)
        else:
            continue
if __name__ == "__main__":
    show_image()

