import cv2
import pyzbar.pyzbar as pyzbar
import json

auto_paint_flag = False
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

if __name__ == "__main__":
    scan_qr()    
