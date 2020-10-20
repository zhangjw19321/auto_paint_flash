import cv2
frame = cv2.imread("padding.png",0)
print(frame.shape)
cv2.imwrite("gray.png",frame)
ret,thresh = cv2.threshold(frame,100,255,cv2.THRESH_BINARY_INV)
convert = cv2.bitwise_not(thresh)
print("thresh is: ", frame)
cv2.imwrite("thresh.png",convert)
