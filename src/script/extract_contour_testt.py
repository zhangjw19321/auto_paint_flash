import sys
import requests
import base64
import cv2
def koutu(i_image):
    response = requests.post(
        'http://www.picup.shop/api/v1/matting?mattingType=2', 
        files={'file': open(i_image, 'rb')},
        headers={'APIKEY': '0879d40ea2eb4bc7be77d4910ceba5ba'},
    )

    with open('out.png', 'wb') as out:
        out.write(response.content)

    img_info = response.content.decode("utf-8").split(",")[-3].split('"')[-2]
    imgdata = base64.b64decode(img_info)
    image_file = open("temp_extract.png","wb")
    image_file.write(imgdata)
    image_file.close()
    frame = cv2.imread("temp_extract.png")
    return frame

if __name__ == "__main__":
    in_file = "renwu.png"
    frame = koutu(in_file) 
    cv2.imshow("frame",frame)   
    cv2.waitKey(3000)