import sys
import requests
import base64
import cv2
from PIL import Image

def covert_transparent2black(image_path):
    try:
        imagePtah = image_path
        img = Image.open(imagePtah)
        if img.mode != 'RGBA':
            image = img.convert('RGBA')
        width = img.width
        height = img.height
        image = Image.new('RGB', size=(width, height), color=(0, 0, 0))
        image.paste(img, (0, 0), mask=img)
        image.save(image_path)        
    except Exception as e:
        print(e)

def koutu(i_image):
    response = requests.post(
        'http://www.picup.shop/api/v1/matting2?mattingType=2', 
        files={'file': open(i_image, 'rb')},
        headers={'APIKEY': '0879d40ea2eb4bc7be77d4910ceba5ba'},
    )
    img_info = response.content.decode("utf-8").split(",")[-3].split('"')[-2]
    imgdata = base64.b64decode(img_info)
    image_path = "src/temp/temp_extract.png"
    image_file = open(image_path,"wb")
    image_file.write(imgdata)
    image_file.close()
    covert_transparent2black(image_path)
    frame = cv2.imread(image_path)
    return frame

if __name__ == "__main__":
    in_file = "renwu.png"
    frame = koutu(in_file) 
    cv2.imshow("frame",frame)   
    cv2.waitKey(3000)