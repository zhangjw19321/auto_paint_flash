import numpy as np
import cv2 as cv
import cv2
from add_pic import *
from extract_contour_by_tools import *
from generate_flash import *


################## step one: get a picutre #############
# gril student boy mogu bear sheep
# may need to convert to ＲGB
image_file = "src/input/gril.jpeg"
image_file = "thresh.png"
raw_frame = cv2.imread(image_file)

################# step two: auto paint ################
from AttentionedDeepPaint.colorize import *
colored_frame = paint_color(raw_frame)
save_colored_name = "src/temp/colored_frame.png"
cv2.imwrite(save_colored_name,colored_frame)

################ step three: extract contour #########
transparent_coloed_frame = koutu(save_colored_name)
cv2.imwrite("src/temp/transparent_colored.png",transparent_coloed_frame)

############### step four: generate animate ##########
from Animate.animate_api import *
image = "src/temp/transparent_colored.png"
driving_video = "src/driving_video/dance7.mp4"
model = "taichi"
animate(image,driving_video,model)
'''
############# step five: generate flash ############
background_image = "src/backgroud/backgroud.jpeg"
cloud_image = "src/backgroud/cloud_contour.png"
animation_video = "result.mp4"
flow_picture(background_image,cloud_image,animation_video)
'''



if __name__ == "__main__":
    pass 
    # flow_picture()
    # frame = cv.imread("cloud_contour.png")
    # fangshe(frame)
