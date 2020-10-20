import numpy as np
import cv2 as cv
import cv2
from add_pic import *
from extract_contour_by_tools import *
from generate_flash import *

background_image = "backgroud.jpeg"
cloud_image = "cloud.jpeg"
animation_video = "Animate/result.mp4"
flow_picture(background_image,cloud_image,animation_video)