from PIL import Image
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import os

vidcap = cv2.VideoCapture('video.mp4')
success, image = vidcap.read()
a = Image

for i in range(100):
    

b = cv2.imread('frames/frame100.jpg')
c = Image.fromarray(cv2.subtract(b, a).astype(np.uint8))
c.show()
