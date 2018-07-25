from PIL import Image
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import os

writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30,(640,480))

for frame in range(10):
    c = np.random.randint(0, 255, (480,640,3)).astype('uint8')
    print c.shape
    writer.write(c)

writer.release()
