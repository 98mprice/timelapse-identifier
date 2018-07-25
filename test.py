from PIL import Image
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import os

'''b = cv2.imread('frames/frame100.jpg')
c = Image.fromarray(cv2.subtract(b, a).astype(np.uint8))
c.show()'''
a = cv2.imread('frames/frame0.jpg')
vidcap = vidcap = cv2.VideoCapture('cloud_test.mp4')
fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter("output.mp4", fourcc,  fps, (a.shape[1], a.shape[0]))

for i in range(0, 100):
    if i % 10 == 0:
        a = cv2.imread('frames/frame%d.jpg' % i)
    b = cv2.imread('frames/frame%d.jpg' % i)
    c = cv2.subtract(b, a)
    video.write(c)
cv2.destroyAllWindows()
video.release()
