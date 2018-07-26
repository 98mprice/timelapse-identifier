from PIL import Image
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import os

'''b = cv2.imread('frames/frame100.jpg')
c = Image.fromarray(cv2.subtract(b, a).astype(np.uint8))
c.show()'''

vidcap = cv2.VideoCapture('videoplayback.mp4')
fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
success, a = vidcap.read()

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter("www.mp4", fourcc,  fps, (a.shape[1], a.shape[0]))

for i in range(1, 500):
    if i % 30 == 0:
        a = b
    success, b = vidcap.read()
    c = cv2.subtract(b, a)
    video.write(c)
cv2.destroyAllWindows()
video.release()
