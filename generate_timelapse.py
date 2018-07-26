from PIL import Image
import numpy as np
import sys
import cv2
import os

if not os.path.exists("/output/timelapse_frames"):
    os.makedirs("/output/timelapse_frames")
if not os.path.exists("/output/video_frames"):
    os.makedirs("/output/video_frames")

vidcap = cv2.VideoCapture('/input/timelapse.mp4')
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
success, a = vidcap.read()

for i in range(1, length):
    if i % 30 == 0:
        a = b
    success, b = vidcap.read()
    c = cv2.subtract(b, a)
    print("timelapse %d" % i)
    cv2.imwrite('/output/timelapse_frames/%d.png' % i, c)

vidcap = cv2.VideoCapture('/input/video.mp4')
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
success, a = vidcap.read()

for i in range(1, length):
    if i % 30 == 0:
        a = b
    success, b = vidcap.read()
    c = cv2.subtract(b, a)
    print("video %d" % i)
    cv2.imwrite('/output/video_frames/%d.png' % i, c)
