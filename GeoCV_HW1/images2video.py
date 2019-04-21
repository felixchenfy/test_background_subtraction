

import numpy as np
import cv2
import sys, os, time
import numpy as np
import simplejson
import sys, os
import csv
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.append(CURR_PATH+"../")
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

def get_filenames(path, format="*"):
    import glob
    list_filenames = sorted(glob.glob(path+'/'+format))
    return list_filenames  # Path to file wrt current folder

if 1:
    image_folder = CURR_PATH + 'cam_3_res/'
    video_name = CURR_PATH + './res_video.avi'
    fnames = get_filenames(image_folder)
    N = len(fnames)
    image_start = 0
    image_end = 1500
    framerate = 10
    FASTER_RATE = 3

# Read image and save to video'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
cnt = 0
for i in range(image_start, image_end+1):
    cnt += 1
    fname = fnames[i]
    frame = cv2.imread(fname)
    if cnt==1:
        width = frame.shape[1]
        height = frame.shape[0]
        video = cv2.VideoWriter(video_name, fourcc, framerate, (width,height))
    if i%FASTER_RATE ==0:
        print("Processing the {}/{}th image: {}".format(cnt, image_end - image_start + 1, fname))
        cv2.imshow("", frame)
        q = cv2.waitKey(10)
        video.write(frame)

cv2.destroyAllWindows()
video.release()