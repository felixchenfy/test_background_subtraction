
import numpy as np
import cv2
import matplotlib.pyplot as plt
import collections 

from mylib import stack_two_images, ImageReader

def cv2_show(src_img, bg_img):
    # bg_img: gray image, higher pixel value means background
    # bg_img = bg_img.astype(np.float)/255 # change to gray with range [0, 1]
    img = stack_two_images(src_img, bg_img)
    cv2.imshow('frame', img)
    k = cv2.waitKey(10)
    if k!=-1 and chr(k)=='q':
        return False 
    return True

def compute_optical_flow(prvs, next):
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    Ivx=flow[...,0]
    Ivy=flow[...,1]
    V = np.sqrt(Ivx**2 + Ivy**2)

    
    print(V.min(), V.max())
    V *= 10
    V = V.astype(np.uint8)
    return V

if __name__=="__main__":

    # Initialize
    image_reader = ImageReader(source="from_folder", folder_name="cam_3/")

    # Read some images and saved to list
    MAX_IMAGE = 1000
    cnt_img = 0
    while cnt_img < MAX_IMAGE and image_reader.have_image():

        src_img = image_reader.read(resize_rate = 0.15)
        cnt_img += 1

        if cnt_img == 1:
            prvs = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(src_img)
            hsv[...,1] = 255
            cnt=0
        else:
            
            
            next = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
            V = compute_optical_flow(prvs, next)

            if not cv2_show(src_img, V):
                break
            prvs = next
