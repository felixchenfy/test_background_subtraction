# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

class OpticalFlow(object):
    def __init__(self):
        self.min_max_speed = 20

        self.img_color_curr = None
        self.img_gray_curr = None
        self.img_color_prev = None
        self.cnt_img = 0

    def insert_image(self, img_color):
        self.cnt_img += 1
        self.img_color_prev = self.img_color_curr
        self.img_color_curr = img_color
        self.flow_magnitude = None

    def compute_optical_flow(self):
        if self.cnt_img == 1:
            return self.get_black_image(depth = 0)
        else:
            # Compute optical flow: vx and vy
            gray_curr = cv2.cvtColor(self.img_color_curr,cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.img_color_prev,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
            # Compute magnitude
            flow_vx=flow[...,0]
            flow_vy=flow[...,1]
            self.flow_magnitude = np.sqrt(flow_vx**2+flow_vy**2)
            
            return self.flow_magnitude

    def flow_to_image(self, flow_magnitude):
        flow_uint8 = (flow_magnitude*10).astype(np.uint8)
        flow_img = cv2.cvtColor(flow_uint8, cv2.COLOR_GRAY2BGR)
        return flow_img

    def get_black_image(self, depth):
        s = self.img_color_curr.shape
        if depth == 0:
            return np.zeros((s[0], s[1]))
        else:
            return np.zeros((s[0], s[1], depth))

    def get_mask_of_moving(self):
        if self.cnt_img == 1:
            return self.get_black_image(depth = 0)
        else:
            V = self.flow_magnitude
            mask =  V / max(self.min_max_speed, V.max())
            mask[mask>1]=1
            mask = mask**(0.5)
            return mask
            

class ModelBackground(object):
    def __init__(self, changing_rate = 0.03):
        self.img_color_bg = None
        self.cnt_img = 0
        self.changing_rate = changing_rate

    def insert_image(self, img_color):
        self.cnt_img += 1
        if self.cnt_img == 1: # Init background model
            if 0: # init with black
                self.img_color_bg = np.zeros_like(img_color)
            else: # init with first image
                self.img_color_bg = img_color.copy()

        # Make background model similar to the current image
        self.img_color_bg = (1 - self.changing_rate) * self.img_color_bg + \
                                self.changing_rate * img_color

    def get_background_image(self):
        return self.img_color_bg.astype( np.uint8 )

    def get_mask_of_foreground(self, img_color_curr, min_max_diff = 200):
        I1 = img_color_curr.astype(np.float)
        I2 = self.img_color_bg.astype(np.float)
        d = I1 - I2
        mask = np.sqrt( d[..., 0]**2 + d[..., 1]**2 + d[..., 2]**2 )
        mask = mask / max(min_max_diff, mask.max())
        mask[mask>1]=1
        mask = mask**(0.5)
        return mask
        

def mask2image(mask):
    m = (mask * 255).astype(np.uint8)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)

def setMaskOntoImage(mask, img):
    mask = mask.reshape(img.shape[0], img.shape[1], 1)
    res = img * mask
    res = res.astype(np.uint8)
    return res

if __name__=="__main__":

    image_folder = "my_images/"
    # image_folder = "my_images2/"
    image_folder_out = image_folder[:-1] + "_res/"

    of = OpticalFlow()
    bg = ModelBackground()

    for cnt_img in range(130, 400+1):
        filename = int2str(cnt_img, 5)+".png"
        img = cv2.imread(image_folder + filename)
        img = cv2.resize(img, (0,0), fx=0.6, fy=0.6) 

        # Compute optical flow
        of.insert_image(img)
        flow_magnitude = of.compute_optical_flow()
        print("{}th image".format(cnt_img), ", max vel = ", flow_magnitude.max())
        # flow_img = of.flow_to_image(flow_magnitude)
        mask_of = of.get_mask_of_moving()

        # Compute background model
        bg.insert_image(img)
        # bg_img = bg.get_background_image()
        mask_bg = bg.get_mask_of_foreground(img)

        # Plot

        img_disp_of = np.hstack( (img, mask2image( mask_of ),
            setMaskOntoImage(mask_of, img) ))

        img_disp_bg = np.hstack( (img, mask2image( mask_bg ),
            setMaskOntoImage(mask_bg, img) ))

        mask2 = (mask_bg * mask_of)**(0.9)
        img_disp_combine = np.hstack( (img, mask2image( mask2 ),
            setMaskOntoImage(mask2, img) ))


        img_disp = np.vstack((img_disp_of, img_disp_bg, img_disp_combine))
        cv2.imshow("optical flow", img_disp)
        cv2.imwrite(image_folder_out + filename, img_disp)

        # Waitkey
        q = cv2.waitKey(1)
        if q!=-1 and chr(q) == 'q':
            break
  
cv2.destroyAllWindows()
