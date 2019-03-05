# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mycv import mycv

def notShowAxisLabel():
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
cap = cv2.VideoCapture("v_Biking_g01_c01.avi")
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
cnt=0
while(1):
    cnt+=1
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    Ivx=flow[...,0]
    Ivy=flow[...,1]
    if 0:
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
    else:
        plt.clf()
        gap=20
        Ivx=Ivx[0::gap,0::gap]
        Ivy=Ivy[0::gap,0::gap]
        if 0:
            if cnt==1:
                fig = plt.figure(figsize=(17, 6))
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            notShowAxisLabel()
            plt.subplot(122)
            plt.quiver(Ivx, Ivy)
            notShowAxisLabel()
            mycv.savefig("frames/IandOF_"+"{:03d}".format(cnt)+".png", border_size=10)
            plt.pause(0.0001)
        else:
            plt.clf()
            plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            # plt.axis('off')
            mycv.savefig("frames/image_"+"{:03d}".format(cnt)+".png", border_size=10)
            plt.pause(0.01)
            plt.clf()
            plt.quiver(Ivx, Ivy)
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            # plt.axis('off')
            mycv.savefig("frames/opflow_"+"{:03d}".format(cnt)+".png", border_size=10)
            plt.pause(0.01)

    prvs = next

plt.show()
cap.release()
cv2.destroyAllWindows()