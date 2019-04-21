
import numpy as np
import cv2

# https://docs.opencv.org/3.4/db/d5c/tutorial_py_bg_subtraction.html

int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

FROM_VIDEO = False
if FROM_VIDEO:
    cap = cv2.VideoCapture('vtest.avi')
else:
    image_folder = "my_images/"
    image_folder_out = image_folder[:-1] + "_res/"

fgbg = cv2.createBackgroundSubtractorMOG2()

for cnt_img in range(130, 400+1):
    if FROM_VIDEO:
        ret, frame = cap.read()
    else:
        filename = int2str(cnt_img, 5)+".png"
        frame = cv2.imread(image_folder + filename)
    print("{}th image".format(cnt_img))
    
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(100)
    if k!=-1 and chr(k)=='q':
        break
cap.release()
cv2.destroyAllWindows()
