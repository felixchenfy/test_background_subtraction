
import numpy as np
import cv2

from mylib import stack_two_images, ImageReader

if __name__=="__main__":

    # Initialize
    image_reader = ImageReader(source="from_folder", folder_name="cam_3/")
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg = cv2.createBackgroundSubtractorKNN()

    # Process
    cnt_img = 0
    while image_reader.have_image():

        # Read image
        frame = image_reader.read(resize_rate=0.2)
        cnt_img += 1
        print("{}th image".format(cnt_img))
        
        # Process
        fgmask = 255 - fgbg.apply(frame)
        
        # Plot
        img_disp = stack_two_images(frame, fgmask)
        cv2.imshow('frame', img_disp)
        k = cv2.waitKey(100)
        if k!=-1 and chr(k)=='q':
            break
            
    cv2.destroyAllWindows()
