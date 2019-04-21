
import numpy as np
import cv2
import matplotlib.pyplot as plt
import collections 

from mylib import stack_two_images, ImageReader
# from mydisp import show


def cv2_show(img):
    cv2.imshow('frame', img)
    k = cv2.waitKey(10)
    if k!=-1 and chr(k)=='q':
        return False
    return True

if __name__=="__main__":

    # Initialize
    folder_name = "cam_3/"
    output_folder = "cam_3_resized/"
    image_reader = ImageReader(source="from_folder", folder_name=folder_name)

    while image_reader.have_image():

        src_img = image_reader.read(resize_rate = 0.25)
        file_name = image_reader.filenames[image_reader.cnt_image-1]
        file_name = file_name.split('/')[-1]

        cv2.imwrite(output_folder + file_name, img=src_img)

        if not cv2_show(src_img):
            break 
        print("Processing {}th image".format(image_reader.cnt_image))