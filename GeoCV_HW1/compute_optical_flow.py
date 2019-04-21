
import numpy as np
import cv2
import matplotlib.pyplot as plt
import collections 

from mylib import stack_two_images, ImageReader
# from mydisp import show

def cv2_show(src_img, bg_img):
    # bg_img: gray image, higher pixel value means background
    # bg_img = bg_img.astype(np.float)/255 # change to gray with range [0, 1]
    img = stack_two_images(src_img, bg_img)
    cv2.imshow('frame', img)
    k = cv2.waitKey(10)
    if k!=-1 and chr(k)=='q':
        return False
    return True

def plt_show(i, size=(6,10)):
    plt.figure(figsize=size)
    plt.imshow(i, cmap='gray')
    plt.show()
    return True

def compute_std_image(imgs_list):
    imgs_list = np.array(imgs_list) 
    std_img = np.std(imgs_list, axis=0)
    # print(std_img.max(), std_img.min())
    std_img *= 3
    std_img = 255 - std_img.astype(np.uint8)
    # print(std_img.shape)
    return std_img

def compute_sum_changes(imgs_list):
    
    def to_float(img):
        return img.astype(np.float)

    diff_img = np.zeros(imgs_list[0].shape, dtype=np.float)
    prev = to_float(imgs_list[0])
    for i in range(1, len(imgs_list)):
        curr = to_float(imgs_list[i])
        diff_img += np.abs(curr - prev)
        prev = curr 

    diff_img *= 3.0 / len(imgs_list)

    print(diff_img.max(), diff_img.min())
    return 255 - diff_img.astype(np.uint8)

if __name__=="__main__":

    # Initialize
    image_reader = ImageReader(source="from_folder", folder_name="cam_3/")
    # fgbg = cv2.createBackgroundSubtractorMOG2()

    imgs_list = collections.deque()

    # Read some images and saved to list
    MAX_IMAGE = 1000
    cnt_img = 0
    WINDOW_SIZE = 100
    while cnt_img < MAX_IMAGE and image_reader.have_image():

        src_img = image_reader.read(resize_rate = 0.15)

        # sharpen images
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        src_img = cv2.filter2D(src_img, -1, kernel)

        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, ksize=(3,3), sigmaX=1)
        imgs_list.append(gray)
        cnt_img += 1
        print("Reading the {}th image".format(cnt_img))

        if cnt_img >= WINDOW_SIZE:
            # bg_img = compute_std_image(imgs_list)
            bg_img = compute_sum_changes(imgs_list)
            if not cv2_show(src_img, bg_img):
                break 
            if len(imgs_list) > WINDOW_SIZE:
                imgs_list.popleft()