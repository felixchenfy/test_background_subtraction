

import numpy as np
import cv2

int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

def sharpen_image(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def erode_gray_image(gray):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gray, kernel, iterations = 1)
    return erosion

def is_color_img(img):
    return len(img.shape)==3 and img.shape[2]==3 and (type(img[0][0][0])!=np.float64)

def stack_two_images(src, gray):
    if not is_color_img(src):
        img1 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    else:
        img1 = src 
    img2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return np.hstack((img1, img2))
        
def cv2_show(src_img, background_img):
    # background_img = background_img.astype(np.float)/255 # change to gray with range [0, 1]
    img = stack_two_images(src=src_img, gray=background_img)
    cv2.imshow('frame', img)
    k = cv2.waitKey(10)
    if k!=-1 and chr(k)=='q':
        return False
    return True 

class ImageReader(object):
    def __init__(self, source, folder_name=None, video_name=None):
        assert source in ["from_video", "from_folder"]

        if source == "from_video":
            self.video_name = video_name
            self.cap = cv2.VideoCapture(self.video_name)
        else: # from_folder
            self.folder_name = folder_name + "/" if folder_name[-1] != "/" else folder_name
            self.filenames = self.get_filenames(self.folder_name)

        self.source = source 
        self.cnt_image = 0

    def __del__(self):
        if self.source == "from_video":
            self.cap.release()

    def read(self, filename=None, cap=None, resize_rate=0.2):

        if self.source == "from_video":
            ret, frame = self.cap.read()
            assert not ret, "All images in video were read out"
        else: # from_folder
            frame = cv2.imread(self.filenames[self.cnt_image])

        self.cnt_image += 1

        if resize_rate != 1:
            frame = cv2.resize(src=frame, dsize=(0, 0), dst=None, fx=resize_rate, fy=resize_rate)

        return frame 

    def have_image(self):
        if self.source == "from_video":
            return True
        else: # from_folder
            return self.cnt_image < len(self.filenames)
    
    def get_prev_img_filename(self):
        return self.filenames[self.cnt_image-1].split('/')[-1]

    def get_filenames(self, path, format="*"):
        import glob
        list_filenames = sorted(glob.glob(path+'/'+format))
        return list_filenames  # Path to file wrt current folder