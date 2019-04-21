import cv2
import numpy as np
import matplotlib.pyplot as plt

# define functions for plotting
def convert(img):
    '''change image color from "BGR" to "RGB" for plt.plot()'''
    if len(img.shape)==3 and img.shape[2]==3 and (type(img[0][0][0])!=np.float64):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show(i, size=(6,10)):
    i = convert(i)
    plt.figure(figsize=size)
    plt.imshow(i)
    plt.show()
    
def show2(image_list, titles, size=(10,5)):
    images = [convert(i) for i in image_list]
    plt.figure(figsize=size)
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])

def show3(image_list, titles, size=(15,5)):
    images = [convert(i) for i in image_list]
    plt.figure(figsize=size)
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
    
def show22(image_list, titles, size=(10,14)):    
    images = [convert(i) for i in image_list]
    plt.figure(figsize=size)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])