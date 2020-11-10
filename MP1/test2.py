import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

def blueChannel(img_in):
    image = cv2.imread(img_in)
    b, g, r = cv2.split(image) #Splits image into 3 channels
    return b

def greenChannel(img_in):
    image = cv2.imread(img_in)
    b, g, r = cv2.split(image) #Splits image into 3 channels
    return g

def redChannel(img_in):
    image = cv2.imread(img_in)
    b, g, r = cv2.split(image) #Splits image into 3 channels
    return r

def convoFilter(img_in, kernel):
    data = np.array(img_in) #This converts img to data array
    
    modified = np.zeros_like(data) #Use zero_like since we want the same size matrix as image
    #Algorithm:
        #Flip the kernel 180
    horiFlip = np.fliplr(kernel) #Flip Horizontally
    kernel = np.flipud(horiFlip) #Flip Vertically
    
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]
    
    h = kernel.shape[0]//2
    w = kernel.shape[1]//2
        #Zero Pad : Height = shape[0] Width = shape[1]
    for x in range(h, data.shape[0]-h):
        for y in range(w, data.shape[1]-w):
            modified = 0
        #Traverse through Pixel
    for i in range(k_h):
        for j in range(k_w):
            modified = modified + (kernel[i][j] * data[i: x-h+i, j: y-w+j])

    print(modified) #Prints data matrices to console
    return modified

def convolution(img_in, kernel):
    blueImage = blueChannel(img_in)
    blueConv = convoFilter(blueImage, kernel)
    redImage = redChannel(img_in)
    redConv = convoFilter(redImage, kernel)
    greenImage = greenChannel(img_in)
    greenConv = convoFilter(greenImage, kernel)
    result = cv2.merge((blueConv, greenConv, redConv))
    return result

if __name__ == "__main__":
    #Kernel Filters
    none = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    blur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9
    sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16

    img_in = cv2.imread('lena.png')
    img_out = cv2.filter2D(img_in, -1, sharp)
    cv2.imshow("Altered Image", img_out)
    cv2.waitKey(0)