import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

def convolution(img_in, kernel):
    data = np.array(plt.imread(img_in)) #This converts img to data array
    
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
            zeroPad = 0
        #Traverse through Pixel
    for i in range(k_h):
        for j in range(k_w):
            zeroPad = zeroPad + (kernel[i][j] * data[i: x-h+i, j: y-w+j])

    print(zeroPad) #Prints data matrices to console
    return zeroPad
    

if __name__ == "__main__":
    none = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    blur = np.array([[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]])
    sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16

    img_in = 'art.png'
    img_out = convolution(img_in, none)
    plt.imshow(img_out)
    plt.show()
