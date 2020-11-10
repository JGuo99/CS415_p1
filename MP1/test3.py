import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import math as m

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
    #Height = shape[0]; Width = shape[1]
    imgHeight = data.shape[0] #Height of original Image
    imgWidth = data.shape[1] #Width of original Image
    kernHeight = kernel.shape[0]
    kernWidth = kernel.shape[1]
    modified = np.zeros_like(data) #Use zero_like since we want the same size matrix as image. This is the output array.
    #Algorithm:
        #Flip the kernel 180
    horiFlip = np.fliplr(kernel) #Flip Horizontally
    kernel = np.flipud(horiFlip) #Flip Vertically

    #Zero Padding
    zeroPad = np.zeros((imgHeight+2, imgWidth+2)) #Pad the array with -> Make a NxN into N+2xN+2
    zeroPad[1:-1, 1:-1] = data #Put the data of the image in the center of the padded array

        #Traverse through Pixel
    for i in range(imgHeight):
        for j in range(imgWidth):
            modified[i, j] = np.sum(kernel * zeroPad[i: i+kernHeight, j: j+kernWidth])

    #print(modified) #Prints data matrices to console
    return modified

def correFilter(img_in, kernel):
    data = np.array(img_in) #Converts img to darray
    imgHeight = data.shape[0] #Height of original Image
    imgWidth = data.shape[1] #Width of original Image
    kernHeight = kernel.shape[0]
    kernWidth = kernel.shape[1]
    modified = np.zeros_like(data)
    #Algorithm:
        #Zero Padding
    zeroPad = np.zeros((imgHeight+2, imgWidth+2))
    zeroPad[1:-1, 1:-1] = data 

        #Traverse through Pixel
    for i in range(imgHeight):
        for j in range(imgWidth):
            modified[i, j] = np.sum(kernel * zeroPad[i: i+kernHeight, j: j+kernWidth]) 

    return modified

def medFilter(img_in, k_size):
    data = np.array(img_in)
    imgHeight = data.shape[0]
    imgWidth = data.shape[1]

    window = []
    modified = np.zeros_like(data)
    pointer = m.floor(k_size/2)
    
    #Algorithm:
    for x in range(imgHeight):
        for y in range(imgWidth):
            for z in range(k_size):
                if x + z - pointer > imgHeight - 1 or x + z - pointer < 0:
                    for i in range(k_size):
                        window.insert(0,0) #Insert Zero in front of array
                elif y + z + pointer > imgWidth - 1 or y + z - pointer < 0:
                        window.insert(0,0) #Insert Zero in front of array
                else:
                    for j in range(k_size):
                        window.insert(0, data[x + z - pointer][y + j - pointer])
            #Median
            window.sort()
            modified[x][y] = window[m.floor(len(window)/2)]
            window = []
    return modified 

def convolution(img_in, kernel):
    blueImage = blueChannel(img_in)
    blueConv = convoFilter(blueImage, kernel)
    greenImage = greenChannel(img_in)
    greenConv = convoFilter(greenImage, kernel)
    redImage = redChannel(img_in)
    redConv = convoFilter(redImage, kernel)

    result = cv2.merge((blueConv, greenConv, redConv))
    return result

def correlation(img_in, kernel):
    blueImage = blueChannel(img_in)
    blueCorr = correFilter(blueImage, kernel)
    greenImage = greenChannel(img_in)
    greenCorr = correFilter(greenImage, kernel)
    redImage = redChannel(img_in)
    redCorr = correFilter(redImage, kernel)
    
    result = cv2.merge((blueCorr, greenCorr, redCorr))
    return result

def median_filter(img_in, kernel):
    blueImage = blueChannel(img_in)
    blueMed = medFilter(blueImage, kernel)
    greenImage = greenChannel(img_in)
    greenMed = medFilter(greenImage, kernel)
    redImage = redChannel(img_in)
    redMed = medFilter(redImage, kernel)
    
    result = cv2.merge((blueMed, greenMed, redMed))
    return result
    

if __name__ == "__main__":
    none = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    mean = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9
    meanFive = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])/25
    sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
    gaussFive = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])/256
    diff = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    k_size = 3

    img_in = 'art.png'
    # img_out = convolution(img_in, mean)
    # img_out = correlation(img_in, mean)
    img_out = median_filter(img_in, k_size)

    cv2.imshow("Altered Image", img_out)
    cv2.waitKey(0)
