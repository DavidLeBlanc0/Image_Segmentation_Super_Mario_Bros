import numpy as np
import cv2
import os

def preprocess_image(numpyImg, scaleFac = 1, greyscale = True, overscan = 0):
    if type(overscan) == list and len(overscan) == 1:
        overscan = overscan[0]
    if overscan != 0:
        if type(overscan) == tuple or type(overscan) == list:
            assert(len(overscan) == 4)
            #print(overscan)
            numpyImg = numpyImg[
                overscan[0]:overscan[1],
                overscan[2]:overscan[3],
            ]
        else:
            numpyImg = numpyImg[
                overscan:numpyImg.shape[0] - overscan,
                overscan:numpyImg.shape[1] - overscan,
            ]
    
    newXres = int(numpyImg.shape[0]/scaleFac)
    newYres = int(numpyImg.shape[1]/scaleFac)

    numpyImg = cv2.resize(numpyImg, dsize=(newXres, newYres), 
        interpolation=cv2.INTER_NEAREST)
    if greyscale:
        numpyImg = cv2.cvtColor(numpyImg, cv2.COLOR_RGB2GRAY)
    #numpyImg = cv2.normalize(numpyImg, None, norm_type=NORM_MINMAX)

    if greyscale:
        numpyImg = numpyImg[...,np.newaxis]

    numpyImg = numpyImg/255

    return numpyImg