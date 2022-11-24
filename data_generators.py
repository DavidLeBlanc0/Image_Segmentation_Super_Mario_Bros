import keras
import cv2
import numpy as np
import os
from image_functions import preprocess_image
import random

class DataGen(keras.utils.Sequence):
    def __init__(self, imagePaths, gridPaths, batchSize, overscan,
    shuffle = True,):
        """Initialization
        :param imagePaths: a list of paths to all the images
        :param gridPaths: a list of paths to all the grids
        :param batchSize: the number of images / grids per batch
        :param shuffle: whether to shuffle the dataset after each epoch
        """
        self.imagePaths = imagePaths
        self.gridPaths = gridPaths
        assert(len(self.imagePaths) == len(self.gridPaths))
        self.shuffle = shuffle
        self.overscan = overscan

        self.batchSize = batchSize
        self.on_epoch_end() # This shuffles the dataset

    def __len__(self):
        """
        Returns the number of batches per epoch
        """
        return int(np.floor(len(self.imagePaths) / self.batchSize))

    
    def __getitem__(self, idx):
        'Returns one batch for training'
        batchXPaths = self.imagePaths[idx * self.batchSize:(idx + 1) *
        self.batchSize]
        batchYpaths = self.gridPaths[idx * self.batchSize:(idx + 1) *
        self.batchSize]

        images = []
        grids = []

        for imagePath in batchXPaths:
            img = cv2.imread(imagePath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_image(img, overscan = self.overscan) # Source imgs r full res
            images.append(img)

        for gridPath in batchYpaths:
            grid = np.loadtxt(gridPath)
            grids.append(grid.T)


        return (np.array(images), np.array(grids))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.imagePaths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

def get_paths_from_dataset(datasetRootFolder):
    imageFolder = os.path.join(datasetRootFolder, "imgs")
    gridFolder = os.path.join(datasetRootFolder, "grids")

    imagePaths = []
    gridPaths = []

    for imgPath in os.listdir(imageFolder):
        imgPath = os.path.join(imageFolder, imgPath)
        imagePaths.append(imgPath)

    for gridPath in os.listdir(gridFolder):
        gridPath = os.path.join(gridFolder, gridPath)
        gridPaths.append(gridPath)

    return imagePaths, gridPaths