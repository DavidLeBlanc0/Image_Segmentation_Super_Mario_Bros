import segmentation_models
import segmentation_model_eval
import data_generators
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import evaluation_tools

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.model_selection import KFold


plt.rcParams['svg.fonttype'] = 'none' # Makes svg text save as text, not paths

MATRIX_LABELS = ["None", "Ground", "Player", "Enemy", "Hazard"]
MODEL_TYPES = ['small', 'full']
BATCH_SIZE = 32
MAX_EPOCHS = 100

if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description =
        "Runs experiments for the thesis")

    argParser.add_argument("-n", "--name", 
        required = True,
        type = str,
        help = "The name of the directory to save the results in. A directory will be created if it does not exist")

    argParser.add_argument("-i", "--input", 
        required = True,
        type = str,
        help = "The name of the dataset directory with grids and imgs folders")

    argParser.add_argument("-m", "--model",
        required = True,
        choices = MODEL_TYPES,
        help = "What type of model to train. Choices:"
    )

    argParser.add_argument("-o", "--overscan",
        required = True,
        type = int,
        help = "How many pixels of overscan to perform on the images"
    )

    argParser.add_argument("--evalMode",
        action="store_true",
        help = "Load a model and treat the dataset as a test set"
    )

    argParser.add_argument("--kfolds",
        required = False,
        type = int,
        default = 10,
        help = "How many folds of cross validation to perform.")


    argParser.add_argument("--testPercent",
    required = False,
    type = int,
    default = 5,
    help = "Percent (integer) to remove from dataset for testing.")

    args = argParser.parse_args()

    print("Checking eval directory...")
    if not os.path.exists(args.name):
        os.mkdir(args.name)


    print("Assembling dataset...")
    imagePaths, gridPaths = data_generators.get_paths_from_dataset(args.input)
    imagePaths = np.array(imagePaths)
    gridPaths = np.array(gridPaths)
    assert(len(imagePaths) == len(gridPaths))

    print("Shuffling Dataset...")
    shuffleIdxs = np.arange(len(imagePaths))
    imagePaths = imagePaths[shuffleIdxs]
    gridPaths = gridPaths[shuffleIdxs]

    print("Removing test set from training...")
    testSplit = len(imagePaths) - int(len(imagePaths)*(args.testPercent/100))
    testImagePaths = imagePaths[testSplit:]
    testGridPaths = gridPaths[testSplit:]
    imagePaths = imagePaths[:testSplit]
    gridPaths = gridPaths[:testSplit]

    # Is a datagen appropriate here?
    testGen = data_generators.DataGen(
        imagePaths = testImagePaths,
        gridPaths = testGridPaths,
        overscan = args.overscan,
        batchSize = len(testImagePaths)
    )

    if args.evalMode:
        print("Evaluation mode active...")

        #dataGen  = data_generators.customDataGen(imagePaths, gridPaths, BATCH_SIZE, overscan = 8)

        print("ERROR: NOT IMPLEMENTED")
        sys.exit()

    kTrainLosses = []
    kValLosses = []
    kTestMatrices = []
    kf1Scores = []
    splitNo = 0

    kFolder = KFold(n_splits = args.kfolds)
    for trainIdxs, valiIdxs in kFolder.split(imagePaths):
        trainDatagen = data_generators.DataGen(
            imagePaths = imagePaths[trainIdxs],
            gridPaths = gridPaths[trainIdxs],
            overscan = args.overscan,
            batchSize = BATCH_SIZE)
        valiDatagen = data_generators.DataGen(
             imagePaths = imagePaths[valiIdxs],
            gridPaths = gridPaths[valiIdxs],
            overscan = args.overscan,
            batchSize = BATCH_SIZE)

        # Determine IO dims
        testImg, testGrid = trainDatagen.__getitem__(0)
        testImg = testImg[0]
        testGrid = testGrid[0]


        assert trainDatagen.__len__() > 0
        assert valiDatagen.__len__() > 0


        if args.model == 'small':
            #print("Building a small model...")
            model = segmentation_models.create_small_model(inputDims=testImg.shape, outputDims=testGrid.shape)

        if args.model == 'large':
            #print("Building a large model...")
            model = segmentation_models.create_large_model(inputDims=testImg.shape, outputDims=testGrid.shape)

        """
        TRAINING
        """

        es  = EarlyStopping(monitor='val_loss', patience = 10, min_delta=0.0001)
        log = CSVLogger(os.path.join(args.name, "training_log_" + str(splitNo) + ".csv"), append = False)

        hist = model.fit(x = trainDatagen, validation_data = valiDatagen, 
            callbacks = [es, log], epochs = MAX_EPOCHS, verbose = 0)
        kTrainLosses.append(hist.history['loss'])
        kValLosses.append(hist.history['val_loss'])
        model.save(os.path.join(args.name, "model_" + str(splitNo) + ".h5"))

        """
        TESTING
        """
        testImgs, testLabels = testGen.__getitem__(0)

        testPreds = model.predict(testImgs, verbose = 0)
        testPreds = testPreds.flatten()
        testPreds = np.round(testPreds)
        testPreds = testPreds.clip(min=0, max = len(MATRIX_LABELS) - 1)
        testLabelsFlat = testLabels.flatten()
        confMatrix = sklearn.metrics.confusion_matrix(testLabelsFlat, testPreds)
        confMatrix = confMatrix.astype('float') / confMatrix.sum(axis=1)[:, np.newaxis]
        #print(sklearn.metrics.classification_report(testLabelsFlat, testPreds))
        #print(confMatrix)
        kTestMatrices.append(confMatrix)
        f1Score = sklearn.metrics.f1_score(testLabelsFlat, testPreds, average='macro')
        kf1Scores.append(f1Score)



        splitNo += 1
        print("Completed fold",splitNo)

    """
    EVAL
    """
    kTrainLosses = np.array(kTrainLosses)
    kValLosses = np.array(kValLosses)
    kTrainLosses = evaluation_tools.pad_to_dense(kTrainLosses)
    kValLosses = evaluation_tools.pad_to_dense(kValLosses)

    kTrainLossMean = np.nanmean(kTrainLosses, axis = 0)
    kTrainLossStd = np.nanstd(kTrainLosses, axis = 0)
    kTrainLossStd = kTrainLossStd / np.sqrt(np.size(kTrainLosses, axis = 0))
    kValLossMean = np.nanmean(kValLosses, axis = 0)
    kValLossStd = np.nanstd(kValLosses, axis = 0)
    kValLossStd = kValLossStd / np.sqrt(np.size(kValLosses, axis = 0))

    kConfMean = np.mean(kTestMatrices, axis = 0)
    kConfStd = np.std(kTestMatrices, axis = 0)

    evaluation_tools.plot_confusion_matrix(cm = kConfMean, cms = kConfStd, classes=MATRIX_LABELS)
    plt.savefig(os.path.join(args.name, "confMatrix.svg"))

    #print(kConfMean)

    numEpochs = np.arange(len(kTrainLossMean))
    fig, ax = plt.subplots(figsize=(14,6), dpi = 300)

    ax.plot(kTrainLossMean, 'b', label='Training Loss')
    ax.fill_between(numEpochs, kTrainLossMean - kTrainLossStd,kTrainLossMean + kTrainLossStd, facecolor='b', alpha=0.2)

    ax.plot(kValLossMean, 'r', label='Validation Loss')
    ax.fill_between(numEpochs, kValLossMean - kValLossStd,kValLossMean + kValLossStd, facecolor='r', alpha=0.2)
    ax.legend(loc = 'upper right')

    kf1Scores = np.asarray(kf1Scores)
    np.savetxt(os.path.join(args.name, "macroF1Scores.csv"), kf1Scores, delimiter=',')
    kf1Mean = np.mean(kf1Scores)
    kf1Std = np.std(kf1Scores)
    kf1Std = kf1Std / np.sqrt(np.size(kf1Scores))

    with open(os.path.join(args.name, "f1Summary.txt"), 'w') as f:
        f.write("Mean Macro Averaged F1: ")
        f.write(str(kf1Mean))
        f.write("\n")
        f.write("Standard Error: ")
        f.write(str(kf1Std))

    #np.savetxt(os.path.join(args.name, "macroF1ScoresMean.csv"), kf1Mean, delimiter=',')
    #np.savetxt(os.path.join(args.name, "macroF1ScoreStd.csv"), kf1Std, delimiter=',')


    plt.savefig(os.path.join(args.name, "lossGraph.svg"))

        

    
