
# External Libraries
import keras
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import os
import contextlib
import seaborn as sn
import pandas as pd

MATRIX_NAME = "confusion_matrix.png"
MATRIX_LABELS = ["None", "Ground", "Player", "Enemy", "Hazard"]
VID_NAME = "eval_vid.mp4"
PREDS_NAME = "eval_preds.svg"
MODEL_JSON_NAME = "model_details.json"
MODEL_SUMMARY_NAME = "model_summary.txt"
FIG_DPI = 300

def eval_custom_encoder(model, dataGen, evalPath = "./"):
    """
    Loads a custom encoder from given path and calls eval() with autoencoder
    :param dataSourcePath:  path to dataset folder. Should have imgs and 
    grids as folders
    :param autoencoderPath: path to autoencoder h5 file 
    """
    #imagePaths, gridPaths = data_generators.get_paths_from_dataset(dataSourcePath)

    #dataGen  = data_generators.customDataGen(imagePaths, gridPaths, BATCH_SIZE, overscan = 8)
    #model = keras.models.load_model(modelPath)

    # MODEL SUMMARY
    #model.summary()
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(evalPath, MODEL_JSON_NAME), "w") \
            as json_file:
        json_file.write(model_json)

    with open(os.path.join(evalPath, MODEL_SUMMARY_NAME), 'w') as f:
        with contextlib.redirect_stdout(f):
            model.summary()
    
    # TEST SET CREATION
    testLabels = []
    datNum = 0
    # Ensure samples with all labels are present
    while len(np.unique(testLabels)) < len(MATRIX_LABELS):
        testImgs, testLabels = dataGen.__getitem__(datNum)
        datNum += 1

    # PRED VS ACTUAL VS SOURCE
    # NOTE: If you place this after the confusion matrix, the colour is wrong
    numShown = 3
    f, axarr = plt.subplots(3, 3)
    preds = model.predict(testImgs)
    
    for i in range(numShown):
        axarr[i,0].imshow(testImgs[i])
        axarr[i,1].imshow(preds[i])
        axarr[i,2].imshow(testLabels[i])

    plt.savefig(os.path.join(evalPath, PREDS_NAME), dpi = FIG_DPI)
    plt.clf()

    # CONFUSION MATRIX
    preds = model.predict(testImgs)
    preds = preds.flatten()
    preds = np.round(preds)
    preds = preds.clip(min=0, max = len(MATRIX_LABELS) - 1)
    testLabelsFlat = testLabels.flatten()
    print(np.unique(preds))
    print(np.unique(testLabelsFlat))
    confMatrix = sklearn.metrics.confusion_matrix(testLabelsFlat, preds)
    confMatrix = confMatrix.astype('float') / confMatrix.sum(axis=1)[:, np.newaxis]
    print(sklearn.metrics.classification_report(testLabelsFlat, preds))

    sn.set(font_scale=1.2)
    heatmap = sn.heatmap(pd.DataFrame(confMatrix), 
    annot=True,
    xticklabels = MATRIX_LABELS,
    yticklabels = MATRIX_LABELS)
    plt.xticks(rotation = 0)
    plt.yticks(rotation = 0)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.tight_layout()
    heatmap.get_figure().savefig(os.path.join(evalPath, MATRIX_NAME),
    dpi = FIG_DPI)