# added for girder interaction as plugin arbor task
from girder_worker.app import app
from girder_worker.utils import girder_job
from tempfile import NamedTemporaryFile

import billiard as multiprocessing
from billiard import Queue, Process 
import json

#-------------------------------------------

import torch
from torch.utils.data import Dataset as BaseDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import datasets, models, transforms
import torchnet.meter.confusionmeter as cm

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

import openslide as op
import argparse
import numpy as np
import torchvision
import cv2
import time
from skimage.io import imread
from tifffile import imsave
import matplotlib.pyplot as plt
import time
import random
import os, glob
import copy
import pandas as pd
import albumentations as albu
from albumentations import Resize
import gc
import timm
from radam import RAdam

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPORTING_INTERVAL = 10

IMAGE_SIZE = 224
PRINT_FREQ = 20


@girder_job(title='survivability')
@app.task(bind=True)
def survivability(self,image_file, segment_image_file,**kwargs):
    print('running Arbor Nova survivability')
    print(" input image filename = {}".format(image_file))


    # set the UI to 0% progress initially. stdout is parsed by the ui
    print(f'progress: {0}')

    # find and run all models in the models directory. Return the average value of the models
    # as the final result
    resultArray = []
    resultArraySecondBest = []
    resultArrayMean = []
    models = glob.glob('./models/surv*')
    totalFolds = len(models)
    for fold,model in enumerate(models):
        print('**** running with model',model)
        print('****')  
        predict_values = start_remote_process(image_file,segment_image_file,model,fold,totalFolds)
        print(predict_values)
        resultArraySecondBest.append(predict_values['secondBest'])
        resultArrayMean.append(predict_values['mean'])
        #resultArray.append(predict_values)
        print('result is now:',resultArray)
    print('completed all folds')

 
    # NOTE: normalizing network output to range 0..1 to make the plot look better
    # we round the values to four decimal places so the rendering looks better
    predict_values_2nd = round(sum(resultArraySecondBest) / len(resultArraySecondBest),4)
    predict_values_mean = round(sum(resultArrayMean) / len(resultArrayMean),4)

    # for a while, we were rescaling by the exponential function, but this has been removed
    #predict_values_2nd = round(math.exp(sum(resultArraySecondBest) / len(resultArraySecondBest)),4)
    #predict_values_mean = round(math.exp(sum(resultArrayMean) / len(resultArrayMean)),4)

    # new output of classification statistics in a string
    statistics = generateStatsString(predict_values_2nd,predict_values_mean)
    print(statistics)

    # find the average of the model results
    #predict_values = sum(resultArray) / len(resultArray)

    # new output of classification statistics in a string
    #statistics = generateStatsString(predict_values)
    # generate unique names for multiple runs.  Add extension so it is easier to use
    statoutname = NamedTemporaryFile(delete=False).name+'.json'
    open(statoutname,"w").write(statistics)

    # return the name of the output file
    return statoutname


# calculate the statistics for the image by converting to numpy and comparing masks against
# the tissue classes. create masks for each class and count the number of pixels
def generateStatsString(second,mean):         
    statsDict = {'secondBest':second,'mean':mean }
    # convert dict to json string
    print('statsdict:',statsDict)
    statsString = json.dumps(statsDict)
    return statsString

#-------------------------------------------
# call subprocess because we need to have a different set of python dependencies but
# still want to start and stop the process to clear out the GPU memory. 


def start_remote_process(image_file,segmentation_mask,modelFilePath,foldCount,totalFolds):
    print('-----------------------------------')
    print('pretend running fold:',foldCount)
    print('image:',image_file)
    print('segment:',segmentation_mask)
    print('model file:',modelFilePath)
    return {'secondBest':0.34,'mean':0.33}
