# added for girder interaction as plugin arbor task
from girder_worker.app import app
from girder_worker.utils import girder_job
from tempfile import NamedTemporaryFile

import torch
import subprocess

import json

#-------------------------------------------


import argparse
import numpy as np

import time
import random
import os, glob
import copy


from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPORTING_INTERVAL = 10

IMAGE_SIZE = 224
PRINT_FREQ = 20

# define global variable that is set according to whether GPUs are discovered
USE_GPU = True


@girder_job(title='survivability')
@app.task(bind=True)
def survivability(self,image_file, segment_image_file,**kwargs):
    global USE_GPU
    print('running ensemble survivability model')
    #print(" input image filename = {}".format(image_file))

    gpu_count = torch.cuda.device_count()
    if torch.cuda.is_available():
        print('cuda is available')
        print('using',gpu_count,'CUDA devices available')
    else:
        print('cuda is not available')

    # set the UI to 0% progress initially. stdout is parsed by the ui
    print('progress: 2.5')
    print('progress: 5')

    # find and run all models in the models directory. Return the average value of the models
    # as the final result
    resultArray = []
    resultArraySecondBest = []
    resultArrayMean = []
    models = glob.glob('/rms_infer_web/models/surv*')
    totalFolds = len(models)
    for fold,model in enumerate(models):
        print('**** running with model',model)
        predict_values = start_remote_process(image_file,segment_image_file,model,fold,totalFolds)
        print(predict_values)
        resultArraySecondBest.append(float(predict_values['secondBest']))
        resultArrayMean.append(float(predict_values['mean']))
        #resultArray.append(predict_values)
        progressPercent = round((fold+1)/totalFolds*100,2)
        print('progress:',progressPercent)
        print('progress:',progressPercent+1)
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
    statsDict = {"secondBest":second,"mean":mean }
    # convert dict to json string
    print('statsdict:',statsDict)
    statsString = json.dumps(statsDict)
    return statsString

#-------------------------------------------
# call subprocess because we need to have a different set of python dependencies but
# still want to start and stop the process to clear out the GPU memory. 


def start_remote_process(image_file,segmentation_mask,modelFilePath,foldCount,totalFolds):
    #print('-----------------------------------')
    #print('pretend running fold:',foldCount)
    #print('image:',image_file)
    #print('segment:',segmentation_mask)
    #print('model file:',modelFilePath)

    print('spawning subprocess to run inference')
    outputText = subprocess.run(['/rms_infer_web/survive_shell.sh',image_file,segmentation_mask,modelFilePath],
                        stdout=subprocess.PIPE).stdout.decode('utf-8')

    #print('--------------------------')
    #print('output of subprocess was:')
    print(outputText)
    outputAsDict = json.loads(outputText)
    print(outputAsDict)
    return outputAsDict
