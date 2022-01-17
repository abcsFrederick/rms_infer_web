
# for job
from girder_jobs.constants import JobStatus
from girder_jobs.models.job import Job
import json

import sys

# needed for exponential function applied to NN predicted value
import math

# needed to copy imagery from girder to files for endpoint access
import girder_client

#-------------------------------------------

import torch
import torch.nn as nn
import openslide as op
import argparse
import numpy as np
from skimage.io import imread
import time
import random
import os, glob
from albumentations import Resize
import gc
import timm

from PIL import Image

from argparse import Namespace

Image.MAX_IMAGE_PIXELS = None

# globals
globals = {}
globals['mongo-database'] = 'RMS_infer'
globals['mongo-host'] = 'localhost'
globals['mongo-log-collection'] = 'logging'
globals['mongo-port'] = 27017
globals['timezone'] = 'US/Eastern'
globals['girderUser'] = 'anonymous'
globals['girderPassword'] = 'letmein'
globals['docker'] = True

if globals['docker'] == True:
    globals['modelPath'] = '/rms_infer_web/models/'
else:
    globals['modelPath'] = '/home/ubuntu/rms_infer_web/models/'


client = None
log_collection = None

# this is a function called by Girder's createLocalJob() call

# we spawn a "local job" that is executed by the Girder thread, but acts like
# a job.  We can get logs back from local jobs, so this is used to parse the
# log by the UI for progress updates.  The called routine logs "progress: x"
# occasionally, which the UI is looking for.  If the job fails for some reason,
# we print out the reason for the failure in the job log, which show up in 
# girder's 'error.log'

def localJobFunction(job):
    print("Hooray!  I created a local job! ")
    print('job:',job)
    # run the inference job
    print('CWD is:',os.getcwd())
    # tell girder that we are running now
    job = Job().updateJob(job, status=JobStatus.RUNNING)
    try:
        stats = performInferenceFunction(job)
       
    except Exception as e:
        print('oh no....')
        print(e)

        job = Job().updateJob(job, status=JobStatus.ERROR,log=repr(e))
    else:
        status = JobStatus.SUCCESS
        job = Job().updateJob(
            job, log=stats, status=status)
    



    # ************************* image inferencing entry point *****


def performInferenceFunction(job):   
    print('POST job:',job)
    
    try:
        print('job:',job)
        image_filename = job['kwargs']['imageFileName']
        imageId = job['kwargs']['imageId']
        segment_filename = job['kwargs']['segmentFileName']
        segmentId = job['kwargs']['segmentId']

    except:
        print('could not read image file variable')

    # hard to find the file on the disk, so download again.  Inefficient, but it works.
    # fetch the actual image data from the uploaded record.  Do this once so all folds 
    # can have access. 

    print('finding image in girder backend')
    gc = girder_client.GirderClient(apiUrl='http://localhost:8080/girder/api/v1')
    login = gc.authenticate(globals['girderUser'],globals['girderPassword'])
    print('logged into girder successfully.')
    print('trying to local filename of file',imageId)
    try:
        fileRec = gc.getFile(imageId)
        print('found file',fileRec['_id'])
        print('found image file record',fileRec)
        print('downloading file')
        gc.downloadFile(fileRec['_id'],'/tmp/imageFile')
        image_file = '/tmp/imageFile'
        print('setting infile name and downloaded it')
    except:
        print('could not find or read matching image file in girder')
    try:
        segfileRec = gc.getFile(segmentId)
        #print('found file',fileRec['_id'])
        print('found segment file record',segfileRec)
        # hard to find the file on the disk, so download again.  Inefficient, but it works
        print('downloading segmentfile')
        gc.downloadFile(segfileRec['_id'],'/tmp/segmentFile')
        segment_file = '/tmp/segmentFile'
    except:
        print('could not find or read matching segmentation file in girder')    

    # setup the GPU environment for pytorch
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DEVICE = torch.device(f"cuda:0")
    print('perform forward inferencing')

    # find and run all models in the models directory. Return the average value of the models
    # as the final result
    resultArraySecondBest = []
    resultArrayMean = []
    models = glob.glob('./models/surv*')
    totalFolds = len(models)
    print('found ',totalFolds,'models to run')
    # set an initial 5% progress while the first fold runs
    logMessage = 'progress: 2%\n'
    print(logMessage)
    job = Job().updateJob(job, log=logMessage)

    # loop through the models and update the job progress after each one completes
    for fold,model in enumerate(models):
        print('**** running with model',model)
        print('****')  
        predict_values = survival_inferencing(image_file,segment_file,model,fold,totalFolds)
        #predict_values = {'secondBest':0.1,'mean':0.2}
        print('prediction: ',predict_values)
        resultArraySecondBest.append(predict_values['secondBest'])
        resultArrayMean.append(predict_values['mean'])
        #print('progress:',(fold+1)/totalFolds*100.0)
        logMessage = 'progress: %4d%%\n' % ((fold+1)/totalFolds * 100)
        print(logMessage)
        job = Job().updateJob(job, log=logMessage)
    print('completed all folds')

    # find the average of the model results
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

    # log and return the result
    logMessage = statistics+'\n'
    job = Job().updateJob(job, log=logMessage)
    return statistics


# count the number of polygons so we can print a graph of the number of detected regions
def generateStatsString(second,mean):         
    statsDict = {'secondBest':second,'mean':mean }
    # convert dict to json string
    print('statsdict:',statsDict)
    statsString = json.dumps(statsDict)
    return statsString


#---------------------------
# begin survivability code 
#---------------------------


def reset_seed(seed):
    """
    ref: https://forums.fast.ai/t/accumulating-gradients/33219/28
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse():
    # values were collected by running the original code and determing the argument that were sent. Two different
    # models are used (wnum=11 and wnum=18)
    args = Namespace(batch_size=1000, data='./', deterministic=False, evaluate=False, 
            lr=0.1, momentum=0.9, numgenes=4, pretrained=False, print_freq=10, prof=-1, 
            resume='', sync_bn=False, tnum=1, vnum=2, weight_decay=0.0001, wnum=11, workers=64)
    return args

def convert_to_tensor(batch):
    num_images = batch.shape[0]
    tensor = torch.zeros((num_images, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8).cuda(non_blocking=True)

    mean = torch.tensor([0.0, 0.0, 0.0]).cuda().view(1, 3, 1, 1)
    std = torch.tensor([255.0, 255.0, 255.0]).cuda().view(1, 3, 1, 1)

    for i, img in enumerate(batch):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] = torch.from_numpy(nump_array)

    tensor = tensor.float()
    tensor = tensor.sub_(mean).div_(std)
    return tensor


def load_best_model(model, path_to_model, best_prec1=0.0):
    if os.path.isfile(path_to_model):
        print("=> loading checkpoint '{}'".format(path_to_model))
        checkpoint = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}), best_precision {}"
              .format(path_to_model, checkpoint['epoch'], best_prec1))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(path_to_model))


class W11_Classifier(nn.Module):
    def __init__(self, n_classes, numgenes):
        super(W11_Classifier, self).__init__()
        self.effnet = timm.create_model('resnet18d', pretrained=True)
        in_features = 1000
        hazard_func = 1

        self.final_act = nn.Tanh()
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.alpha_dropout = nn.AlphaDropout(0.25)
        self.l0 = nn.Linear(in_features, 64, bias=True)
        self.l1 = nn.Linear(numgenes, 64, bias=True)
        self.l2 = nn.Linear(64, 64, bias=True)
        self.l3 = nn.Linear(128, hazard_func, bias=True)

    def forward(self, input, gene_muts):
        x = self.effnet(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l0(x)
        x = self.relu(x)  # 64
        x = self.dropout(x)

        y = self.l1(gene_muts)
        y = self.elu(y)
        y = self.alpha_dropout(y)
        y = self.l2(y)
        y = self.elu(y)  # 64
        y = self.alpha_dropout(y)

        z = torch.cat((x, y), dim=1)
        z = self.l3(z)
        z = self.final_act(z)

        return z


class W18_Classifier(nn.Module):
    def __init__(self, n_classes, numgenes):
        super(W18_Classifier, self).__init__()
        self.effnet = timm.create_model('resnet18d', pretrained=True)
        in_features = 1000
        hazard_func = 1

        self.final_act = nn.Tanh()
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.alpha_dropout = nn.AlphaDropout(0.5)
        self.l0 = nn.Linear(in_features, 64, bias=True)
        self.l1 = nn.Linear(numgenes, 64, bias=True)
        self.l2 = nn.Linear(64, 64, bias=True)
        self.l3 = nn.Linear(64, hazard_func, bias=True)

    def forward(self, input, gene_muts):
        x = self.effnet(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l0(x)
        x = self.relu(x)  # 64
        x = self.dropout(x)
        z = self.l3(x)
        z = self.final_act(z)

        return z


def smart_sort(x, permutation):
    ret = x[permutation]
    return ret

IMAGE_SIZE = 224
PRINT_FREQ = 50
class_names = ['0', '1', '2']
num_classes = len(class_names)


def survival_inferencing(image_file,segment_file,model,fold,totalFolds):
    reset_seed(1)

    args = parse()
    working_number = args.wnum
    test_number = args.tnum
    valid_number = args.vnum

    working_number_str = '%01d' % working_number
    test_number_str = '%01d' % test_number
    valid_number_str = '%01d' % valid_number

    torch.backends.cudnn.benchmark = True

    best_prec1 = 0.0

    time.sleep(10)  # Time interval can be increased
    saved_weights_list = [model]
    print(saved_weights_list)

    # there are two different classifier types, use the model filename to delineate
    # which type e.g. 'surv_w11_xxx' will be a type 11 instead of type 18

    print('model type:',model[14:17])
    if model[14:17] == 'w11':
        print('found model type w11')
        model = W11_Classifier(num_classes, args.numgenes)
    elif model[14:17] == 'w18':
        print('found model type w18')
        model = W18_Classifier(num_classes, args.numgenes)
    else:
        print("***************** error: could not find matching model type to load for file",model)

    model.eval()
    model = nn.DataParallel(model)
    model = model.cuda()
    model = load_best_model(model, saved_weights_list[-1], best_prec1)
    print('Loading model is finished!!!!!!!')

    #print('args were:',args)
    prediction = wsi_inferencing(model,image_file,segment_file,fold, totalFolds, args)
    return prediction

# sort python list of numbers in numeric order
def sortNumericList(l):
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i] > l[j]:
                l[i], l[j] = l[j], l[i]
    return l

# return the mean of numbers in a list
def meanOfNumericList(l):
    sum = 0
    for i in range(len(l)):
        sum += l[i]
    return (sum / len(l))

# convert between -1..1 normalization and 0..1 normalization
def zeroToOneNorm(input):
    offset = input - (-1.0)
    newoffset = offset/2.0
    return newoffset


# given filenames for the image and segmentation, run the model forward and return the prediction 

def wsi_inferencing(model, image_file, segment_file, fold, totalFolds, args):
    IMAGE_SIZE_40x = 224 * 2
    IMAGE_SIZE_20x = 224
    model.eval()

    image_files = [image_file]

    patients_2nd = torch.zeros(len(image_files), 1)
    patients_avg = torch.zeros(len(image_files), 1)
    medians = torch.zeros(4000 // args.batch_size)
    sorted_medians = []
    other_index = 0

    # decide how long this will take and prepare to give status updates in the log file
    iteration_count = 10
    report_interval = 1
    report_count = 0
   
    print('reading H&E image_file:',image_file)
    # read the image file
    wholeslide = op.OpenSlide(image_file)
    sizes = wholeslide.level_dimensions[0]
    image_height = sizes[1]
    image_width = sizes[0]

    # read and resize the segmentation file to match the image size
    label_org = imread(segment_file)
    aug = Resize(p=1.0, height=image_height, width=image_width)
    augmented = aug(image=label_org, mask=label_org)
    label = augmented['mask']

    # sample 4000 different random patches on the image to determine the survivability
    for k in range(4000 // args.batch_size):
        image_width_start = 0
        image_width_end = image_width - IMAGE_SIZE - 1

        image_height_start = 0
        image_height_end = image_height - IMAGE_SIZE - 1

        x_coord = 0
        y_coord = 0

        patch_index = 0
        image_batch = np.zeros((args.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
        gene_batch = torch.zeros((args.batch_size, args.numgenes), dtype=torch.uint8).cuda(non_blocking=True)

        for j in range(args.batch_size):
            picked = False
            while (picked == False):
                x_coord = random.sample(range(image_width_start, image_width_end), 1)[0]
                y_coord = random.sample(range(image_height_start, image_height_end), 1)[0]
                label_patch = label[y_coord:y_coord + IMAGE_SIZE_40x, x_coord:x_coord + IMAGE_SIZE_40x]

                if (np.sum(label_patch // 255) > int(IMAGE_SIZE_40x * IMAGE_SIZE_40x * 0.50)) and (
                        np.sum(label_patch == 127) == 0):
                    picked = True
                else:
                    picked = False

            read_region = wholeslide.read_region((x_coord, y_coord), 0, (IMAGE_SIZE_40x, IMAGE_SIZE_40x))
            large_image_patch = np.asarray(read_region)[:, :, :3]
            image_aug = Resize(p=1.0, height=IMAGE_SIZE_20x, width=IMAGE_SIZE_20x)
            image_augmented = image_aug(image=large_image_patch)
            image_patch = image_augmented['image']

            image_batch[patch_index, :, :, :] = image_patch
            # the gene tensor is always zeros, but it is defined in the model so we have to pass it in
            gene_batch[patch_index, :] = torch.from_numpy(np.asarray([0, 0, 0, 0]))
            patch_index += 1

        with torch.no_grad():
            image_tensor = convert_to_tensor(image_batch)
            gene_tensor = gene_batch.float()
            logits = model(image_tensor, gene_tensor)
            permu = torch.argsort(logits, dim=0, descending=False)
            #print('permu: ', permu)

            logits = logits.view(-1)
            #print('Logits after view: ', logits)
            logits = smart_sort(logits, permu)
            logits = logits.view(-1, 1)
            #print('After logits: ', logits)

        median_index = k % (4000 // args.batch_size)
        medians[median_index] = logits[args.batch_size // 2, 0]
    
        # check that it is time to report progress.  If so, print it and flush I/O to make sure it comes 
        # out right after it is printed 
        report_count += 1
        if (report_count > report_interval):
            percent_complete = 100.0 * float(fold) / float(totalFolds)
            #print(f'progress: {percent_complete}')
            sys.stdout.flush()
            report_count = 0
 
        # this only runs the last time
        if k % (4000 // args.batch_size) == (4000 // args.batch_size - 1):
            sorted_medians = sortNumericList(medians)
            print('Median Values: ', sorted_medians)
            local_medians = sorted_medians.numpy()
            print('local medians',local_medians)
            #medians[:] = 0.
            returnVal = {'secondBest':local_medians[-2],'mean':np.mean(local_medians)}
            print('prediction:',returnVal)

    gc.collect()
    return returnVal
