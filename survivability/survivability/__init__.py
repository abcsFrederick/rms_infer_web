from girder import plugin
from girder.plugin import GirderPlugin
from tempfile import NamedTemporaryFile

from girder.api.rest import Resource
from girder.api.describe import Description, describeRoute, RestException
from girder.api import access
from girder.models.file import File

import pymongo
from pymongo import MongoClient
import girder_client
import json
import logging
import arrow
import datetime

#import requests
import string
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

# used for girder client login 
globals = {}
globals['girderUser'] = 'anonymous'
globals['girderPassword'] = 'letmein'

#-------------------------------------------


#subprocessMethod = 'billiard'
#subprocessMethod = 'torch'
subprocessMethod = 'none'

if (subprocessMethod == "torch"):
    print('setup torch multiprocessing')
    import torch.multiprocessing as multiprocessing
    #import billiard as multiprocessing
    from torch.multiprocessing import Queue, Process
    from torch.multiprocessing import set_start_method
elif (subprocessMethod == "billiard"):
    print('setup billiard multiprocessing')
    import billiard
    import billiard as multiprocessing
    from billiard import Queue, Process
else:
    print('no subprocess method set')


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

# declare the handlers for get, put, post, delete
class SurvivabilityInference_API(Resource):
    def __init__(self):
        super(SurvivabilityInference_API, self).__init__()
        self.resourceName = 'survivability'

        #the POST is used to retrieve 
        self.route('GET', (), self.parseGetCommands)
        self.route('PUT', (), self.parsePutCommands)
        self.route('POST', (), self.performInference)
        #self.route('DELETE', (':id',), self.deleteRemoteIntakeRecords)

  

    # define a handler for the GET option that looks for a standard format command/data object
    # and dispatches to the correct handling routine.  This is organized where there is a function for 
    # each command.  This dispatcher just checks that the arguments are valid and calls the correct handler
 
    @access.public
    @describeRoute( 
        Description('dispatcher for GET API calls')
        .param('command', 'the api command string.', required=True,paramType='query')
        .param('data', 'the  command data as a JSON object.', required=False,paramType='query')
        .errorResponse())

    def parseGetCommands(self,params):
        # made the data param optonal because some commands might not need it?  Maybe this is inviting more 
        # parsing, but the testing could 
        self.requireParams(('command'), params)
        print('received GET command with params:',params)

        # check that the URL has the proper arguments (a command and a data argument) 
        try:
            commandName = params['command']
            print('infer: received command:', commandName)
        except ValueError:
            raise RestException('Invalid JSON command name passed in request body.')
        # dispatch successful commands
        if params['command'] == 'get_log':
            return self.getLogRecords()       
        elif params['command'] == 'get_stats':
            return self.getStats(params)     
        else:
            print('infer: incorrect command for GET dispatch detected:', params)
            response = {'status':'failure','data':params}
            return response



    @access.public
    @describeRoute( 
        Description('dispatcher for PUT API calls')
        .param('command', 'the api command string.', required=True,paramType='query')
        .param('data', 'the  command data as a JSON object.', required=False,paramType='query')
        .errorResponse())

    def parsePutCommands(self,params):
        print('received a PUT command')
        pass


    # configure the logger and open a database connection
    def setupLogging(self):
        global log_collection
        global client

        # setup database-based log
        client = MongoClient(globals['mongo-host'] ,globals['mongo-port'])
        db = client[globals['mongo-database']] 
        log_collection = db[globals['mongo-log-collection']]

    # do any cleanup 
    def closeLogChannels(self):
        global client
        client.close()

    def logActivityDetails(self,message,level='Info'):
        global log_collection
        # write database  with timestamp
        utc = arrow.utcnow()
        localtime = utc.to(globals['timezone'] )
        timestring = localtime.format('YYYY-MM-DD HH:mm:ss ZZ')
        logstring = timestring+' '+message
        log_collection.insert({'timestamp': logstring, 'message': message,
            'year':localtime.year,
            'month':localtime.month,
            'day':localtime.day,
            'hour':localtime.hour,
            'weekday':localtime.weekday()}
            )


    # routine that writes log message and writes them to the database 
    def logActivity(self,message,loglevel='Info'):
        self.setupLogging()
        self.logActivityDetails(message,loglevel)
        self.closeLogChannels()


    # return the records of system use by reading the log in the mongoDB instance    
    def getLogRecords(self):
        client = MongoClient(globals['mongo-host'] ,globals['mongo-port'])
        db = client[globals['mongo-database']]
        form_collection = db[globals['mongo-log-collection']]
        # put the entire record into the mongo database
        query = {}
        records = form_collection.find(query,{})
        response = {}
        # copy the records into a list and return the list
        logrecords = []
        for x in records:
            logrecords.append(x)
            print('log list: returning ',x)
        response['result'] = logrecords
        response['status'] = 'success'
        client.close()
        return response

    # dummy placeholder
    def getStats(self):
        pass


    # ************************* image inferencing entry point *****

    # define handle dispatcher for POST calls. 
    @access.public
    @describeRoute( 
        Description('dispatcher for POST API calls')
        .param('data', 'the  command data as a JSON object.', required=False,paramType='query')
        .errorResponse())

    def performInference(self,params):   
        print('POST params:',params)
       
        try:
            print('params:',params)
            image_filename = params['imageFileName']
            imageId = params['imageId']
            segment_filename = params['segmentFileName']
            segmentId = params['segmentId']

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
        for fold,model in enumerate(models):
            print('**** running with model',model)
            print('****')  
            predict_values = survival_inferencing(image_file,segment_file,model,fold,totalFolds)
            #predict_values = {'secondBest':0.1,'mean':0.2}
            print('prediction: ',predict_values)
            resultArraySecondBest.append(predict_values['secondBest'])
            resultArrayMean.append(predict_values['mean'])
            print('progress:',(fold+1)/totalFolds*100.0)
        print('completed all folds')

        # find the average of the model results
        # NOTE: normalizing network output to range 0..1 to make the plot look better
        # we round the values to four decimal places so the rendering looks better
        predict_values_2nd = round(math.exp(sum(resultArraySecondBest) / len(resultArraySecondBest)),4)
        predict_values_mean = round(math.exp(sum(resultArrayMean) / len(resultArrayMean)),4)

        # new output of classification statistics in a string
        statistics = self.generateStatsString(predict_values_2nd,predict_values_mean)
        print(statistics)

        # build the response object that contains both status and results
        response = {'status':'success','stats':statistics}
        print(response)

        # return the name of the output file
        return response


    # count the number of polygons so we can print a graph of the number of detected regions
    def generateStatsString(self,second,mean):         
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


#---------------------------




# for Girder 3.0: we need to declare the plugin using a 
# different class.  The load function is required to initialize 
# the plugin code and specify the route (.survivability) where the new 
# functions will be accessed

class GirderPlugin(plugin.GirderPlugin):
    DISPLAY_NAME = 'survivability'

    def load(self,info):
        info['apiRoot'].survivability = SurvivabilityInference_API()

