#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json

from arbor_nova_tasks.arbor_tasks.fnlcr import infer_rhabdo 
from arbor_nova_tasks.arbor_tasks.fnlcr import infer_wsi 
from arbor_nova_tasks.arbor_tasks.fnlcr import infer_rms_map
from arbor_nova_tasks.arbor_tasks.fnlcr import wsi_thumbnail
from arbor_nova_tasks.arbor_tasks.fnlcr import myod1
from arbor_nova_tasks.arbor_tasks.fnlcr import survivability 
from arbor_nova_tasks.arbor_tasks.fnlcr import cohort 

from girder.api import access
from girder.api.describe import Description, autoDescribeRoute
from girder.api.rest import filtermodel, Resource
from girder.models.setting import Setting
from girder.models.file import File
from girder.models.item import Item
from girder.constants import AccessType


from girder_worker_utils.transforms.girder_io import GirderFileId, GirderUploadToItem
from girder_worker_utils.transforms.contrib.girder_io import GirderFileIdAllowDirect

from girder_slurm.models.slurm import Slurm as slurmModel
from girder_slurm import utils as slurmUtils
from girder_slurm.constants import PluginSettings as slurmPluginSettings
import girder_slurm.girder_io.input as slurmGirderInput
import girder_slurm.girder_io.output as slurmGirderOutput
from girder_jobs.models.job import Job


class ArborNova(Resource):
    def __init__(self):
        super(ArborNova, self).__init__()
        self.resourceName = 'arbor_nova'
        self.route('POST', ('infer_rhabdo', ), self.infer_rhabdo)
        self.route('POST', ('infer_wsi', ), self.infer_wsi)
        self.route('POST', ('infer_wsi_hpc', ), self.infer_wsi_hpc)
        self.route('POST', ('infer_rms_map', ), self.infer_rms_map)
        self.route('POST', ('wsi_thumbnail', ), self.wsi_thumbnail)
        self.route('POST', ('myod1', ), self.myod1)
        self.route('POST', ('myod1_hpc', ), self.myod1_hpc)
        self.route('POST', ('survivability', ), self.survivability)
        self.route('POST', ('survivability_hpc', ), self.survivability_hpc)
        self.route('POST', ('cohort', ), self.cohort)
    @access.token
    @filtermodel(model='job', plugin='jobs')



# ---DNN infer command line for FNLCR
    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('perform forward inferencing using a pretrained network')
        .param('imageId', 'The ID of the source, a TIF image file.')
        .param('outputId', 'The ID of the output item where the output file will be uploaded.')
        .param('statsId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def infer_rhabdo(
            self, 
            imageId, 
            outputId,
            statsId
    ):
        result = infer_rhabdo.delay(
                #GirderFileId(imageId), 
                GirderFileIdAllowDirect(imageId), 
                girder_result_hooks=[
                    GirderUploadToItem(outputId),
                    GirderUploadToItem(statsId),

                ])
        return result.job

    # ---DNN infer command line for FNLCR
    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('perform forward inferencing using a pretrained network')
        .param('imageId', 'The ID of the source, an Aperio .SVS image file.')
        .param('outputId', 'The ID of the output item where the output file will be uploaded.')
        .param('statsId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def infer_wsi(
            self, 
            imageId, 
            outputId,
            statsId
    ):
        result = infer_wsi.delay(
                #GirderFileId(imageId), 
                GirderFileIdAllowDirect(imageId), 
                girder_result_hooks=[
                    GirderUploadToItem(outputId),
                    GirderUploadToItem(statsId),
                ]
                )
        return result.job


    # ---cancer map version of segmentation method
    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('perform forward inferencing using a pretrained network')
        .param('imageId', 'The ID of the source, an Aperio .SVS image file.')
        .param('outputId', 'The ID of the output item where the output file will be uploaded.')
        .param('statsId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def infer_rms_map(
            self, 
            imageId, 
            outputId,
            statsId
    ):
        result = infer_rms_map.delay(
                #GirderFileId(imageId), 
                GirderFileIdAllowDirect(imageId),               
                girder_result_hooks=[
                    GirderUploadToItem(outputId),
                    GirderUploadToItem(statsId),
                ]
                )
        return result.job



    # --- generate a thumbnail from a pyramidal image
    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('generate a wsi_thumbnail')
        .param('imageId', 'The ID of the source, an Aperio .SVS image file.')
        .param('outputId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def wsi_thumbnail(
            self, 
            imageId, 
            outputId
    ):
        result = wsi_thumbnail.delay(
                #GirderFileId(imageId), 
                GirderFileIdAllowDirect(imageId), 
                girder_result_hooks=[
                    GirderUploadToItem(outputId)
                ])
        return result.job

     # --- DNN myod1 model inference.  This is a classification model from FNLCR
     # --- that produces a probability of MYOD1 mutation
    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('perform classification through forward inferencing using a pretrained network')
        .param('imageId', 'The ID of the source, an Aperio .SVS image file.')
        .param('segmentId', 'The ID of the segmentation image, a PNG or TIFF image.')
         .param('statsId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def myod1(
            self, 
            imageId, 
            segmentId,
            statsId
    ):
        result = myod1.delay(
                #GirderFileId(imageId), 
                GirderFileIdAllowDirect(imageId), 
                #GirderFileId(segmentId), 
                GirderFileIdAllowDirect(segmentId), 
                girder_result_hooks=[
                    GirderUploadToItem(statsId),
                ])
        return result.job

     # --- DNN survivability model inference.  This is a classification model from FNLCR
     # --- that produces a classification of low-to-high risk for survivability prediction
    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('perform classification through forward inferencing using a pretrained network')
        .param('imageId', 'The ID of the source, an Aperio .SVS image file.')
        .param('segmentId', 'The ID of the segmentation image, a PNG or TIFF image.')
        .param('fastmode', 'a binary flag indicating user desire to return a faster, approximate solution')
        .param('statsId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def survivability(
            self, 
            imageId,
            segmentId,
            fastmode,
            statsId,

    ):
        result = survivability.delay(
                #GirderFileId(imageId), 
                GirderFileIdAllowDirect(imageId), 
                #GirderFileId(segmentId), 
                GirderFileIdAllowDirect(segmentId),
                fastmode, 
                girder_result_hooks=[
                    GirderUploadToItem(statsId),
                ])
        return result.job


     # --- return a cohort of data for use by the display algorithms
    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('return a cohort of data')
        .param('cohortName', 'the key to specify which cohort (e.g. "myod1" or "survivability")')
        .param('outnameId', 'The ID of the output item where the data file will be uploaded.')
 
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def cohort(
            self, 
            cohortName,
            outnameId
    ):
        result = cohort.delay(
                cohortName, 
                girder_result_hooks=[
                    GirderUploadToItem(outnameId),
                ])
        return result.job

    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('perform forward inferencing using a pretrained network')
        .param('imageId', 'The ID of the source, an Aperio .SVS image file.')
        .param('outputId', 'The ID of the output item where the output file will be uploaded.')
        .param('statsId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def infer_wsi_hpc( ## 128 Mem
            self,
            imageId, 
            outputId,
            statsId
    ):
        user = self.getCurrentUser()
        token = self.getCurrentToken()
        # Set up environmnent for infer wsi
        # if SlurmEnabled:
        location = Setting().get(slurmPluginSettings.SHARED_PARTITION)
        # passing env absolute path for re-running
        shared_partition_requirements_directory = os.path.join(location, 'env')
        taskName = 'rms_infer_wsi'
        env = os.path.join(shared_partition_requirements_directory, 'rms_env')
        # Use slurm model to create a slurFm job
        # Make handler as 'slurm_handler' instead on 'worker_handler' will force task to run on slurm cluster

        job = slurmModel().createJob(title='inferWSI_on_hpc', type='rms_infer',
                                     taskName=taskName,
                                     taskEntry='infer_wsi.py',
                                     modules=['torch/1.7.0'],
                                     env=env, mem=128,
                                     handler='slurm_handler', user=user)

        jobToken = Job().createJobToken(job)
        inputFile = File().load(imageId, user=user)
        # Use slurm util to setup task json schema
        inputs = {
            'input': slurmGirderInput.girderInputSpec(
                            inputFile, resourceType='file', token=token)
        }
        reference = json.dumps({'jobId': str(job['_id'])})
        pushItem = Item().load(outputId, level=AccessType.WRITE, user=self.getCurrentUser())
        pushStatsItem = Item().load(statsId, level=AccessType.WRITE, user=self.getCurrentUser())
        # Name can be anything because output will be found based on slurm job id
        outputs = {
            'seg': slurmGirderOutput.girderOutputSpec(pushItem, token,
                                                    parentType='item',
                                                    name='',
                                                    reference=reference),
            'stats': slurmGirderOutput.girderOutputSpec(pushStatsItem, token,
                                                    parentType='item',
                                                    name='',
                                                    reference=reference),
        }
        job['meta'] = {
            'creator': 'rms_infer',
            'task': 'rms_infer',
        }
        job['kwargs'] = {
            'inputs': inputs,
            'outputs': outputs,
            'jobInfo': slurmUtils.jobInfoSpec(job, jobToken),
            'auto_convert': True,
            'validate': True,
        }
        job['kwargs']['env'] = env
        job = Job().save(job)
        # Schedule to start slurm job which will trigger slurm handler to submit sbatch
        slurmModel().scheduleSlurm(job)

        return job

    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('perform classification through forward inferencing using a pretrained network')
        .param('imageId', 'The ID of the source, an Aperio .SVS image file.')
        .param('segmentId', 'The ID of the segmentation image, a PNG or TIFF image.')
         .param('statsId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def myod1_hpc(
            self, 
            imageId, 
            segmentId,
            statsId
    ):
        user = self.getCurrentUser()
        token = self.getCurrentToken()
        # Set up environmnent for infer wsi
        # if SlurmEnabled:
        location = Setting().get(slurmPluginSettings.SHARED_PARTITION)
        # passing env absolute path for re-running
        shared_partition_requirements_directory = os.path.join(location, 'env')
        taskName = 'rms_myod1'
        # share same env with infer_wsi
        env = os.path.join(shared_partition_requirements_directory, 'rms_myod1_env')
        
        job = slurmModel().createJob(title='myod1_on_hpc', type='rms_myod1',
                                     taskName=taskName,
                                     taskEntry='myod1.py',
                                     modules=['torch/1.7.0'],
                                     env=env,
                                     handler='slurm_handler', user=user)

        jobToken = Job().createJobToken(job)
        inputFile = File().load(imageId, user=user)
        sefFile = File().load(segmentId, user=user)
        # Use slurm util to setup task json schema
        inputs = {
            'input': slurmGirderInput.girderInputSpec(
                            inputFile, resourceType='file', token=token),
            'segmentation': slurmGirderInput.girderInputSpec(
                            sefFile, resourceType='file', token=token)
        }
        reference = json.dumps({'jobId': str(job['_id'])})
        pushStatsItem = Item().load(statsId, level=AccessType.WRITE, user=self.getCurrentUser())
        # Name can be anything because output will be found based on slurm job id
        outputs = {
            'stats': slurmGirderOutput.girderOutputSpec(pushStatsItem, token,
                                                    parentType='item',
                                                    name='',
                                                    reference=reference),
        }
        job['meta'] = {
            'creator': 'rms_infer',
            'task': 'rms_infer',
        }
        job['kwargs'] = {
            'inputs': inputs,
            'outputs': outputs,
            'jobInfo': slurmUtils.jobInfoSpec(job, jobToken),
            'auto_convert': True,
            'validate': True,
        }
        job['kwargs']['env'] = env
        job = Job().save(job)
        # Schedule to start slurm job which will trigger slurm handler to submit sbatch
        slurmModel().scheduleSlurm(job)

        return job

    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('perform classification through forward inferencing using a pretrained network')
        .param('imageId', 'The ID of the source, an Aperio .SVS image file.')
        .param('segmentId', 'The ID of the segmentation image, a PNG or TIFF image.')
        .param('statsId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def survivability_hpc(
            self, 
            imageId,
            segmentId,
            statsId
    ):
        user = self.getCurrentUser()
        token = self.getCurrentToken()
        # Set up environmnent for infer wsi
        # if SlurmEnabled:
        location = Setting().get(slurmPluginSettings.SHARED_PARTITION)
        # passing env absolute path for re-running
        shared_partition_requirements_directory = os.path.join(location, 'env')
        taskName = 'rms_survivability'
        # share same env with infer_wsi
        env = os.path.join(shared_partition_requirements_directory, 'rms_env')
        
        job = slurmModel().createJob(title='survivability_on_hpc', type='rms_survivability',
                                     taskName=taskName,
                                     taskEntry='survivability.py',
                                     modules=['torch/1.7.0'],
                                     env=env,
                                     handler='slurm_handler', user=user)

        jobToken = Job().createJobToken(job)
        inputFile = File().load(imageId, user=user)
        sefFile = File().load(segmentId, user=user)
        # Use slurm util to setup task json schema
        inputs = {
            'input': slurmGirderInput.girderInputSpec(
                            inputFile, resourceType='file', token=token),
            'segmentation': slurmGirderInput.girderInputSpec(
                            sefFile, resourceType='file', token=token)
        }
        reference = json.dumps({'jobId': str(job['_id'])})
        pushStatsItem = Item().load(statsId, level=AccessType.WRITE, user=self.getCurrentUser())
        # Name can be anything because output will be found based on slurm job id
        outputs = {
            'stats': slurmGirderOutput.girderOutputSpec(pushStatsItem, token,
                                                    parentType='item',
                                                    name='',
                                                    reference=reference),
        }
        job['meta'] = {
            'creator': 'rms_infer',
            'task': 'rms_infer',
        }
        job['kwargs'] = {
            'inputs': inputs,
            'outputs': outputs,
            'jobInfo': slurmUtils.jobInfoSpec(job, jobToken),
            'auto_convert': True,
            'validate': True,
        }
        job['kwargs']['env'] = env
        job = Job().save(job)
        # Schedule to start slurm job which will trigger slurm handler to submit sbatch
        slurmModel().scheduleSlurm(job)

        return job
