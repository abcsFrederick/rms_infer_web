==========
rms_infer_web
==========

A turnkey Rhabdomyosarcoma web interface for pre-trained models developed by the Imaging
and Visualization Group at the Frederick National Lab for Cancer Research

Docker Installation 
-------------------
The easiest way to install this system locally is to use the companion repository to automatically build a docker
container.  See https://github.com/knowledgevis/rms_infer_docker.

Native Installation
------------
To Do the set up on a local system, building without using a docker container, please see the instructions below:

* Do this work with Python3
* Create two virtualenvs, the first virtualenv, called **girder**, is for girder, the segmentation, and myod1 algorithms. The second virtualenv, called **rms_venv**, is for the survivability app.
* Install mongo and rabbitmq

* In virtualenv **girder** run the following commands, it doesn't matter where you run them from:

.. code-block:: bash

    $ pip install --pre girder[plugins]
    $ girder build

* These commands need to be run in the **girder** virtualenv from specific locations.

.. code-block:: bash

    $ cd rms_infer_web/girder_worker_tasks    
    $ pip install -e .                     # install gw tasks for producer
    $ cd ../../rms_infer_web/girder_plugin
    $ pip install -e .                     # install girder plugin
    $ girder serve                         # start serving girder
    $ pip install --pre girder-worker
    $ cd rms_infer_web/girder_worker_tasks    
    $ pip install -e .                     # install gw tasks for consumer
    $ girder-worker                        # start girder-worker
    $ pip install torch==1.4.0
    $ pip install torchvision==0.5.0
    $ pip install efficientnet-pytorch==0.6.3
    $ pip install opencv-python
    $ pip install albumentations
    $ pip install scikit-image
    $ pip install segmentation_models_pytorch==0.1.0 --no-dependencies
    $ pip install timm==0.1.18  --no-dependencies
    $ pip install torchnet
    # install large_image for reading image formats the find-links helps this run fast
    $ pip install large_image[sources] --find-links https://girder.github.io/large_image_wheels 
    $ pip install pretrainedmodels
    $ pip install python-dotenv
 

* In virtualenv **rms_venv** run the following commands, it doesn't matter where you run them from:

.. code-block:: bash

        $  pip install girder_client
        $  pip install opencv-python
        $  pip install torch==1.7.1  
        $  pip install scikit-image
        $  pip install albumentations
        $  pip install segmentation_models_pytorch==0.1.3 --no-dependencies 
        $  pip install pretrainedmodels
        $  pip install torchvision==0.8.2 --no-dependencies
        $  pip install efficientnet-pytorch==0.6.3
        $  pip install timm==0.3.2
        $  pip install openslide-python


Features
--------

Installs several REST endpoints for uploading images, running forward inferencing jobs on pre-trained 
RMS segmentation, MYOD1 mutation, and survivability models.  The models were trained at the Frederick
National Lab for Cancer Research. 


TODO
----

* Need to cleanup the images uploaded and the outputs which are stored in Girder after each run.  
