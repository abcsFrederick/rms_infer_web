#!/bin/bash

source /home/clisle/proj/nci/code/rms_venv/bin/activate
echo "image is" $1 
echo "segmentation is" $2 
echo "model is" $3 
echo python version is `python --version`
echo timm version is `pip freeze | grep timm`
python ./survive_subprocess.py $1 $2 $3
 
