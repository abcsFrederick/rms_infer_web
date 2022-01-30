#!/bin/bash

# initialize the python interpreter and stack for this method
source /rms_venv/bin/activate
#echo "image is" $1 
#echo "segmentation is" $2 
#echo "model is" $3 
#echo python version is `python --version`
#echo timm version is `pip freeze | grep timm`
python /rms_infer_infer/survive_subprocess.py $1 $2 $3
 
