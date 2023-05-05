#!/bin/bash

pip install -q mmdet
pip install -q -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpose"

pip install -r requirements.txt
# pip install -q -e .

# rtmpose
# mim install -q 'mmcls>=1.0.0rc5'