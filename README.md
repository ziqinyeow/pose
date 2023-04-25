# 🚀 A Pose Estimation Toolbox To The Moon

A pose estimation repo heavily inspired by HuggingFace SDK, building for only pose estimation inference and post-processing.

## Current Supported Model for Inference

1. Movenet (TFLite)
2. Way more to come ...

## Roadmap

Still in rapid development.

## Guide

```python
python demo.py
```

```python
from pose import AutoModel, load, viz

model = AutoModel.from_pretrained("movenet")

# for image data
img = load("data/run.png", return_tensors="tf")
viz(img, model)

# for video data
# vid = load("data/bike.mp4", return_tensors="tf")
# viz(vid, model) # create a .gif file of the inference video

```
