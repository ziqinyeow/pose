from pose import AutoModel, load, viz

model = AutoModel.from_pretrained("movenet")
img = load("data/run.png", return_tensors="tf")
viz(img, model)

# vid = load("data/bike.mp4", return_tensors="tf")
# viz(vid, model) # create a .gif file of the inference video
