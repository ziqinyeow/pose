from pose import AutoModel, load, viz


model = AutoModel.from_pretrained("movenet")
image = load("test/run.png", return_tensors="tf")
logits = model(image)
viz(image, logits)


# print(logits.shape)
