from pose import AutoModel, load, viz


model = AutoModel.from_pretrained("movenet")
image = load("test/bike.mp4", return_tensors="tf")
print(image.shape)

logits = model(image)
# viz(image, logits)


print(logits.shape, type(logits))
