from pose import pipeline
import time


# print(Wiki.list_models())
start_time = time.time()
pipeline(
    "rtmpose-m",
    "./data/human-pose.jpg",
    save="output_torch.jpg",
    show=False,
)
end_time = time.time()
print("Inference done, time cost: {:.4f}s".format(end_time - start_time))
exit()
# pipeline("movenet", "./data/bike.mp4", show=True, side="right", save=False)
# pipeline("vitpose-b", "data/run.png", side="right", angle=True, show=True)
# pipeline("movenet", "./data/run.png", show=True, side="right", save=False)
# pipeline("vitpose-b", "data/bike.mp4", side="right", show=True)
# exit()

# image inferencing
# pipeline(
#     "movenet", "data/run.png", show=True, save=False
# )  # give path name in save or just boolean


# Optimize for crop region algorithm - video
# pipeline("movenet", "./data/", show=True, side="right", save=False)

pipeline(
    "movenet",
    "https://www.youtube.com/watch?v=1VYhyppWTDc&ab_channel=GlobalCyclingNetwork",
    show=True,
)
exit()

# Manual image inference
import cv2
from pose import AutoModel, load, plot, resize

frame = next(load(src="data/run.png"))  # (H, W, C)
frame = resize(frame)  # resize to squared frame

model = AutoModel.from_pretrained("movenet")
keypoints = model(frame)  # (1, 1, 17, 3)

plot(frame, keypoints, conf_thres=0)

cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# exit()


# Manual video inference - bad result -> use pipeline API with crop region optimized
import cv2
from pose import AutoModel, load, plot

model = AutoModel.from_pretrained("rtmpose-l")

for frame in load(src="data/bike.mp4"):
    keypoints = model(frame)  # (1, 1, 17, 3)
    plot(frame, keypoints, conf_thres=0)
    cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
