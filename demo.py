from pose import pipeline

# image inferencing
pipeline(
    "movenet", "data/run.png", show=True, save=True
)  # give path name in save or just boolean

exit()

# Optimize for crop region algorithm - video
pipeline("movenet", "./data/bike.mp4", show=True, save="./result/bike-inference.mp4")

# pipeline(
#     "movenet",
#     "https://www.youtube.com/watch?v=1VYhyppWTDc&ab_channel=GlobalCyclingNetwork",
#     show=True,
# )

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


# Manual video inference - bad result -> use pipeline API with crop region optimized
import cv2
from pose import AutoModel, load, plot

model = AutoModel.from_pretrained("movenet")

for frame in load(src="data/bike.mp4"):
    keypoints = model(frame)  # (1, 1, 17, 3)
    plot(frame, keypoints, conf_thres=0)
    cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
