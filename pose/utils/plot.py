import cv2
import numpy as np


# fmt: off
coco = np.array([
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle"
    ])

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
# fmt: on


def plot(im: np.ndarray, kpts: np.ndarray, steps=3, side=None, conf_thres=0.5):
    # kpts : [1, numOfPose, 17, 3]
    # Plot the skeleton and keypoints for coco datatset 17 keypoints
    kpts = kpts.flatten()

    H, W, _ = im.shape

    # fmt: off
    skeleton = [[1, 0], [2, 0], [3, 1], [4, 2], [5, 7], [6, 8], 
                [7, 9], [8, 10], [11, 13], [12, 14], [13, 15], [14, 16],
                [5, 11], [6, 12], [11, 12], [5, 6], [2, 4], [3, 5], [4, 6]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    # fmt: on

    left_kpts = {0, 1, 3, 5, 7, 9, 11, 13, 15}
    right_kpts = {0, 2, 4, 6, 8, 10, 12, 14, 16}
    left_skeleton = {0, 2, 4, 6, 8, 10, 12}
    right_skeleton = {1, 3, 5, 7, 9, 11, 13}

    radius = 5
    num_kpts = len(kpts) // steps
    # print(num_kpts, im.shape)

    for kid in range(num_kpts):
        conf = kpts[steps * kid + 2]
        if (
            side == "right"
            and kid not in right_kpts
            or side == "left"
            and kid not in left_kpts
            or conf < conf_thres
        ):
            continue
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid + 1] * W, kpts[steps * kid + 0] * H
        # print(x_coord, y_coord)

        cv2.circle(
            im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1
        )

    for sk_id, sk in enumerate(skeleton):
        conf1 = kpts[(sk[0]) * steps + 2]
        conf2 = kpts[(sk[1]) * steps + 2]

        if (
            side == "right"
            and sk_id not in right_skeleton
            or side == "left"
            and kid not in left_skeleton
            or conf1 < conf_thres
            or conf2 < conf_thres
        ):
            continue
        r, g, b = pose_limb_color[sk_id]
        pos1 = (
            int(kpts[(sk[0]) * steps + 1] * W),
            int(kpts[(sk[0]) * steps + 0] * H),
        )
        pos2 = (
            int(kpts[(sk[1]) * steps + 1] * W),
            int(kpts[(sk[1]) * steps + 0] * H),
        )
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
