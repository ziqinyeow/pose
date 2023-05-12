import cv2
import numpy as np
from .angle import calculate_angle
from .transform import resize


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

POSE_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def plot(
    im: np.ndarray,
    kpts: np.ndarray,
    steps=3,
    side=None,
    show_angle=False,
    conf_thres=0.5,
):
    # kpts : [numOfPose, 17, 3]
    # Plot the skeleton and keypoints for coco datatset 17 keypoints
    number_of_poses = kpts.shape[0]
    kpts = kpts.reshape(number_of_poses, -1)

    H, W, _ = im.shape

    # fmt: off
    skeleton = [[1, 0], [2, 0], [3, 1], [4, 2], [5, 7], [6, 8], 
                [7, 9], [8, 10], [11, 13], [12, 14], [13, 15], [14, 16],
                [5, 11], [6, 12], [11, 12], [5, 6], [2, 4], [3, 5], [4, 6]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    
    angles = [
        [5, 7, 9], [6, 8, 10], # elbow (left, right)
        [7, 5, 11], [8, 6, 12], # shoulder
        [5, 11, 13], [6, 12, 14], # hip
        [11, 13, 15], [12, 14, 16] # knee
    ]
    
    distance = [
        # upper arm - shoulder to elbow
        [POSE_DICT['left_shoulder'], POSE_DICT['left_elbow']], # left
        [POSE_DICT['right_shoulder'], POSE_DICT['right_elbow']], # right
        
        # lower arm - elbow to wrist
        [POSE_DICT['left_elbow'], POSE_DICT['left_wrist']], # left
        [POSE_DICT['right_elbow'], POSE_DICT['right_wrist']], # right
        
        # upper leg - hip to knee
        [POSE_DICT['left_hip'], POSE_DICT['left_knee']], # left
        [POSE_DICT['right_hip'], POSE_DICT['right_knee']], # right
        
        # lower leg - knee to ankle
        [POSE_DICT['left_knee'], POSE_DICT['left_ankle']], # left
        [POSE_DICT['right_knee'], POSE_DICT['right_ankle']], # right
        
        # 
        # [POSE_DICT['']]
    ]
    # fmt: on

    left_kpts = {0, 1, 3, 5, 7, 9, 11, 13, 15}
    right_kpts = {0, 2, 4, 6, 8, 10, 12, 14, 16}
    left_skeleton = {0, 2, 4, 6, 8, 10, 12}
    right_skeleton = {1, 3, 5, 7, 9, 11, 13}

    radius = 5

    for kpt in kpts:
        num_kpts = len(kpts) // steps

        # plot keypoints - circle
        for kid in range(num_kpts):
            conf = kpt[steps * kid + 2] if steps == 3 else None
            if (
                side == "right"
                and kid not in right_kpts
                or side == "left"
                and kid not in left_kpts
                or conf is not None
                and (conf < conf_thres)
            ):
                continue
            r, g, b = pose_kpt_color[kid]
            x_coord, y_coord = kpt[steps * kid + 1], kpt[steps * kid + 0]

            if x_coord < 1:
                x_coord *= W
            if y_coord < 1:
                y_coord *= H

            cv2.circle(
                im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1
            )

        # plot skeleton joints - line
        for sk_id, sk in enumerate(skeleton):
            conf1 = kpt[(sk[0]) * steps + 2] if steps == 3 else None
            conf2 = kpt[(sk[1]) * steps + 2] if steps == 3 else None

            if (
                side == "right"
                and sk_id not in right_skeleton
                or side == "left"
                and kid not in left_skeleton
                or conf1 is not None
                and conf1 < conf_thres
                or conf2 is not None
                and conf2 < conf_thres
            ):
                continue
            r, g, b = pose_limb_color[sk_id]

            x_coord1, y_coord1 = kpt[(sk[0]) * steps + 1], kpt[(sk[0]) * steps + 0]
            x_coord2, y_coord2 = kpt[(sk[1]) * steps + 1], kpt[(sk[1]) * steps + 0]

            if x_coord1 < 1:
                x_coord1 = int(x_coord1 * W)
                y_coord1 = int(y_coord1 * H)

            if x_coord2 < 1:
                x_coord2 = int(x_coord2 * W)
                y_coord2 = int(y_coord2 * H)

            pos1 = (int(x_coord1), int(y_coord1))
            pos2 = (int(x_coord2), int(y_coord2))

            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

        # plot angle - text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        color = (255, 255, 255)
        thickness = 2
        if show_angle:
            for i, (j1, j2, j3) in enumerate(angles):
                if (
                    side == "right"
                    and i in [0, 2, 4, 6]
                    or side == "left"
                    and i in [1, 3, 5, 7]
                ):
                    continue
                x1_coord, y1_coord = kpt[steps * j1 + 1], kpt[steps * j1 + 0]
                x2_coord, y2_coord = kpt[steps * j2 + 1], kpt[steps * j2 + 0]
                x3_coord, y3_coord = kpt[steps * j3 + 1], kpt[steps * j3 + 0]

                if x1_coord < 1:
                    x1_coord = int(x1_coord * W)
                    y1_coord = int(y1_coord * H)

                if x2_coord < 1:
                    x2_coord = int(x2_coord * W)
                    y2_coord = int(y2_coord * H)

                if x3_coord < 1:
                    x3_coord = int(x3_coord * W)
                    y3_coord = int(y3_coord * H)

                ang = calculate_angle(
                    a=[x1_coord, y1_coord],
                    b=[x2_coord, y2_coord],
                    c=[x3_coord, y3_coord],
                )

                cv2.putText(
                    im,
                    str(round(ang, 2)),
                    (int(x2_coord), int(y2_coord)),
                    font,
                    scale,
                    color,
                    thickness,
                )
