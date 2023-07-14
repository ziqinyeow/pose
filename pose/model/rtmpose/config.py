MODEL = {
    "rtmpose-t": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth",
    "rtmpose-s": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth",
    "rtmpose-m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
    "rtmpose-l": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth",
}

CONFIG_PATH = {
    "rtmpose-t": "pose/model/rtmpose/conf/rtmpose-t_8xb256-420e_coco-256x192.py",
    "rtmpose-s": "pose/model/rtmpose/conf/rtmpose-s_8xb256-420e_coco-256x192.py",
    "rtmpose-m": "pose/model/rtmpose/conf/rtmpose-m_8xb256-420e_coco-256x192.py",
    "rtmpose-l": "pose/model/rtmpose/conf/rtmpose-l_8xb256-420e_coco-256x192.py",
}

DET_MODEL = {
    "rtmpose-n": "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
    "rtmpose-t": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    "rtmpose-s": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    "rtmpose-m": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
    "rtmpose-l": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
}

DET_CONFIG_PATH = {
    "rtmpose-n": "pose/model/rtmpose/conf/rtmdet_nano_320-8xb32_coco-person.py",
    "rtmpose-t": "pose/model/rtmpose/conf/rtmdet_tiny_8xb32-300e_coco.py",
    "rtmpose-s": "pose/model/rtmpose/conf/rtmdet_s_8xb32-300e_coco.py",
    "rtmpose-m": "pose/model/rtmpose/conf/rtmdet_m_8xb32-300e_coco.py",
    "rtmpose-l": "pose/model/rtmpose/conf/rtmdet_l_8xb32-300e_coco.py",
}


def CONFIG(model: str):
    # det = "rtmpose-n" if model in ["rtmpose-t", "rtmpose-s"] else "rtmpose-m"
    return {
        "backend": "onnx",
        "steps": 2,
        "pose_config": CONFIG_PATH[model],
        "pose_ckpt": MODEL[model],
        "det_config": DET_CONFIG_PATH[model],
        "det_ckpt": DET_MODEL[model],
    }
