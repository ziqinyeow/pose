MODEL = {
    "rsnpose-18": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn18_8xb32-210e_coco-256x192-9049ed09_20221013.pth",
    "rsnpose-50": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn50_8xb32-210e_coco-256x192-c35901d5_20221013.pth",
    "2xrsnpose-50": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_2xrsn50_8xb32-210e_coco-256x192-9ede341e_20221013.pth",
    "3xrsnpose-50": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_3xrsn50_8xb32-210e_coco-256x192-c3e3c4fe_20221013.pth"
}

CONFIG_PATH = {
    "rsnpose-18": "pose/model/rsnpose/conf/td-hm_rsn18_8xb32-210e_coco-256x192.py",
    "rsnpose-50": "pose/model/rsnpose/conf/td-hm_rsn50_8xb32-210e_coco-256x192.py",
    "2xrsnpose-50": "pose/model/rsnpose/conf/td-hm_2xrsn50_8xb32-210e_coco-256x192.py",
    "3xrsnpose-50": "pose/model/rsnpose/conf/td-hm_3xrsn50_8xb32-210e_coco-256x192.py"
}

DET_MODEL = {
    "rsnpose-18": "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
    "rsnpose-50": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    "2xrsnpose-50": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    "3xrsnpose-50": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
}

DET_CONFIG_PATH = {
    "rsnpose-18": "pose/model/rtmpose/conf/rtmdet_nano_320-8xb32_coco-person.py",
    "rsnpose-50": "pose/model/rtmpose/conf/rtmdet_tiny_8xb32-300e_coco.py",
    "2xrsnpose-50": "pose/model/rtmpose/conf/rtmdet_s_8xb32-300e_coco.py",
    "3xrsnpose-50": "pose/model/rtmpose/conf/rtmdet_m_8xb32-300e_coco.py",
}

def CONFIG(model: str):
    return {
        "backend": "pt",
        "steps": 2,
        "pose_config": CONFIG_PATH[model],
        "pose_ckpt": MODEL[model],
        "det_config": DET_CONFIG_PATH[model],
        "det_ckpt": DET_MODEL[model],
    }