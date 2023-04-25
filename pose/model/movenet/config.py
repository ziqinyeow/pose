from ..type import BACKEND

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
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

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_COLOR = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}

MODEL = {
    # default
    "movenet": "singlepose_thunder.tflite",
    # lightning - singlepose
    "movenet.lightning.single": "singlepose_lightning.tflite",
    "movenet.lightning.single.int8": "singlepose_lightning_int8.tflite",
    "movenet.lightning.single.fp16": "singlepose_lightning_fp16.tflite",
    # lightning - multipose
    "movenet.lightning.multi.fp16": "multipose_lightning_fp16.tflite",
    # thunder - singlepose
    "movenet.thunder.single": "singlepose_thunder.tflite",
    "movenet.thunder.single.int8": "singlepose_thunder_int8.tflite",
    "movenet.thunder.single.fp16": "singlepose_thunder_fp16.tflite",
}

DEFAULT_BACKEND: BACKEND = "tflite"


def CONFIG(model, backend=None):
    backend = backend or DEFAULT_BACKEND
    return {
        "backend": backend,
        "input_size": 192 if "lightning" in model else 256,
        "model_path": MODEL[model] if backend == "tflite" else None,
        "keypoint_dict": KEYPOINT_DICT,
        "edge_color_dict": KEYPOINT_EDGE_COLOR,
    }
