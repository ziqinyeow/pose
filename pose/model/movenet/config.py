from ..type import BACKEND

MODEL = {
    # default
    "movenet": "singlepose_thunder.tflite",
    # lightning - singlepose
    "movenet.lightning.single": "singlepose_lightning.tflite",
    "movenet.lightning.single.int8": "singlepose_lightning_int8.tflite",
    "movenet.lightning.single.fp16": "singlepose_lightning_fp16.tflite",
    # lightning - multipose - not supported yet
    # "movenet.lightning.multi.fp16": "multipose_lightning_fp16.tflite",
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
        "model_path": MODEL[model],
        "resize": True,
        "crop_region": True,
    }
