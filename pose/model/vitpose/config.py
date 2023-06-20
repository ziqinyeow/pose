import importlib
from ..type import BACKEND

MODEL = {"vitpose-b": "vitpose-b.pth",
         "vitpose-l": "vitpose-l.pth"}

MODEL_CONFIG = {"vitpose-b": "vitpose_base_coco_256x192",
                "vitpose-l": "vitpose_large_coco_256x192"}

DEFAULT_BACKEND: BACKEND = "pt"


def CONFIG(model, backend=None):
    backend = backend or DEFAULT_BACKEND

    module = importlib.import_module(
        f"pose.model.vitpose.configs.{MODEL_CONFIG[model]}"
    )
    model_cfg = getattr(module, "model")
    data_cfg = getattr(module, "data_cfg")

    return {
        "backend": backend,
        # "input_size": 192 if "lightning" in model else 256,
        "model_path": MODEL[model],
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
    }
